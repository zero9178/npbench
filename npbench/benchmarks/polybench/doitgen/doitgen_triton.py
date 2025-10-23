import torch
import triton
import triton.language as tl


@triton.jit
def _doitgen_kernel(
    A_ptr,  # (NR, NQ, NP)
    C4_ptr,  # (NP, NP)
    NR,
    NQ,
    NP,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Each program handles one (r, q) pair
    pid = tl.program_id(axis=0)

    # Convert linear program ID to (r, q) indices
    r = pid // NQ
    q = pid % NQ

    # Boundary check
    if r >= NR or q >= NQ:
        return

    # For this (r, q), we need to compute A[r, q, :] @ C4
    # A[r, q, :] has shape (NP,)
    # C4 has shape (NP, NP)
    # Result has shape (NP,)

    # Base pointer for A[r, q, :]
    a_base = A_ptr + r * NQ * NP + q * NP

    # Process output in blocks
    for p_start in range(0, NP, BLOCK_SIZE):
        # Output indices we're computing
        p_offs = p_start + tl.arange(0, BLOCK_SIZE)
        p_mask = p_offs < NP

        # Accumulator for this block of outputs (use fp32 for computation)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Compute dot product: sum over k of A[r,q,k] * C4[k,p]
        for k in range(NP):
            # Load A[r, q, k] (scalar)
            a_val = tl.load(a_base + k)
            a_val = tl.cast(a_val, tl.float32)

            # Load C4[k, p_offs] (vector)
            c4_ptr_k = C4_ptr + k * NP + p_offs
            c4_vals = tl.load(c4_ptr_k, mask=p_mask, other=0.0)
            c4_vals = tl.cast(c4_vals, tl.float32)

            # Accumulate
            acc += a_val * c4_vals

        # Store result back to A[r, q, p_offs]
        result_ptr = a_base + p_offs
        tl.store(result_ptr, tl.cast(acc, DTYPE), mask=p_mask)


def kernel(NR, NQ, NP, A: torch.Tensor, C4: torch.Tensor):
    """
    Performs the doitgen operation: A[r, q, :] = A[r, q, :] @ C4 for all r, q

    Args:
        NR: First dimension of A
        NQ: Second dimension of A
        NP: Third dimension of A (and both dimensions of C4)
        A: Input/output tensor of shape (NR, NQ, NP)
        C4: Input matrix of shape (NP, NP)
    """
    # Ensure tensors are contiguous
    A_c = A.contiguous()
    C4_c = C4.contiguous()

    # Determine dtype
    dtype = A.dtype
    if dtype == torch.float32:
        DTYPE = tl.float32
    else:  # float64
        DTYPE = tl.float64

    # Grid: one program per (r, q) pair
    grid = (NR * NQ,)

    # Block size for processing output dimension
    BLOCK_SIZE = 64

    # Launch kernel
    _doitgen_kernel[grid](
        A_c,
        C4_c,
        NR,
        NQ,
        NP,
        BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
        DTYPE=tl.constexpr(DTYPE),
    )
