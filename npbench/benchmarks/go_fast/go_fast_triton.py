import torch
import triton
import triton.language as tl
import itertools

def generate_config():
    base = [
        (64, 4, 3),
        (128, 4, 3),
        (32, 4, 3),
        (128, 8, 4),  
    ]
    return [triton.Config(
                kwargs={"BLOCK_SIZE_N": n},
                num_warps=w, num_stages=s)
            for (n, w, s) in base]

    trace_out += tl.sum(acc, axis = 0)

    A = A + trace_out
    return A

@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def _trace_of_matrix(A, N, trace, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr):

    pid_n = tl.program_id(axis=0)
    identity_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_identity = identity_offs < N

    # go_fast(a):
    #     trace = 0.0
    #     for i in range(N):
    #         trace += np.tanh(a[i, i])
    #     return a + trace

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=DTYPE)
    a_diag = tl.load(A + identity_offs * N + identity_offs, mask=mask_identity, other=0.0)

    # Compute tanh
    val1 = tl.exp(a_diag)
    val2 = tl.exp(-a_diag)
    acc += (val1 - val2) / (val1 + val2)
    sum = tl.sum(acc)
    tl.atomic_add(trace, sum)


@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def _add_trace_to_matrix(A, N, trace, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr):

    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    rows = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cols = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < N
    col_mask = cols < N

    a_matrix = tl.load(A + rows * N + cols, mask=row_mask & col_mask, other=0.0)
    tr =  tl.load(trace)
    res = a_matrix + tr
    tl.store(A + rows * N + cols, res, mask=row_mask & col_mask)

# expected the name of the kernel to be "go_fast" for some reason, error otherwise
def go_fast(A):
    M, N = A.shape
    assert M == N, "Matrix must be square."

    dtype = A.dtype
    assert dtype in (torch.float32, torch.float64)
    DTYPE = tl.float32 if dtype == torch.float32 else tl.float64

    grid_1d = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    grid_2d = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                            triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    trace = torch.zeros(1, dtype=A.dtype, device=A.device)
    _trace_of_matrix[grid_1d](A, N, trace, DTYPE)
    _add_trace_to_matrix[grid_2d](A, N, trace, DTYPE)



