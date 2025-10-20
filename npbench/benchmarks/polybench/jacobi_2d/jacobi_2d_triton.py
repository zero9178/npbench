# 

import triton
import triton.language as tl
import torch

@triton.jit
def jacobi2d_step(src_ptr, dst_ptr,
                  N: tl.int32,
                  stride0: tl.int32, stride1: tl.int32,
                  BLOCK: tl.constexpr):

    pid_x = tl.program_id(0)  # tiles along rows (i)
    pid_y = tl.program_id(1)  # tiles along cols (j)

    num_x = tl.num_programs(axis=0)
    num_y = tl.num_programs(axis=1)

    # Compute global indices of the block
    ii = pid_x * BLOCK + tl.arange(0, BLOCK)[:, None]   # (BLOCK, 1) - row vector
    jj = pid_y * BLOCK + tl.arange(0, BLOCK)[None, :]   # (1, BLOCK) - col vector

    # work only on interior: i in [1, N-2], j in [1, N-2]
    i = ii + 1
    j = jj + 1
    in_bounds = (i < N - 1) & (j < N - 1)

    base = i * stride0 + j * stride1

    c  = tl.load(src_ptr + base, mask=in_bounds, other=0)
    l  = tl.load(src_ptr + i * stride0 + (j-1) * stride1, mask=in_bounds, other=0)
    r  = tl.load(src_ptr + i * stride0 + (j+1) * stride1, mask=in_bounds, other=0)
    u  = tl.load(src_ptr + (i-1) * stride0 + j * stride1, mask=in_bounds, other=0)
    d  = tl.load(src_ptr + (i+1) * stride0 + j * stride1, mask=in_bounds, other=0)

    out = 0.2 * (c + l + r + u + d)
    tl.store(dst_ptr + base, out, mask=in_bounds)


def kernel(TSTEPS: int, A: torch.Tensor, B: torch.Tensor, block: int = 128):
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1]
    # Force dtype + contiguity to match reference
    if A.dtype != torch.float64:
        A = A.to(torch.float64)
    if B.dtype != torch.float64:
        B = B.to(torch.float64)
    A = A.contiguous()
    B = B.contiguous()

    N = A.shape[0]

    # Triton expects strides in elements, not bytes
    s0, s1 = A.stride()  # row-major: (N, 1) for contiguous
    grid = (triton.cdiv(N-2, block), triton.cdiv(N-2, block))

    for _ in range(TSTEPS-1):
        jacobi2d_step[grid](
            A, B, N, s0, s1,
            BLOCK=block
        )
        jacobi2d_step[grid](
            B, A, N, s0, s1,
            BLOCK=block
        )
