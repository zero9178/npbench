import itertools

import torch
import triton
import triton.language as tl

from npbench.infrastructure.triton_utilities import matmul, get_2d_tile_offsets

def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if m != 128 or n != 128
    ]

@triton.autotune(configs=generate_config(), key=["M", "N"], cache_results=True)
@triton.jit()
def _kernel(
            A,  # (M, N)
            X,  # (N, ),
            out,  # (M,),
            M: tl.constexpr, N: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
            ):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    row = (i * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    column = (j * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)

    a = tl.load(
        A + N * row[:, None] + column[None, :],
        mask=(column[None, :] < N) & (row[:, None] < M))
    x = tl.load(X + column, mask=column < N, other=0.0)

    x_sum = tl.sum(a * x[None, :], axis=1)
    tl.atomic_add(out + row, x_sum, sem="release")

def _mat_vec_mul(
            A,  # (M, N)
            X,  # (N, ),
            out,  # (M,)
            ):

    M, N = A.shape
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    _kernel[grid](A, X, out, M, N)


def kernel(A: torch.Tensor, x: torch.Tensor):
    Ax = torch.zeros((A.shape[0],), dtype=A.dtype)
    _mat_vec_mul(A, x, Ax)
    A_T = A.t().contiguous()
    res = torch.zeros((A.shape[1],), dtype=A.dtype)
    _mat_vec_mul(A_T, Ax, res)
    return res
