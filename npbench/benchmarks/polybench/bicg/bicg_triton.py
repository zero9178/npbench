import itertools

import torch
import triton
import triton.language as tl


def generate_config0():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_N": b, "BLOCK_SIZE_K": k}, num_warps=w)
        for b, k, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if b != 128 or k != 128
    ]


@triton.autotune(configs=generate_config0(), key=["K", "N"], cache_results=True)
@triton.jit()
def _kernel0(
    A,  # (K, N)
    X,  # (K, ),
    out,  # (N, ),
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    row = (i * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
    column = (j * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    a = tl.load(
        A + N * row[:, None] + column[None, :],
        mask=(column[None, :] < N) & (row[:, None] < K),
    )
    X = tl.load(X + row, mask=row < K, other=0.0)

    # Perform the reduction of the K dimension. A vector corresponding to an N tile remains.
    a_sum = tl.sum(a * X[:, None], axis=0)
    tl.atomic_add(out + column, a_sum, sem="release")


def generate_config1():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": b, "BLOCK_SIZE_K": k}, num_warps=w)
        for b, k, w in itertools.product([8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8])
        if b != 128 or k != 128
    ]


@triton.autotune(configs=generate_config1(), key=["M", "K"], cache_results=True)
@triton.jit()
def _kernel1(
    A,  # (M, K)
    X,  # (K, ),
    out,  # (M, ),
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    zero = tl.zeros((BLOCK_SIZE_K,), out.dtype.element_ty)
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    row = (i * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    column = (j * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(
        A + K * row[:, None] + column[None, :],
        (column[None, :] < K) & (row[:, None] < M),
    )
    X = tl.load(X + column, mask=column < K, other=zero)

    # Perform the reduction of the K dimension. A vector corresponding to an M tile remains.
    a_sum = tl.sum(a * X[None, :], axis=1)
    tl.atomic_add(out + row, a_sum, sem="release")


def kernel(A: torch.Tensor, p: torch.Tensor, r: torch.Tensor):
    # return r @ A, A @ p
    out0 = torch.zeros((A.shape[1],), dtype=A.dtype)
    out1 = torch.zeros((A.shape[0],), dtype=A.dtype)

    grid0 = lambda meta: (
        triton.cdiv(A.shape[0], meta["BLOCK_SIZE_K"]),
        triton.cdiv(A.shape[1], meta["BLOCK_SIZE_N"]),
    )
    grid1 = lambda meta: (
        triton.cdiv(A.shape[0], meta["BLOCK_SIZE_M"]),
        triton.cdiv(A.shape[1], meta["BLOCK_SIZE_K"]),
    )
    _kernel0[grid0](A, r, out0, A.shape[0], A.shape[1])
    _kernel1[grid1](A, p, out1, A.shape[0], A.shape[1])
    return out0, out1
