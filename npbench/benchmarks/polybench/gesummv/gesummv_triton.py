import itertools

import torch
import triton
import triton.language as tl


def generate_config():
    """
    Generates many config instances for the purpose of auto-tuning.
    'num_warps' is especially useful for performance when reduction is involved as it may enable or disable certain
    cross-warp optimizations.
    """
    return [triton.Config(kwargs={'BLOCK_SIZE_N': b, 'BLOCK_SIZE_K': k}, num_warps=w) for b, k, w in
            itertools.product([8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8])
            if b != 128 or k != 128]


@triton.autotune(configs=generate_config(),
                 key=['N'],
                 cache_results=True
                 )
@triton.jit()
def _kernel(alpha, beta,
            A,  # (N, N)
            B,  # (N, N)
            X,  # (N, ),
            out,  # (N, ),
            N: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            ):
    zero = tl.zeros((BLOCK_SIZE_K,), out.dtype.element_ty)
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    row = (i * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    row = tl.expand_dims(row, axis=1)

    column = (j * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
    column = tl.expand_dims(column, axis=0)
    a = tl.load(A + N * row + column, (column < N) & (row < N))
    b = tl.load(B + N * row + column, (column < N) & (row < N))

    column = (j * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
    x = tl.load(X + column, mask=column < N, other=zero)
    x = tl.expand_dims(x, axis=0)

    # Note: use tl.dot when implementing for anything that isn't fp64.
    # Perform the reduction of the K dimension. A vector corresponding to an N tile remains.
    a_sum = tl.sum(a * x, axis=1)
    b_sum = tl.sum(b * x, axis=1)

    value = alpha * a_sum + beta * b_sum
    tl.atomic_add(out + tl.reshape(row, (BLOCK_SIZE_N,)), value, sem="release")


def kernel(alpha, beta,
           A,  # (N, N)
           B,  # (N, N)
           x  # (N, )
           ):
    """
    Triton implementation of:
        return alpha * A @ x + beta * B @ x

    Note that these are two simultaneous matrix-vector multiplies.
    The implementation uses a tiling strategy that both tiles the rows of the matrix (size N) and the columns of the
    matrix and vector simultaneously, hereon called the K dimension.
    The K dimension is the dimension being reduced (i.e. added up and removed by the dot product).

    We parallelize both over K and N. When parallelizing over K we must use an atomic add, as multiple threads will
    accumulate into the same result vector for every K tile.
    """

    # Note: Needs to be zero initialized as the kernel accumulates into the triton.
    out = torch.zeros_like(x)

    N = x.shape[0]
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(N, meta['BLOCK_SIZE_K']))
    _kernel[grid](alpha, beta, A, B, x, out, N)
    return out
