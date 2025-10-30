import torch
import triton
import triton.language as tl
from npbench.infrastructure.triton_utilities import get_2d_tile_offsets, matmul

"""
Similarly to the correlation kernel, there is a significantly more efficient
algorithm with a single matrix multiplication instead of a loop:

mean = np.mean(data, axis=0)
data -= mean
cov = (data.T @ data) / (float_n - 1.0)
"""

@triton.jit
def _kernel_mean(
  data,
  M,
  N,
  out_mean,
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    tile, mask, rows, columns = get_2d_tile_offsets(
        x=j * BLOCK_SIZE_N,
        y=i * BLOCK_SIZE_M,
        tile_width=BLOCK_SIZE_N,
        tile_height=BLOCK_SIZE_M,
        matrix_width=N,
        matrix_height=M,
    )
    values = tl.load(data+tile, mask)
    row_sum = tl.sum(values, axis=0)/M
    tl.atomic_add(out_mean + columns, row_sum, mask=columns < N)

@triton.jit
def _kernel_center(
  data,
  mean,
  M,
  N,
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
):
    i=tl.program_id(axis=0)
    j=tl.program_id(axis=1)

    tile, mask, rows, columns = get_2d_tile_offsets(
        x=j * BLOCK_SIZE_N,
        y=i * BLOCK_SIZE_M,
        tile_width=BLOCK_SIZE_N,
        tile_height=BLOCK_SIZE_M,
        matrix_width=N,
        matrix_height=M,
    )

    values = tl.load(data + tile, mask)
    means = tl.load(mean + columns, mask=columns < N)
    tl.store(data + tile, values - means, mask)


def kernel(M, float_n, data:torch.Tensor):
    M, N = data.shape
    mean = torch.zeros((N,), dtype=data.dtype)

    grid_mean = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_M = tl.constexpr(32) # for now, no autotuning needed. speedup is already significant
    BLOCK_SIZE_N = tl.constexpr(64)

    _kernel_mean[grid_mean](data, M, N, mean, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M)

    grid_center = grid_mean
    _kernel_center[grid_center](data, mean, M, N, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M)

    return matmul(data.T, data)/ (float_n - 1.0)

