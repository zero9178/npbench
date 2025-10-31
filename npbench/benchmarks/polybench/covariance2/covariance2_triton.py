import torch
import triton

from npbench.benchmarks.polybench.covariance.covariance_triton import _kernel_mean, _kernel_center
from npbench.infrastructure.triton_utilities import matmul


def kernel(M, float_n, data):
      data = data.T  # Now shape is (M, N) instead of (N, M)
      grid = lambda meta: (
          triton.cdiv(M_transposed, meta["BLOCK_SIZE_M"]),
          triton.cdiv(N_transposed, meta["BLOCK_SIZE_N"]),
      )
      M_transposed, N_transposed = data.shape  # M_transposed = M, N_transposed = N

      mean = torch.zeros((N_transposed,), dtype=data.dtype)
      _kernel_mean[grid](data, M_transposed, N_transposed, mean)
      _kernel_center[grid](data, mean, M_transposed, N_transposed)

      return matmul(data, data.T) / (float_n - 1.0)  # data @ data.T instead of data.T @ data
