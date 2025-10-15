import torch
from npbench.infrastructure.triton_utilities import matmul

def kernel(A: torch.Tensor, x: torch.Tensor):
    A_T = A.t()
    res = matmul(A_T, A)
    return matmul(res, x)