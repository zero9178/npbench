import torch
from npbench.infrastructure.triton_utilities import matmul

def kernel(A: torch.Tensor, x: torch.Tensor):
    x_col = x.view(-1, 1).contiguous()
    Ax = matmul(A, x_col)
    A_T = A.t().contiguous()
    return matmul(A_T, Ax).t()
