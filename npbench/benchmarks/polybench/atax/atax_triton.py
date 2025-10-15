import torch
from npbench.infrastructure.triton_utilities import matmul

def kernel(A: torch.Tensor, x: torch.Tensor):
    A_T = A.t().contiguous()
    x = x.view(-1, 1).t().contiguous()
    res = matmul(A_T, A)
    return matmul(x, res)
