import torch
from npbench.infrastructure.triton_utilities import matmul

def kernel(A: torch.Tensor, p: torch.Tensor, r: torch.Tensor):
    r = r.reshape(1, -1)
    p = p.reshape(-1, 1)
    return matmul(r, A).squeeze(), matmul(A, p).squeeze()
