import torch

from npbench.infrastructure.triton_utilities import mat_vec_mul

def kernel(A: torch.Tensor, x: torch.Tensor):
    Ax = torch.zeros((A.shape[0],), dtype=A.dtype)
    mat_vec_mul(A, x, Ax)
    A_T = A.t().contiguous()
    res = torch.zeros((A.shape[1],), dtype=A.dtype)
    mat_vec_mul(A_T, Ax, res)
    return res
