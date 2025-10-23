import itertools
import torch
import triton
import triton.language as tl
from npbench.infrastructure.triton_utilities import matmul


"""
SOLUTION 1

python3 run_benchmark.py -b k2mm -f triton
***** Testing Triton with k2mm on the S dataset, datatype default *****
NumPy - default - validation: 43ms
Triton - default - first/validation: 29657ms
Triton - default - default - validation: SUCCESS
Triton - default - median: 107ms
"""
def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if m != 128 or n != 128
    ]

@triton.autotune(configs=generate_config(), key=["M", "N"], cache_results=True)
@triton.jit
def _kernel(
    R_ptr, D_ptr,
    M: tl.int32, N: tl.int32,
    stride_rm: tl.int32, stride_rn: tl.int32,
    stride_dm: tl.int32, stride_dn: tl.int32,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    r_ptrs = R_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    d_ptrs = D_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn

    r = tl.load(r_ptrs, mask=mask)
    d = tl.load(d_ptrs, mask=mask)

    out = alpha * r + beta * d
    tl.store(d_ptrs, out, mask=mask)

def kernel(alpha: float, beta: float, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
    T = matmul(A, B)
    res = matmul(T, C)

    M, N = D.shape
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    _kernel[grid](
        res, D,
        M, N,
        res.stride(0), res.stride(1),
        D.stride(0), D.stride(1),
        alpha, beta,
    )