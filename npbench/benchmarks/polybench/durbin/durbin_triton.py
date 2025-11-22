import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE": bs,
        }, num_warps=nw)
        for bs in [1024, 2048, 4096, 8192]
        for nw in [1, 2, 4, 8]
    ], key=["N"]
)
@triton.jit
def durbin_kernel(
    y_ptr,
    r_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, 
):
    alpha = -tl.load(r_ptr)
    beta = tl.cast(1.0, tl.float64)
    tl.store(y_ptr, alpha)

    j_block = tl.arange(0, BLOCK_SIZE)
    for k in tl.range(1, N):
        beta = beta * (1.0 - alpha * alpha)
        r_k = tl.load(r_ptr + k)

        dot_product_sum = tl.cast(0.0, tl.float64)
        for j_start in tl.range(0, k, BLOCK_SIZE):
            j = j_start + j_block         
            j_rev = (k - 1) - j           
            
            mask = j < k
            r_vec = tl.load(r_ptr + j, mask=mask, other=0.0)
            y_rev_vec = tl.load(y_ptr + j_rev, mask=mask, other=0.0)
            dot_product_sum += tl.sum(r_vec * y_rev_vec, axis=0)
            
        alpha = -(r_k + dot_product_sum) / beta
        for j_start in tl.range(0, k, BLOCK_SIZE):
            j = j_start + j_block
            j_rev = (k - 1) - j
            
            mask = j < k
            
            y_old = tl.load(y_ptr + j, mask=mask, other=0.0)
            y_rev_vec = tl.load(y_ptr + j_rev, mask=mask, other=0.0)
            y_new = y_old + alpha * y_rev_vec
            tl.store(y_ptr + j, y_new, mask=mask)
            
        tl.store(y_ptr + k, alpha)

def kernel(r: torch.Tensor):
    N = r.shape[0]
    y = torch.empty_like(r, dtype=torch.float64, device=r.device)
    
    durbin_kernel[(1,)](
        y,
        r,
        N,
    )
    return y