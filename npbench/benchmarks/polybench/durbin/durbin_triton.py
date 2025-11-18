import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({
            "N_BLOCK_SIZE": bs,
        }, num_warps=nw)
        for bs in [1024, 2048, 4096, 8192]
        for nw in [1, 2, 4, 8]
    ], key=["N"]
)
@triton.jit
def durbin_kernel(
    y_ptr,
    r_orig_ptr,
    N,
    N_BLOCK_SIZE: tl.constexpr, 
):
    
    r0 = tl.load(r_orig_ptr + tl.program_id(0) * tl.program_id(1) * 0) 
    
    alpha = -r0
    
    beta = tl.cast(1.0, tl.float64)
    
    tl.store(y_ptr, alpha)

    k = tl.constexpr(1)
    while k < N:
        beta_new = beta * (1.0 - alpha * alpha)
        
        r_k = tl.load(r_orig_ptr + k)

        dot_product_sum = tl.cast(0.0, tl.float64)
        
        j_block = tl.arange(0, N_BLOCK_SIZE)
        
        for j_start in range(0, k, N_BLOCK_SIZE):
            j = j_start + j_block         
            j_rev = (k - 1) - j           
            
            mask = j < k
            
            r_orig_vec = tl.load(r_orig_ptr + j, mask=mask, other=0.0)
            y_rev_vec = tl.load(y_ptr + j_rev, mask=mask, other=0.0)
            
            dot_product_sum += tl.sum(r_orig_vec * y_rev_vec, axis=0)
            
        numerator = r_k + dot_product_sum
            
        alpha_new = -numerator / beta_new
        for j_start in range(0, k, N_BLOCK_SIZE):
            j = j_start + j_block
            j_rev = (k - 1) - j
            
            mask = j < k
            
            y_old = tl.load(y_ptr + j, mask=mask, other=0.0)
            y_rev = tl.load(y_ptr + j_rev, mask=mask, other=0.0)
            y_new = y_old + alpha_new * y_rev
            tl.store(y_ptr + j, y_new, mask=mask)
            
        tl.store(y_ptr + k, alpha_new)

        alpha = alpha_new
        beta = beta_new
        k += 1

def kernel(r: torch.Tensor):
    N = r.shape[0]
    y_out = torch.empty_like(r, dtype=torch.float64, device=r.device)
    
    durbin_kernel[(1,)](
        y_out,
        r,
        N,
    )