import itertools

import torch
import triton
import triton.language as tl



def generate_config():
    """
    Generates many config instances for the purpose of auto-tuning.
    'num_warps' is especially useful for performance when reduction is involved as it may enable or disable certain
    cross-warp optimizations.
    """
    return [triton.Config(kwargs={'BLOCK_SIZE': b}, num_warps=w) for b, w in
            itertools.product([8, 16, 32, 64], [1, 2, 4, 8])]


@triton.autotune(configs=generate_config(),
                 key=[],
                 cache_results=True
                 )
@triton.jit
def _kernel_durbin_iteration_cdot(
        r_flipped_ptr,  # pointer to flipped r
        y_ptr,  # pointer to y (read and write)
        out_dot_ptr,
        k,  # scalar: current iteration
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < k

    r_vals = tl.load(r_flipped_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    partial_dot = tl.sum(r_vals * y_vals)

    tl.atomic_add(out_dot_ptr, partial_dot)

@triton.autotune(configs=generate_config(),
                 key=[],
                 cache_results=True
                 )
@triton.jit
def _kernel_durbin_iteration_update_uptok(
      y_ptr,              # pointer to y (read and write)
      alpha,              # scalar: the alpha value
      k,                  # scalar: how many elements to update
      BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < k

    y_vals = tl.load(y_ptr+offsets, mask=mask, other=0.0) #id for add is 0
    y_flipped_vals = tl.load(y_ptr+(k-1-offsets), mask=mask, other=0.0)# wouldn't need another load if the block size was always guaranteed to be bigger than k. could flip in program.

    y_vals += alpha*y_flipped_vals

    tl.store(y_ptr + offsets, y_vals, mask=mask)


def kernel(r):
    N = r.shape[0]
    y = torch.empty_like(r)
    r_flipped = torch.flip(r, dims=[0]) # precompute flipping of r

    alpha = -r[0].item()
    beta = 1.0
    y[0] = -r[0]
    grid_cdot = lambda k_val: lambda meta : (triton.cdiv(k_val, meta['BLOCK_SIZE']),)
    grid_uptok = lambda k_val: lambda meta : (triton.cdiv(k_val, meta['BLOCK_SIZE']),)

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        dot_result = torch.zeros(1, dtype=r.dtype, device=r.device)
        _kernel_durbin_iteration_cdot[grid_cdot(k)](r_flipped,y,dot_result,k)
        alpha = (- (r[k] + dot_result[0])/beta).item()
        _kernel_durbin_iteration_update_uptok[grid_uptok(k)](y, alpha, k)
        y[k] = alpha

