import torch
import triton
import triton.language as tl
import itertools
from npbench.infrastructure.triton_utilities import get_1d_tile_offsets

def generate_config():
    base = [
        (64, 32, 4, 3),
        (128, 32, 4, 3),
        (64, 64, 4, 3),
        (128, 128, 8, 4),  
    ]
    return [triton.Config(
                kwargs={"BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k},
                num_warps=w, num_stages=s)
            for (n, k, w, s) in base]

@triton.autotune(configs=generate_config(), key=["N", "K"], cache_results=True)
@triton.jit
def _kernel(L, x, b, N, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr,
            BLOCK_SIZE_K : tl.constexpr):

    pid_m = tl.program_id(axis=0) # rows 
    pid_n = tl.program_id(axis=1)  # cols

    if pid_n > pid_m:
        return

    # Compute local offsets within that tile
    # tl.arange(0, BLOCK_SIZE) = [0, 1, 2, ..., BLOCK_SIZE-1]
    row_offs = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    col_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_row = row_offs < N
    mask_col = col_offs < N

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=DTYPE)

    for k0 in range(0, pid_m*BLOCK_SIZE_N, BLOCK_SIZE_K):
        # For a fixed i, split j ∈ [0, i) into tiles of size BLOCK_K and accumulate partial sums.
        # Each tile loads a vector of L[i, j0:j0+BLOCK_SIZE_K]
        # and the matching vector x[j0:j0+BLOCK_SIZE_K]
        tile, mask = get_1d_tile_offsets(k0, BLOCK_SIZE_K, N)

        L_tile = tl.load(L + k0 * N + tile, mask=mask)  # (BLOCK_SIZE_K,)
        x_tile = tl.load(x + tile, mask=mask)          # (BLOCK_SIZE_K,)

        acc += tl.sum(L_tile * x_tile, axis=0)


    # Load b[i] and x[j]
    x_vec = tl.load(x + col_offs, mask=mask_col, other=0.0)  # (BLOCK_SIZE x 1)
    b_vec = tl.load(b + row_offs, mask=mask_row, other=0.0)  # (BLOCK_SIZE x 1)

    # x[i] = (b[i] - s) / L[i, i]
    b_subtract = b_vec - acc
    L_diag = tl.load(L + row_offs * N + col_offs, mask=mask_row & mask_col, other=0.0)  # (BLOCK_SIZE,)
    x_result = b_subtract / L_diag
    tl.store(x + row_offs, x_result, mask=mask_row)


def kernel(L, x, b):
    # Assume A is a square matrix of size NxN
    N, M = L.shape
    x_len = x.shape[0]
    b_len = b.shape[0]
    assert x_len == N, "x length must match L dimensions"
    assert b_len == N, "b length must match L dimensions"
    assert N == M, "L must be a square matrix"

    dtype = L.dtype
    assert dtype in (torch.float32, torch.float64)

    DTYPE = tl.float32 if dtype == torch.float32 else tl.float64

    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),  # rows
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),  # cols
    )


    #     for i in range(N):
    #         s = 0.0
    #         for j in range(i):
    #             s += L[i, j] * x[j]
    #         x[i] = (b[i] - s) / L[i, i]
    _kernel[grid](L, x, b, N, DTYPE)
