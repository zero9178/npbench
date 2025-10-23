import torch
import triton
import triton.language as tl

@triton.jit
def _mvt_kernel(
    x1_ptr,  
    x2_ptr, 
    y1_ptr,
    y2_ptr,
    A_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # - Program i computes x1[i] (using the full row A[i, :])
    x1_scalar_acc = tl.load(x1_ptr+pid)
    for j_start in range(0, N, BLOCK_SIZE):
        offsets_j = j_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets_j < N
        A_row_offsets =  A_ptr + pid * N + offsets_j
        y1_offsets = y1_ptr + offsets_j
        A_row = tl.load(A_row_offsets, mask=mask)
        y1 = tl.load(y1_offsets, mask=mask)
        x1_scalar_acc += tl.sum(A_row*y1)

    tl.store(x1_ptr+pid,x1_scalar_acc)

    # - The same program i also computes x2[i] (using the full column A[:, i]). very inefficient for now, maybe there is a better way using square blocks? for later...
    x2_scalar_acc = tl.load(x2_ptr+pid)
    for j_start in range(0, N, BLOCK_SIZE):
        offsets_j = j_start + tl.arange(0,BLOCK_SIZE)
        mask = offsets_j < N
        A_col_offsets = A_ptr + pid + offsets_j * N 
        y2_offsets = y2_ptr + offsets_j
        A_col = tl.load(A_col_offsets, mask=mask)# bad data locality but oh well.
        y2 = tl.load(y2_offsets, mask=mask)
        x2_scalar_acc += tl.sum(A_col*y2)
    tl.store(x2_ptr+pid, x2_scalar_acc)

def kernel(x1:torch.Tensor, x2:torch.Tensor, y_1:torch.Tensor, y_2:torch.Tensor, A:torch.Tensor):
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    y_1 = y_1.contiguous()
    y_2 = y_2.contiguous()


    dtype = A.dtype
    if dtype == torch.float32:
        DTYPE = tl.float32
    else:  # float64
        DTYPE = tl.float64
    
    N, N = A.shape
    # Grid: one program per i value
    grid = (N,)
    BLOCK_SIZE = 64

    _mvt_kernel[grid](x1, x2, y_1, y_2, A, N, BLOCK_SIZE = tl.constexpr(BLOCK_SIZE), DTYPE = tl.constexpr(DTYPE))