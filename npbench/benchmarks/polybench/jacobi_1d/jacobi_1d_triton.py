import torch
import triton.language as tl
import triton  


@triton.jit
def _add(z_ptr, x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)# pid for current triton kernel instance
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N

    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    z_ptrs = z_ptr + offsets

    x = tl.load(x_ptrs, mask=mask)
    y = tl.load(y_ptrs, mask=mask)

    z = x + y

    tl.store(z_ptrs, z, mask=mask)

@triton.jit
def _mul_by_constant(res_ptr, x_ptr, constant, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)# pid for current triton kernel instance
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N

    x_ptrs = x_ptr + offsets

    x = tl.load(x_ptrs, mask=mask)

    res = x * constant

    res_ptrs = res_ptr + offsets
    tl.store(res_ptrs, res, mask=mask)


@triton.jit
def step(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    _add(B_ptr+1,A_ptr,A_ptr+1,N-2, BLOCK_SIZE)
    _add(B_ptr+1,B_ptr+1,A_ptr+2,N-2, BLOCK_SIZE)
    _mul_by_constant(B_ptr+1, B_ptr+1, 0.33333, N-2, BLOCK_SIZE)

@triton.jit
def run_fn(TSTEPS, A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    for t in range(1, TSTEPS):
        step(A_ptr, B_ptr, N, BLOCK_SIZE)
        step(B_ptr, A_ptr, N, BLOCK_SIZE)

def kernel(TSTEPS, A, B):
    N = A.shape[0] # helped by Claude
    grid = (triton.cdiv(N-2, 1024),) #heavily helped by Claude
    run_fn[grid](TSTEPS, A, B, N, 1024)