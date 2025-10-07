import torch
import triton.language as tl
import triton  


@triton.jit
def _add(z_ptr, x_ptr, y_ptr, N):
    offsets = tl.arange(0,N)

    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    z_ptrs = z_ptr + offsets

    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)

    z = x + y

    tl.store(z_ptrs, z)

@triton.jit
def _mul_by_constant(res_ptr, x_ptr, constant, N):
    offsets = tl.arange(0,N)
    x_ptrs = x_ptr + offsets

    x = tl.load(x_ptrs)

    res = x * constant

    res_ptrs = res_ptr + offsets
    tl.store(res_ptrs, res)


@triton.jit
def step(A_ptr, B_ptr, N):
    _add(B_ptr+1,A_ptr,A_ptr+1,N-2)
    _add(B_ptr+1,B_ptr+1,A_ptr+2,N-2)
    _mul_by_constant(B_ptr+1, B_ptr+1, 0.33333, N-2)

@triton.jit
def run_fn(TSTEPS, A_ptr, B_ptr, N):
    for t in range(1, TSTEPS):
        step(A_ptr, B_ptr, N)
        step(B_ptr, A_ptr, N)

def kernel(TSTEPS, A, B):
    N = A.shape[0] # helped by Claude
    grid = (1,)
    run_fn[grid](TSTEPS, A, B, N)