import torch
import triton.language as tl
import triton  



@triton.jit
def jacobi_step_kernel(A_ptr, B_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)# pid for current triton kernel instance
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offsets < N - 2

    a_left = tl.load(A_ptr + offsets, mask=mask)
    a_center = tl.load(A_ptr + offsets + 1, mask=mask)
    a_right = tl.load(A_ptr + offsets + 2, mask=mask)

    result = 0.33333 * (a_left + a_center + a_right)

    tl.store(B_ptr + 1 + offsets, result, mask=mask)

def kernel(TSTEPS, A, B):
    N = A.shape[0] # helped by Claude
    grid = (triton.cdiv(N-2, 1024),) #heavily helped by Claude
    for t in range(1, TSTEPS):
        jacobi_step_kernel[grid](A, B, N, 1024)
        jacobi_step_kernel[grid](B, A, N, 1024)