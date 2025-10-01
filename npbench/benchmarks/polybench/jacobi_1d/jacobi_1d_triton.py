import triton
import torch
import triton.language as tl

BLOCK_SIZE: tl.constexpr = 1024

@triton.jit
def jacobi_kernel(A, B, barrier_flags, N, TSTEPS):
    pid = tl.program_id(axis=0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (off > 0) & (off < N-1)

    p_in = A
    p_out = B

    for t in range(2*TSTEPS-2):
        left   = tl.load(p_in + off - 1, mask=off > 0, other=0.0)
        middle = tl.load(p_in + off)
        right  = tl.load(p_in + off + 1, mask=off < N-1, other=0.0)

        out = (left + middle + right) * 0.33333
        tl.store(p_out + off, out, mask=mask)
        p_in, p_out = p_out, p_in
        tl.atomic_add(barrier_flags + t, 1)
        while tl.load(barrier_flags + t) < tl.num_programs(axis=0):
            pass

def kernel(TSTEPS, A, B):
    N = len(A)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    barrier_flags = torch.zeros((2*TSTEPS-2,), dtype=torch.int32, device='cuda')
    jacobi_kernel[grid](A, B, barrier_flags, N, TSTEPS)

