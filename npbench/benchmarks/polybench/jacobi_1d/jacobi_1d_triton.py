import triton
import torch
import triton.language as tl

BLOCK_SIZE: tl.constexpr = 1024

@triton.jit
def jacobi_kernel(A, B, N):
    pid = tl.program_id(axis=0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (off > 0) & (off < N-1)

    left   = tl.load(A + off - 1, mask=off > 0, other=0.0)
    middle = tl.load(A + off)
    right  = tl.load(A + off + 1, mask=off < N-1, other=0.0)

    out = (left + middle + right) * 0.33333
    tl.store(B + off, out, mask=mask)

def kernel(TSTEPS, A, B):
    N = len(A)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    graph = torch.cuda.CUDAGraph()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph.capture_begin()

        jacobi_kernel[grid](A, B, N)
        jacobi_kernel[grid](B, A, N)

        graph.capture_end()
    torch.cuda.current_stream().wait_stream(s)

    for t in range(1, TSTEPS):
        graph.replay()
