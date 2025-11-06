import torch
import triton
import triton.language as tl

def get_configs():
    return [
        triton.Config({'BLOCK_SIZE': b}, num_warps=w)
        for b in [64, 128, 256, 512]
        for w in [1, 2, 4, 8]
    ]

@triton.autotune(
    configs=get_configs(),
    key=['TSTEPS', 'N'],
)
@triton.jit
def _kernel(TSTEPS: tl.constexpr, src, dst, N: tl.constexpr, barrier,
            BLOCK_SIZE: tl.constexpr):
    x = tl.program_id(axis=0)
    num_x = tl.num_programs(axis=0)

    for i in range(0, TSTEPS):
        for j in range(2):

            mid_offsets = x * BLOCK_SIZE + 1 + tl.arange(0, BLOCK_SIZE)
            left_offsets = mid_offsets - 1
            right_offsets = mid_offsets + 1

            left = tl.load(src + left_offsets, mask=left_offsets < N - 1)
            mid_mask = mid_offsets < N - 1
            middle = tl.load(src + mid_offsets, mask=mid_mask)
            right = tl.load(src + right_offsets, mask=right_offsets < N)
            s = 0.33333 * (left + middle + right)
            tl.store(dst + mid_offsets, s, mask=mid_mask)
            src, dst = dst, src

            tl.atomic_add(barrier, 1)
            while tl.load(barrier, volatile=True) < num_x * (2 * i + j + 1):
                pass


def kernel(TSTEPS: int, A: torch.Tensor, B: torch.Tensor):
    N = A.size(0)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    barrier = torch.zeros(1)
    _kernel[grid](TSTEPS, A, B, N, barrier)
