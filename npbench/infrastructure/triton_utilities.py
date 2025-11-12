"""
This file contains generic kernels for matrix multiplication using Triton.
The float32 kernel is the one that appears in the official tutorial, while
the float64 was adapted from it. Since the float64 kernel cannot use tl.dot,
it is significantly slower.
Neither of the kernels were tuned specifically. The auto-tuning options are
currently commented out for faster development.
"""
import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def grid_sync(barrier):
    """
    Performs a grid level synchronization among every thread block of the GPU. Threads leave the function as soon as
    every thread has entered this function.
    All memory effects performed prior to this function call are guaranteed to be visible to other threads.

    'barrier' should be a pointer to an integer and is required to be 0 or 2^31 when the first thread enters.
    The value is guaranteed to be 0 or 2^31 when all threads leave.

    CAUTION: This function can deadlock if too many blocks are spawned such that they do not all fit into the warp
    scheduler of all SMs! Add `launch_cooperative_grid=True` to the kernel launch call to cause an error if it would
    deadlock.
    A persistent kernel design that launches exactly as many blocks as there are SMs is recommended when using grid
    level synchronization. See 'jacobi_1d_triton.py'.
    """

    tl.static_assert(barrier.dtype.element_ty == tl.int32)

    # Perform thread synchronization by incrementing a barrier by the value 2^31 in total, causing a sign bit flip.
    # All threads but the one with id 0 increment by 1, the thread with id 0 increments by (2^31 - (num_threads - 1)).
    # This makes it such that all threads observe the sign bit change (ie the change from 0 to 2^31 or vice versa) only
    # as soon as every thread has performed the addition.
    expected = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)
    first = (tl.program_id(0) + tl.program_id(1) + tl.program_id(2)) == 0
    nb = 1
    if first:
        nb = -2147483648 - (expected - 1)

    old_arrive = tl.atomic_add(barrier, nb, sem='release')

    c = True
    while c:
        # Compiles to an atomic load due to incrementing by 0.
        current_arrive = tl.atomic_add(barrier, 0, sem='acquire')
        # Check whether the sign bit/top bit has changed.
        if (old_arrive ^ current_arrive) < 0:
            c = False


@triton.jit
def get_2d_tile_offsets(x: tl.int32,
                        y: tl.int32,
                        tile_width: tl.constexpr,
                        tile_height: tl.constexpr,
                        matrix_width: tl.int32,
                        matrix_height: tl.int32) \
        -> tuple[tl.block_type, tl.block_type, tl.block_type, tl.block_type]:
    """
    Generates a tile of offsets that when added to a matrix of width 'matrix_width' and height 'matrix_height',
    yields a tile of width 'tile_width' and height 'tile_height' positioned at 'x' and 'y' within the matrix.

    All coordinates and dimensions are in 'number of elements' unit.
    Assumes a fully contiguous matrix.

    Returns:
        - The offset tile of shape (tile_height, tile_width).
        - A mask that can be used when loading and storing the tile to stay within the bounds of 'matrix_width' and
          'matrix_height'.
        - A vector containing the indices of all rows in the offset tile.
        - A vector containing the indices of all columns in the offset tile.
    """
    columns = x + tl.arange(0, tile_width)
    rows = y + tl.arange(0, tile_height)
    rows_2d = rows[:, None]
    columns_2d = columns[None, :]
    return matrix_width * rows_2d + columns_2d, (columns_2d < matrix_width) & (rows_2d < matrix_height), rows, columns


@triton.jit
def get_1d_tile_offsets(x, tile_width, vector_width):
    """
    Generates a tile of offsets that when added to a vector of length 'vector_width', yields 'tile_width' many elements
    at the offset 'x'.
    Additionally, yields a mask denoting whether every element in the offset tile is within bounds of the vector.
    """
    tile, mask, rows, columns = get_2d_tile_offsets(x=x, y=0,
                                                    tile_width=tile_width,
                                                    tile_height=1,
                                                    matrix_width=vector_width,
                                                    matrix_height=1)
    return tl.reshape(tile, (tile_width,)), tl.reshape(mask, (tile_width,))


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16}),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_float64(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for float64 matrix multiplication.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)

        # Manual matrix multiplication
        accumulator += tl.sum(a[:, :, None] * b[None, :, :], axis=1)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_float64(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper function for the float64 matrix multiplication kernel.
    """
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float64)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel_float64[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
    ],
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel_float32(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_float32(a: torch.Tensor, b: torch.Tensor, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel_float32[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,  #
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor):
    if a.dtype == torch.float64 and b.dtype == torch.float64:
        return matmul_float64(a, b)
    elif a.dtype == torch.float32 and b.dtype == torch.float32:
        return matmul_float32(a, b)
    else:
        raise NotImplementedError("only float32 and float64 are supported in matmul")


def generate_config_mat_vec_mul():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if m != 128 or n != 128
    ]

@triton.autotune(configs=generate_config_mat_vec_mul(), key=["M", "N"], cache_results=True)
@triton.jit()
def mat_vec_mul_kernel(
            A,  # (M, N)
            X,  # (N,)
            out,  # (M,)
            M: tl.constexpr, N: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
            ):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    tile, mask, row, column = get_2d_tile_offsets(
        x=j * BLOCK_SIZE_N,
        y=i * BLOCK_SIZE_M,
        tile_width=BLOCK_SIZE_N,
        tile_height=BLOCK_SIZE_M,
        matrix_width=N,
        matrix_height=M,
    )
    a = tl.load(A + tile, mask)
    x = tl.load(X + column, mask=column < N, other=0.0)

    x_sum = tl.sum(a * x[None, :], axis=1)
    tl.atomic_add(out + row, x_sum, sem="release")

def mat_vec_mul(
            A,  # (M, N)
            X,  # (N,)
            out,  # (M,)
            ):
    """
    Performs matrix-vector multiplication between matrix A (M, N) and vector X (N,)
    Result is written to vector out (M,)
    """

    M, N = A.shape
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    mat_vec_mul_kernel[grid](A, X, out, M, N)