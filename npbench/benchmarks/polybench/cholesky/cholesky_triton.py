import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(A_ptr, N : tl.constexpr, BLOCK_SIZE_N: tl.constexpr, DTYPE : tl.constexpr):
    # N = rows (height), M = cols (width)
    # For an element (i, j) in row-major order: offset=i*M+j

    a_00 = tl.load(A_ptr + 0)
    a_00 = tl.sqrt(a_00)
    tl.store(A_ptr, a_00)

    for i in range(1, N):
        for j in range(i):
            
            # A[i, j] -= np.dot(A[i, :j], A[j, :j])
            sum_val = tl.zeros((), DTYPE)
            for k in range(j):
                a_ik = tl.load(A_ptr + i*N + k)
                a_jk = tl.load(A_ptr + j*N + k)
                sum_val += a_ik * a_jk
            a_ij = tl.load(A_ptr + i*N + j)
            a_ij -= sum_val

            # A[i, j] /= A[j, j]
            a_jj = tl.load(A_ptr + j*N + j)
            a_ij /= a_jj
            tl.store(A_ptr + i*N + j, a_ij)

        # A[i, i] -= np.dot(A[i, :i], A[i, :i])
        offset_i_i = i*N + i
        sum_val = tl.zeros((), DTYPE)
        for i_itr in range(i):
            offset_i_i_itr = i*N + i_itr
            a_i_i_itr = tl.load(A_ptr + offset_i_i_itr)
            sum_val += a_i_i_itr * a_i_i_itr
        a_ii = tl.load(A_ptr + offset_i_i)
        a_ii -= sum_val
        # A[i, i] = np.sqrt(A[i, i])
        a_ii = tl.sqrt(a_ii)
        tl.store(A_ptr + offset_i_i, a_ii)


def kernel(A: torch.Tensor):
    N, M = A.shape  # N = rows (height), M = cols (width)
    assert N == M, "Cholesky decomposition requires a square matrix"

    grid = lambda meta: (
    triton.cdiv(N, meta['BLOCK_SIZE_N']),  # programs along x (columns)
    triton.cdiv(N, meta['BLOCK_SIZE_N']),  # programs along y (rows)
    )

    if A.dtype == torch.float32:
        DTYPE = tl.float32
    elif A.dtype == torch.float64:
        DTYPE = tl.float64
    else:
        raise TypeError("Use float32/float64")

    BLOCK_SIZE_N = 128
    _kernel[grid](A, N, BLOCK_SIZE_N, DTYPE)


# import torch
# import triton
# import triton.language as tl
# from npbench.benchmarks.polybench.gemm.gemm_triton import kernel as gemm_kernel

# @triton.jit
# def potrf_diag(A, N, lda, k,
#                BK: tl.constexpr, eps: tl.constexpr,
#                DTYPE: tl.constexpr, ACC: tl.constexpr):
#     base = A + k*lda + k

#     zero = tl.zeros((), dtype=ACC)
#     # (0,0)
#     a00 = tl.cast(tl.load(base + 0*lda + 0), ACC)
#     # CHANGED: use eps (cast to ACC), not 0.0
#     a00 = tl.sqrt(tl.maximum(a00, tl.cast(eps, ACC)))   # CHANGED
#     tl.store(base + 0*lda + 0, tl.cast(a00, DTYPE))

#     for i in range(1, BK):
#         # off-diagonals
#         for j in range(0, i):
#             acc = tl.zeros((), dtype=ACC)
#             for q in range(0, j):
#                 li = tl.cast(tl.load(base + i*lda + q), ACC)
#                 lj = tl.cast(tl.load(base + j*lda + q), ACC)
#                 acc += li * lj
#             aij = tl.cast(tl.load(base + i*lda + j), ACC) - acc
#             ljj = tl.cast(tl.load(base + j*lda + j), ACC)
#             aij = aij / ljj
#             tl.store(base + i*lda + j, tl.cast(aij, DTYPE))

#         # diagonal
#         accii = tl.zeros((), dtype=ACC)
#         for q in range(0, i):
#             v = tl.cast(tl.load(base + i*lda + q), ACC)
#             accii += v * v
#         aii = tl.cast(tl.load(base + i*lda + i), ACC) - accii
#         aii = tl.sqrt(tl.maximum(aii, tl.cast(eps, ACC)))
#         tl.store(base + i*lda + i, tl.cast(aii, DTYPE))

#     # zero upper triangle of this BKxBK tile
#     for r in range(0, BK):
#         for c in range(r+1, BK):
#             tl.store(base + r*lda + c, tl.cast(zero, DTYPE))


# @triton.jit
# def trsm_panel(A, N, lda, k, BK: tl.constexpr, BLOCK: tl.constexpr,
#                DTYPE: tl.constexpr, ACC: tl.constexpr):
#     pid = tl.program_id(0)                 # row-tile index
#     bi  = k + BK + pid * BLOCK             # starting row
#     if bi >= N:
#         return
#     m = tl.minimum(BLOCK, N - bi)          # runtime extent (<= BLOCK)

#     Lkk = A + k*lda + k                    # (BK,BK)
#     Aik = A + bi*lda + k                   # (m,BK)

#     # compile-time aranges; mask rows by runtime m
#     rows = tl.arange(0, BLOCK)
#     row_mask = rows < m

#     # Right-side solve: Aik = Aik * inv(Lkk^T)
#     for j in range(0, BK):
#         # col_j := Aik[:, j]
#         col_ptrs = Aik + rows * lda + j
#         col = tl.cast(tl.load(col_ptrs, mask=row_mask, other=0.0), ACC)

#         # divide by diagonal L[j,j]
#         ljj = tl.cast(tl.load(Lkk + j*lda + j), ACC)
#         col = col / ljj

#         # write back updated column j
#         tl.store(col_ptrs, tl.cast(col, DTYPE), mask=row_mask)

#         # update trailing columns t = j+1..BK-1: Aik[:, t] -= col * L[t,j]
#         if j + 1 < BK:
#             for t in range(j+1, BK):
#                 ltj = tl.cast(tl.load(Lkk + t*lda + j), ACC)
#                 t_ptrs = Aik + rows * lda + t
#                 tcol  = tl.cast(tl.load(t_ptrs, mask=row_mask, other=0.0), ACC)
#                 tcol  = tcol - col * ltj
#                 tl.store(t_ptrs, tl.cast(tcol, DTYPE), mask=row_mask)
       

# def kernel(A: torch.Tensor):
#     BK=64
#     BLOCK=128
#     N = A.shape[0]
#     lda = N

#     dtype = A.dtype
#     assert dtype in (torch.float32, torch.float64)

#     # pick Triton types
#     if dtype == torch.float32:
#         DTYPE, ACC = tl.float32, tl.float32
#         eps = 1e-6   # CHANGED: dtype-aware eps
#     else:  # float64
#         DTYPE, ACC = tl.float64, tl.float64
#         eps = 1e-12  # CHANGED: dtype-aware eps
        
#     for k in range(0, N, BK):
#         bk = min(BK, N - k)

#         # 1) POTRF on diagonal tile: single program (serial inside the tile)
#         potrf_diag[(1,)](A, N, lda, k, BK=bk, eps=eps, DTYPE=DTYPE, ACC=ACC)  # CHANGED

#         if k + bk >= N:
#             continue

#         # 2) TRSM: solve panel below diagonal - 1D grid over row tiles
#         grid_y = (N - (k + bk) + BLOCK - 1) // BLOCK
#         trsm_panel[(grid_y,)](A, N, lda, k, BK=bk, BLOCK=BLOCK, DTYPE=DTYPE, ACC=ACC)

#         # 3) GEMM: trailing update - 2D grid over lower-tri tiles
#         # alpha = -1, beta = 1
#         A21 = A[k + bk:, k : k + bk]
#         A22 = A[k + bk:, k + bk:]
#         A21T = A21.transpose(0, 1).contiguous()
#         gemm_kernel(-1.0, 1.0, A22, A21, A21T)

#     return A
