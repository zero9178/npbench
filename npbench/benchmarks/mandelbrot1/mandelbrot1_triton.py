import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_mandelbrot(
      N_ptr,          # output: iteration counts
      Z_real_ptr,     # output: real part of Z
      Z_imag_ptr,     # output: imaginary part of Z
      xmin, xmax, ymin, ymax,  # bounds
      xn, yn,         # grid size
      maxiter,
      horizon,
      BLOCK_SIZE_X: tl.constexpr,
      BLOCK_SIZE_Y: tl.constexpr,
  ):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x_idx = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_idx = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    x_mask = x_idx < xn
    y_mask = y_idx < yn
    mask_2d = x_mask[None, :] & y_mask[:, None]

    x_coords = xmin + x_idx * (xmax - xmin) / (xn - 1.0)  # Shape: (BLOCK_SIZE_X,)
    y_coords = ymin + y_idx * (ymax - ymin) / (yn - 1.0)  # Shape: (BLOCK_SIZE_Y,)

    C_real = x_coords[None, :]  # Broadcast x across rows
    C_imag = y_coords[:, None]  # Broadcast y across columns

    Z_real = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)
    Z_imag = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)

    N_out = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int64)

    for n in range(maxiter):
        Z_abs_sq = Z_real * Z_real + Z_imag * Z_imag
        active = Z_abs_sq < horizon * horizon
        N_out = tl.where(active, n, N_out)

        Z_real_new = Z_real * Z_real - Z_imag * Z_imag + C_real
        Z_imag_new = 2.0 * Z_real * Z_imag + C_imag

        Z_real = tl.where(active, Z_real_new, Z_real)
        Z_imag = tl.where(active, Z_imag_new, Z_imag)

    N_out = tl.where(N_out == maxiter - 1, 0, N_out)

    offsets = y_idx[:, None] * xn + x_idx[None, :]
    tl.store(N_ptr + offsets, N_out, mask=mask_2d)
    tl.store(Z_real_ptr + offsets, Z_real, mask=mask_2d)
    tl.store(Z_imag_ptr + offsets, Z_imag, mask=mask_2d)

def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Allocate output tensors
    # X = torch.Tensor(np.linspace(xmin, xmax, xn, dtype=np.float64))
    # Y = torch.Tensor(np.linspace(ymin, ymax, yn, dtype=np.float64))
    # no need for the following as it can be computed inside the kernel: C = torch.Tensor(X + Y[:, None] * 1j)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = torch.zeros((yn, xn), dtype=torch.int64, device=device)
    Z_real = torch.zeros((yn, xn), dtype=torch.float64, device=device)
    Z_imag = torch.zeros((yn, xn), dtype=torch.float64, device=device)

    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    grid = (triton.cdiv(xn, BLOCK_SIZE_X), triton.cdiv(yn, BLOCK_SIZE_Y))


    _kernel_mandelbrot[grid](
        N,
        Z_real,
        Z_imag,
        xmin,
        xmax,
        ymin,
        ymax,
        xn,
        yn,
        maxiter,
        horizon,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    Z = torch.complex(Z_real, Z_imag)
    return Z, N
