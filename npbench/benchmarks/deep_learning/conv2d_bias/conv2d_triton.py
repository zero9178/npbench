import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_conv2d(
        input_ptr,
        weights_ptr,
        output_ptr,
        bias_ptr,
        N, H, W, C_in,  # Input dimensions
        K,  # Kernel size
        C_out,  # Output channels
        H_out, W_out,  # Output spatial dimensions
        BLOCK_C_IN: tl.constexpr,  # Block size for C_in dimension
):
    # Get which output element this thread computes
    # We'll use a 3D grid: (N * H_out * W_out, C_out, 1)
    # Flatten the spatial dimensions into axis 0
    spatial_idx = tl.program_id(0)
    c_out = tl.program_id(1)

    # Decode spatial index into (n, h_out, w_out)
    n = spatial_idx // (H_out * W_out)
    remainder = spatial_idx % (H_out * W_out)
    h_out = remainder // W_out
    w_out = remainder % W_out

    # Initialize accumulator with bias
    acc = tl.load(bias_ptr + c_out)

    # Compute convolution: sum over (kh, kw, c_in)
    # Vectorize over C_in for better memory bandwidth utilization
    for kh in range(K):
        for kw in range(K):
            # Process C_in in blocks for vectorization
            for c_in_block_start in range(0, C_in, BLOCK_C_IN):
                # Compute indices for this block of input channels
                c_in_offsets = c_in_block_start + tl.arange(0, BLOCK_C_IN)
                c_in_mask = c_in_offsets < C_in

                # Vectorized load of input channels for this spatial position
                # Input indices: [n, h_out + kh, w_out + kw, c_in]
                input_base = n * H * W * C_in + (h_out + kh) * W * C_in + (w_out + kw) * C_in
                input_indices = input_base + c_in_offsets
                input_vals = tl.load(input_ptr + input_indices, mask=c_in_mask, other=0.0)

                # Vectorized load of weights for this kernel position and output channel
                # Weight indices: [kh, kw, c_in, c_out]
                weight_base = kh * K * C_in * C_out + kw * C_in * C_out + c_out
                weight_indices = weight_base + c_in_offsets * C_out
                weight_vals = tl.load(weights_ptr + weight_indices, mask=c_in_mask, other=0.0)

                # Accumulate: dot product over C_in dimension
                acc += tl.sum(input_vals * weight_vals)

    # Store output: [n, h_out, w_out, c_out]
    output_idx = n * H_out * W_out * C_out + h_out * W_out * C_out + w_out * C_out + c_out
    tl.store(output_ptr + output_idx, acc)


def conv2d_bias(input, weights, bias):
    # Get dimensions
    N, H, W, C_in = input.shape
    K = weights.shape[0]  # Assuming square kernel
    C_out = weights.shape[3]
    H_out = H - K + 1
    W_out = W - K + 1

    # Allocate output
    output = torch.empty((N, H_out, W_out, C_out), device=input.device, dtype=input.dtype)

    # Choose block size for input channels (power of 2 for efficiency)
    BLOCK_C_IN = min(128, triton.next_power_of_2(C_in))

    # Launch kernel with 3D grid (Triton only supports up to 3D)
    # Flatten batch and spatial dimensions into one axis
    grid = (N * H_out * W_out, C_out, 1)
    _kernel_conv2d[grid](
        input, weights, output, bias,
        N, H, W, C_in,
        K,
        C_out,
        H_out, W_out,
        BLOCK_C_IN=BLOCK_C_IN
    )

    return output