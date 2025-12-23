import matplotlib.pyplot as plt
import numpy as np

# 1. Roofline Parameters
SMs = 84
warp_size = 32
num_partitions_per_sm = 4
PEAK_FLOPS_PER_CYCLE = 2 * SMs * warp_size * num_partitions_per_sm  # FMAs per cycle
# Identical peak performance for TF32 on an RTX5080 (easier to reach though).
TENSOR_PEAK_FLOPS_PER_CYCLE = PEAK_FLOPS_PER_CYCLE
BANDWIDTH_BYTES_PER_CYCLE = 960 / 2.617  # GB/s / GHz

"""
Cycles are always measured by nsight compute.

# GEMM: Very first matmul in k3mm

A @ B

Compute: (3200, 4000) @ (4000, 3600): 3200 * 4000 * 3600 FMAs = 3200 * 4000 * 3600 * 2 = 92160000000 flops
Memory = (3200 * 4000 + 4000 * 3600) * 4 = 108800000 Bytes

# gesummv:

alpha * A @ x + beta * B @ x

Compute: (1,) * ((N, N) @ N) + (1,) * ((N, N) @ N) = 
= 11200 + (11200 * 11200 * 2) + 11200 + 11200 + (11200 * 11200 * 2)
Memory = (2 * 11200 * 11200 + 11200) * 4 

# Convolution: 2nd one from resnet (3x3 kernel)
for i in range(56):
    for j in range(56):
        for n in range(8):
            for c1 in range(64):
                for c2 in range(64):
                    for k1 in range(3):
                        for k2 in range(3):
                            output[n, i, j, c2] +=
                                input[n, i + k1, j + k2, c1] *
                                weights[k1, k2, c1, c2]

Note: Output must be zero init and is not counted as needing to be transferred in our roofline
since implementations can exist that do not need to reload output (algorithmically speaking).

Compute:
2 * 56 * 56 * 8 * 64 * 64 * 3 * 3
Memory:
((8 * 58 * 58 * 64) + (3 * 3 * 64 * 64)) * 4

# Jacobi_1d:
for t in range(1, TSTEPS):
    B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
    A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

Compute: (TSTEPS - 2) * 2 * ((N - 2) * 3) = (4000 - 2) * 2 * ((32000 - 2) * 3)
Memory: (N,) * 2 * 4 = 32000 * 2 * 4

# Mandelbrot1

Compute: maxiter * 10 * xn * xy = 200 * 10 * 1000 * 1000
Note: Not counting comparisons or where operations (counting those as control operations, not FLOPS),
only counting inner loop (e.g. not zeroing required due to implementation rather than algorithm)

Memory: 4 * 3 * (yn * xn) = 4 * 3 * 1000 * 1000
"""

# Format: (Algorithmic FLOPs, Data loaded from DRAM in bytes, Cycles executed, Label)
data_points = [
    (3200 * 4000 * 3600 * 2, (3200 * 4000 + 4000 * 3600) * 4, 4709621, 'GEMM'),
    (2 * (11200 * 11200 * 2) + 3 * 11200, (2 * 11200 * 11200 + 11200) * 4, 2523497,
     'gesummv'),
    (2 * 56 * 56 * 8 * 64 * 64 * 3 * 3,
     ((8 * 58 * 58 * 64) + (3 * 3 * 64 * 64)) * 4, 10_958_995,
     'conv2d'),
    ((4000 - 2) * 2 * ((32000 - 2) * 3), 32000 * 2 * 4, 15798713,
     'jacobi_1d'),
    (200 * 10 * 1000 * 1000, 4 * 3 * 1000 * 1000, 212500,
     'mandelbrot1'),
]


def plot_roofline(peak_flops, bandwidth, points):
    # Create x-axis range (logarithmic)
    x = np.logspace(-6, 18, 1000, base=2)
    # Calculate y-axis values: min(Peak, Bandwidth * Intensity)
    y = np.minimum(peak_flops, bandwidth * x)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the Roofline boundaries
    ax.plot(x, y, color='gray', linewidth=2)

    # Add Roofline Labels
    ax.text(2 ** -4, bandwidth * 2 ** -4 * 1.4, r'DRAM Roofline',
            color='gray', fontsize=12, ha='left', rotation=45)
    ax.text(2 ** 4, peak_flops * 1.2, r'FP32 Roofline',
            color='gray', fontsize=12, va='bottom')

    # 3. Plot User Data Points
    # First, let's separate points by color or style if needed.
    # Here we loop through to plot markers and labels.

    for flops, data_loaded, cycles, label in points:
        intensity = flops / data_loaded
        perf = flops / cycles
        ax.scatter(intensity, perf, marker='x', s=80, zorder=3)
        # Offset label slightly to the right
        ax.text(intensity * 1.15, perf, label, fontsize=11, va='center', zorder=3)

    # 5. Axis Formatting (Log Scale Base 2)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    # Set limits (configurable if points fall outside)
    ax.set_xlim(2 ** -6, 2 ** 18)
    ax.set_ylim(2 ** -3, peak_flops * 4)

    # Grid and Labels
    ax.grid(True, which="major", ls="--", alpha=.3, linewidth=.5)
    ax.set_xlabel(r'$I(n)$ [flops/byte]', fontsize=12)
    ax.set_ylabel(r'$P(n)$ [flops/cycle]', fontsize=12)
    ax.set_title('Rooflines of select kernel implementations in Triton', fontsize=14)

    plt.tight_layout()
    plt.savefig("roofline.pdf", dpi=600)
    plt.show()


if __name__ == "__main__":
    plot_roofline(PEAK_FLOPS_PER_CYCLE, BANDWIDTH_BYTES_PER_CYCLE, data_points)
