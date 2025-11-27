import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True  # Disabled to avoid LaTeX issues

from scipy.stats.mstats import gmean
from npbench.infrastructure import utilities as util


def bootstrap_ci(data, statfunction=np.median, alpha=0.05, n_samples=300):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""

    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0], ))

    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)

    data = np.array(data)
    if np.prod(data.shape) != max(data.shape):
        raise ValueError("Data must be 1D")
    data = data.ravel()

    boot_indexes = bootstrap_ids(data, n_samples)
    stat = np.asarray([statfunction(data[_ids]) for _ids in boot_indexes])
    stat.sort(axis=0)

    return stat[nvals]


# Custom category: Stencils
# Stencils are computational patterns where each output element is computed
# from neighboring input elements (local spatial/temporal dependencies)
STENCIL_BENCHMARKS = {
    # Iterative solvers with neighbor updates
    'jacobi1d',      # 1D Jacobi iteration
    'jacobi2d',      # 2D Jacobi iteration
    'seidel2d',      # 2D Gauss-Seidel iteration

    # Heat/diffusion equations
    'heat3d',        # 3D heat equation
    'hdiff',         # Horizontal diffusion (weather)

    # Advection
    'vadv',          # Vertical advection (weather)

    # Finite difference methods
    'fdtd_2d',       # Finite Difference Time Domain (2D)
    'adi',           # Alternating Direction Implicit method

    # Convolutions and filters
    'conv2d',        # 2D convolution
    'deriche',       # Deriche edge detection filter

    # Other stencil-like operations
    'gramschm',      # Gram-Schmidt (has neighbor dependencies in modified version)
}

# Camera-ready stencils: representative benchmarks with Triton implementations
# From stencils, only heat3d, jacobi2d have Triton in preset S
# Adding clipping (image processing stencil-like) and doitgen for variety
STENCILS_CAMREADY = {
    'heat3d',        # GPU excellent: ~40x speedup with dace_gpu/triton
    'jacobi2d',      # Good case: moderate speedups
    'clipping',      # Image processing
    'doitgen',       # Tensor multiplication (local dependencies)
}

# Camera-ready LinAlg: Dense matrix operations with variety of performance characteristics
LINALG_CAMREADY = {
    'gemm',          # Matrix multiply - fundamental operation
    'atax',          # Matrix-vector ops
    'syr2k',         # Symmetric rank-2k update - shows high speedups
    'trmm',          # Triangular matrix multiply
    '2mm',           # Chained matrix multiply
    'bicg',          # BiCG iterative solver
}

# Camera-ready Learning: Neural network and statistical operations with Triton
LEARNING_CAMREADY = {
    'softmax',       # Very high speedups with Triton (~70x)
    'mlp',           # Multi-layer perceptron (~40x with GPU)
    'correlat',      # Correlation
    'covarian',      # Covariance
    'cholesky',      # Matrix decomposition used in ML
}

# Camera-ready Misc: Interesting diverse benchmarks
MISC_CAMREADY = {
    'mandel1',       # Mandelbrot set - embarrassingly parallel
    'mandel2',       # Mandelbrot variant
    'floydwar',      # Floyd-Warshall - dynamic programming
    'crc16',         # CRC checksum
    'azimhist',      # Histogram computation
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper'],
                        nargs="?",
                        default='S')
    parser.add_argument("-d",
                        "--domain",
                        type=str,
                        default=None,
                        help="Filter by domain/category (e.g., 'stencil', 'linear-algebra')")
    args = vars(parser.parse_args())

# create a database connection
database = r"npbench.db"
conn = util.create_connection(database)
data = pd.read_sql_query("SELECT * FROM results", conn)

# get rid of kind and dwarf, we don't use them
data = data.drop(['timestamp', 'kind', 'dwarf', 'version'],
                 axis=1).reset_index(drop=True)

# Remove everything that does not have a domain
data = data[data["domain"] != ""]

# Print available domains (including custom categories)
available_domains = sorted(data['domain'].unique())
custom_domains = ['stencils', 'stencils-camready', 'linalg-camready', 'learning-camready', 'misc-camready']
all_domains = sorted(available_domains + custom_domains)
print(f"Available domains: {all_domains}")
print(f"  (Custom categories: {custom_domains})")

# remove everything that does not validate, then get rid of validated column
data = data[data['validated'] == True]
data = data.drop(['validated'], axis=1).reset_index(drop=True)

# Filter by preset
data = data[data['preset'] == args['preset']]
data = data.drop(['preset'], axis=1).reset_index(drop=True)

# Filter by domain if specified
if args['domain'] is not None:
    # Handle custom categories
    if args['domain'] == 'stencils':
        data = data[data['benchmark'].isin(STENCIL_BENCHMARKS)]
        print(f"Filtering by custom category: stencils ({len(STENCIL_BENCHMARKS)} benchmarks)")
        print(f"  Stencil benchmarks: {sorted(STENCIL_BENCHMARKS)}")
    elif args['domain'] == 'stencils-camready':
        data = data[data['benchmark'].isin(STENCILS_CAMREADY)]
        print(f"Filtering by custom category: stencils-camready ({len(STENCILS_CAMREADY)} benchmarks)")
        print(f"  Selected benchmarks: {sorted(STENCILS_CAMREADY)}")
    elif args['domain'] == 'linalg-camready':
        data = data[data['benchmark'].isin(LINALG_CAMREADY)]
        print(f"Filtering by custom category: linalg-camready ({len(LINALG_CAMREADY)} benchmarks)")
        print(f"  Selected benchmarks: {sorted(LINALG_CAMREADY)}")
    elif args['domain'] == 'learning-camready':
        data = data[data['benchmark'].isin(LEARNING_CAMREADY)]
        print(f"Filtering by custom category: learning-camready ({len(LEARNING_CAMREADY)} benchmarks)")
        print(f"  Selected benchmarks: {sorted(LEARNING_CAMREADY)}")
    elif args['domain'] == 'misc-camready':
        data = data[data['benchmark'].isin(MISC_CAMREADY)]
        print(f"Filtering by custom category: misc-camready ({len(MISC_CAMREADY)} benchmarks)")
        print(f"  Selected benchmarks: {sorted(MISC_CAMREADY)}")
    elif args['domain'] in available_domains:
        data = data[data['domain'] == args['domain']]
        print(f"Filtering by domain: {args['domain']}")
    else:
        print(f"Warning: Domain '{args['domain']}' not found. Available domains: {all_domains}")
        exit(1)
else:
    print("No domain filter applied (showing all domains)")

# for each framework and benchmark, choose only the best details,mode (based on median runtime)
aggdata = data.groupby(["benchmark", "domain", "framework", "mode", "details"],
                       dropna=False).agg({
                           "time": "median"
                       }).reset_index()
best = aggdata.sort_values("time").groupby(
    ["benchmark", "domain", "framework", "mode"],
    dropna=False).first().reset_index()
bestgroup = best.drop(["time"], axis=1)
data = pd.merge(left=bestgroup,
                right=data,
                on=["benchmark", "domain", "framework", "mode", "details"],
                how="inner")
data = data.drop(['mode', 'details'], axis=1).reset_index(drop=True)

# Remove dace_cpu, pythran, and numba from comparisons
print("Removing dace_cpu, pythran, and numba from all comparisons")
data = data[data['framework'] != 'dace_cpu']
data = data[data['framework'] != 'pythran']
data = data[data['framework'] != 'numba']

frmwrks = list(data['framework'].unique())
print(f"Frameworks: {frmwrks}")
assert ('numpy' in frmwrks)
frmwrks.remove('numpy')
frmwrks.sort()  # Sort for consistent ordering

# For camera-ready plots, ensure benchmarks have key framework implementations
if args['domain'] and 'camready' in args['domain']:
    print("Ensuring camera-ready benchmarks have implementations for key frameworks...")

    # Check which benchmarks have Triton implementations
    triton_benchmarks = set(data[data['framework'] == 'triton']['benchmark'].unique())
    current_benchmarks = set(data['benchmark'].unique())

    missing_triton = current_benchmarks - triton_benchmarks
    if missing_triton:
        print(f"  WARNING: Removing benchmarks without Triton: {sorted(missing_triton)}")
        data = data[data['benchmark'].isin(triton_benchmarks)]

    # Verify each benchmark has all major frameworks
    for benchmark in sorted(data['benchmark'].unique()):
        bench_frameworks = set(data[data['benchmark'] == benchmark]['framework'].unique())
        if 'triton' not in bench_frameworks:
            print(f"  Removing {benchmark}: missing triton")
            data = data[data['benchmark'] != benchmark]

    remaining = sorted(data['benchmark'].unique())
    print(f"  Final camera-ready benchmarks: {remaining}")

# Calculate speedups and confidence intervals for each benchmark+framework
benchmarks = sorted(data['benchmark'].unique())
results = []

for benchmark in benchmarks:
    bench_data = data[data['benchmark'] == benchmark]

    # Get numpy baseline times
    numpy_times = bench_data[bench_data['framework'] == 'numpy']['time'].values
    if len(numpy_times) == 0:
        continue
    numpy_median = np.median(numpy_times)

    for framework in frmwrks:
        frmwrk_times = bench_data[bench_data['framework'] == framework]['time'].values

        if len(frmwrk_times) == 0:
            continue

        # Calculate speedup (numpy_time / framework_time)
        # Values > 1 mean framework is faster than numpy
        speedups = numpy_median / frmwrk_times
        median_speedup = np.median(speedups)

        # Calculate confidence intervals
        ci = bootstrap_ci(speedups, statfunction=np.median, alpha=0.05, n_samples=300)
        ci_lower = ci[0]
        ci_upper = ci[1]

        # Error bar sizes (distance from median to CI bounds)
        error_lower = median_speedup - ci_lower
        error_upper = ci_upper - median_speedup

        results.append({
            'benchmark': benchmark,
            'framework': framework,
            'speedup': median_speedup,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'error_lower': error_lower,
            'error_upper': error_upper
        })

results_df = pd.DataFrame(results)

# Create grouped bar plot
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(benchmarks))
width = 0.8 / len(frmwrks)  # width of each bar

colors = plt.cm.Set2(np.linspace(0, 1, len(frmwrks)))

for i, framework in enumerate(frmwrks):
    frmwrk_data = results_df[results_df['framework'] == framework]

    # Align data with benchmark order
    speedups = []
    errors_lower = []
    errors_upper = []
    bottoms = []  # Where each bar starts
    heights = []  # Height of each bar

    for bench in benchmarks:
        bench_frmwrk = frmwrk_data[frmwrk_data['benchmark'] == bench]
        if len(bench_frmwrk) > 0:
            speedup = bench_frmwrk['speedup'].values[0]
            speedups.append(speedup)
            errors_lower.append(bench_frmwrk['error_lower'].values[0])
            errors_upper.append(bench_frmwrk['error_upper'].values[0])

            # Bars grow from baseline (y=1)
            # Speedup > 1: bar grows upward from 1
            # Speedup < 1: bar grows downward from 1
            if speedup >= 1:
                bottoms.append(1)
                heights.append(speedup - 1)
            else:
                bottoms.append(speedup)
                heights.append(1 - speedup)
        else:
            speedups.append(0)
            errors_lower.append(0)
            errors_upper.append(0)
            bottoms.append(1)
            heights.append(0)

    # Plot bars with error bars, growing from baseline
    positions = x + i * width - (len(frmwrks) - 1) * width / 2
    ax.bar(positions, heights, width, bottom=bottoms, label=framework, color=colors[i],
           yerr=[errors_lower, errors_upper], capsize=2, error_kw={'linewidth': 0.5})

# Add horizontal line at y=1 (numpy baseline)
ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='NumPy baseline')

ax.set_xlabel('Benchmark', fontsize=12)
ax.set_ylabel('Speedup over NumPy', fontsize=12)

# Update title based on domain filter
title = f'Benchmark Performance Comparison (Preset: {args["preset"]}'
if args['domain']:
    title += f', Domain: {args["domain"]}'
title += ')'
ax.set_title(title, fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks, rotation=45, ha='right')
ax.legend(loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Use log scale for y-axis to better show both speedups and slowdowns
ax.set_yscale('log')

# Dynamically set y-axis limits based on data to prevent overflow
max_speedup = results_df['speedup'].max()
min_speedup = results_df['speedup'].min()

# Add padding: for max, go to next power of 10; for min, go to previous power of 10
y_max = 10 ** math.ceil(math.log10(max_speedup * 1.2))  # 20% padding
y_min = 10 ** math.floor(math.log10(min_speedup * 0.8))  # 20% padding

# Ensure minimum range
y_max = max(y_max, 10)
y_min = min(y_min, 0.1)

ax.set_ylim([y_min, y_max])

plt.tight_layout()

# Update filename based on domain filter
filename = f"barplot-{args['preset']}"
if args['domain']:
    filename += f"-{args['domain']}"
filename += ".pdf"

plt.savefig(filename, dpi=600)
print(f"Saved {filename}")
plt.show()
