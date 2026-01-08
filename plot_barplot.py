import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from npbench.infrastructure import utilities as util

# matplotlib.rcParams['text.usetex'] = True  # Disabled to avoid LaTeX issues

np.random.seed(42)


def bootstrap_ratio_ci_unpaired(num_data, denom_data, alpha=0.05, n_samples=300):
    """
    Calculates the CI for Mean(num) / Mean(denom) assuming INDEPENDENT samples.
    The arrays do NOT need to be the same shape.
    """
    # 1. Flatten and validate inputs
    num_data = np.array(num_data).ravel()
    denom_data = np.array(denom_data).ravel()

    n_num = num_data.shape[0]
    n_denom = denom_data.shape[0]

    # 2. Calculate percentile indices
    alphas = np.array([alpha / 2, 1 - alpha / 2])
    nvals = np.round((n_samples - 1) * alphas).astype(int)

    # 3. Generate independent resampled indices
    # We generate a matrix of (n_samples x original_size) for vectorization
    rng = np.random.default_rng()

    # Resample Numerator
    idx_num = rng.integers(0, n_num, size=(n_samples, n_num))
    boot_num_means = np.mean(num_data[idx_num], axis=1)

    # Resample Denominator (independently)
    idx_denom = rng.integers(0, n_denom, size=(n_samples, n_denom))
    boot_denom_means = np.mean(denom_data[idx_denom], axis=1)

    # 4. Calculate Ratio of Means
    stats = boot_num_means / boot_denom_means
    stats.sort()

    return stats[nvals]


if __name__ == "__main__":
    preset = 'paper'

    # create a database connection
    database = r"npbench_5080.db"
    conn = util.create_connection(database)

    # SQL query to get triton speedup relative to best non-triton framework
    triton_speedup_query = """
                           -- Result with most recent timestamp
                           WITH recent_results AS (SELECT r.benchmark, r.framework, r.preset, r.details, r.time
                                                   FROM results r
                                                   WHERE timestamp == (SELECT MAX(timestamp)
                                                                       FROM results q
                                                                       WHERE r.benchmark = q.benchmark
                                                                         AND r.framework = q.framework
                                                                         AND r.preset = q.preset
                                                                         AND r.details = q.details)),
                                -- For a given timestamp/benchmarking run, average the time.
                                -- TODO: SQLIte does not support median
                                averaged AS (SELECT benchmark, framework, preset, details, AVG(time) AS median
                                             FROM recent_results
                                             GROUP BY benchmark, framework, preset, details),
                                -- pick the lowest time when there are multiple variants for a framework (e.g. in dace_gpu).
                                best_details AS (SELECT benchmark, framework, preset, details, median
                                                 FROM averaged
                                                 WHERE median == (SELECT MIN(median)
                                                                  FROM averaged q
                                                                  WHERE q.benchmark = averaged.benchmark
                                                                    AND q.framework = averaged.framework
                                                                    AND q.preset = averaged.preset)),
                                -- name to use from now on, and filters for paper.
                                final_results AS (SELECT benchmark, framework, details, median
                                                  FROM best_details
                                                  WHERE preset = 'paper'),
                                best_non_triton AS (SELECT benchmark,
                                                           framework,
                                                           details,
                                                           median,
                                                           -- For every benchmark, ranks the performance from 1 to n.
                                                           ROW_NUMBER() OVER (PARTITION BY benchmark ORDER BY median) AS rn
                                                    FROM final_results
                                                    WHERE framework <> 'triton'),
                                triton_times as (SELECT * FROM final_results WHERE framework = 'triton')
                           SELECT t.benchmark         as benchmark,
                                  b.framework         as best_framework,
                                  b.details           as best_details,
                                  b.median / t.median as speedup
                           FROM triton_times t
                                    JOIN best_non_triton b
                                         ON t.benchmark = b.benchmark
                                             AND b.rn = 1
                           ORDER BY speedup DESC
                           """

    results_df = pd.read_sql_query(triton_speedup_query, conn)

    print(f"Total benchmarks with Triton speedup data: {len(results_df)}")

    benchmarks = results_df['benchmark'].tolist()

    # Define colors for each framework (matching the benchmark_grid.pdf)
    framework_colors = {
        'cupy': '#17becf',  # cyan
        'dace_cpu': '#1f77b4',  # dark blue
        'dace_gpu': '#9467bd',  # purple
        'numba': '#1f77b4',  # dark blue
        'pythran': '#2ca02c',  # teal/green
        'triton': '#ff7f0e',  # orange
        'jax': '#d62728',  # red
        'numpy': '#999999',  # gray
    }

    # Create bar plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(benchmarks))
    speedups = results_df['speedup'].values
    best_frameworks = results_df['best_framework'].values

    # Create bars
    bottoms = []
    heights = []
    colors = []
    for speedup, framework in zip(speedups, best_frameworks):
        # Bars grow from baseline (y=1)
        # Speedup > 1: bar grows upward from 1 (triton is faster)
        # Speedup < 1: bar grows downward from 1 (triton is slower)
        if speedup >= 1:
            bottoms.append(1)
            heights.append(speedup - 1)
        else:
            bottoms.append(speedup)
            heights.append(1 - speedup)
        colors.append(framework_colors.get(framework, '#999999'))

    # Plot bars
    ax.bar(x, heights, bottom=bottoms, color=colors)
    for xe, benchmark, speedup, framework, details in zip(x, benchmarks, speedups, best_frameworks,
                                                          results_df['best_details']):
        triton_times = pd.read_sql_query(f"""
        WITH recent_results AS (SELECT r.benchmark, r.framework, r.preset, r.details, r.time
                                                   FROM results r
                                                   WHERE timestamp == (SELECT MAX(timestamp)
                                                                       FROM results q
                                                                       WHERE r.benchmark = q.benchmark
                                                                         AND r.framework = q.framework
                                                                         AND r.preset = q.preset
                                                                         AND r.details = q.details)
                                                                         AND r.benchmark = '{benchmark}'
                                                                         AND preset = 'paper')
        SELECT time FROM recent_results WHERE framework = 'triton'
        """, conn)['time'].values
        other_times = pd.read_sql_query(f"""
                WITH recent_results AS (SELECT r.benchmark, r.framework, r.preset, r.details, r.time
                                                           FROM results r
                                                           WHERE timestamp == (SELECT MAX(timestamp)
                                                                               FROM results q
                                                                               WHERE r.benchmark = q.benchmark
                                                                                 AND r.framework = q.framework
                                                                                 AND r.preset = q.preset
                                                                                 AND r.details = q.details)
                                                                                 AND r.benchmark = '{benchmark}'
                                                                                 AND preset = 'paper')
                SELECT time FROM recent_results WHERE framework = '{framework}' AND details = '{details}'
                """, conn)['time'].values
        ci_lower, ci_upper = bootstrap_ratio_ci_unpaired(other_times, triton_times)
        err_low = speedup - ci_lower
        err_up = ci_upper - speedup

        ax.errorbar(xe, speedup, yerr=[[err_low], [err_up]], fmt='none',
                    ecolor='black', capsize=2, capthick=1, elinewidth=1, zorder=10)
        label_y = speedup * 1.3 if speedup >= 1 else speedup * 0.8

        # Compact scientific notation without 'x': e.g., "1.2e1"
        exp = int(np.floor(np.log10(abs(speedup)))) if speedup != 0 else 0
        mantissa = speedup / (10 ** exp)
        label = f'{mantissa:.1f}e{exp}'
        fontsize = 8

        ax.text(xe, label_y, label, ha='left' if speedup >= 1 else 'right', va='bottom' if speedup >= 1 else 'top',
                fontsize=fontsize, fontweight='bold', rotation=45)

    # Add horizontal line at y=1 (baseline: triton == best non-triton)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Benchmark', fontsize=12)
    # ax.set_ylabel('Triton Speedup vs Best Non-Triton Framework', fontsize=12)

    # Update title
    title = f'Triton Performance vs Best Alternative (Preset: {preset})'
    ax.set_title(title, fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')

    # Create legend with framework colors
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=framework_colors[fw], label=fw)
                       for fw in ['cupy', 'dace_gpu', 'jax', 'numpy']]
    ax.legend(handles=legend_elements, loc='upper right')

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

    # Update filename
    filename = f'triton-speedup-{preset}.pdf'

    plt.savefig(filename, dpi=600)
    print(f"Saved {filename}")
    plt.show()
