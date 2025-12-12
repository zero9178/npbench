import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['text.usetex'] = True  # Disabled to avoid LaTeX issues

from npbench.infrastructure import utilities as util


if __name__ == "__main__":
    preset = 'paper'

    # create a database connection
    database = r"npbench.db"
    conn = util.create_connection(database)

    # SQL query to get triton speedup relative to best non-triton framework
    triton_speedup_query = """
    WITH triton_times AS (
        SELECT
            benchmark,
            preset,
            mode,
            MIN(time) as triton_time
        FROM results
        WHERE framework = 'triton'
        GROUP BY benchmark, preset, mode
    ),
    best_non_triton AS (
        SELECT r.benchmark,
               r.preset,
               r.mode,
               r.framework,
               r.time,
               ROW_NUMBER() OVER (PARTITION BY r.benchmark, r.preset, r.mode ORDER BY r.time) as rn
        FROM results r
        WHERE r.framework != 'triton'
    )
    SELECT
        t.benchmark,
        t.preset,
        t.mode,
        t.triton_time,
        b.time                 as best_non_triton_time,
        b.framework            as best_framework,
        b.time / t.triton_time as speedup
    FROM triton_times t
    JOIN best_non_triton b
        ON t.benchmark = b.benchmark
        AND t.preset = b.preset
        AND t.mode = b.mode
            AND b.rn = 1
    ORDER BY t.benchmark
    """

    data = pd.read_sql_query(triton_speedup_query, conn)

    # Filter by preset
    data = data[data['preset'] == preset]

    print(f"Total benchmarks with Triton speedup data: {len(data)}")

    # Prepare results for plotting
    benchmarks = sorted(data['benchmark'].unique())
    results = []

    for benchmark in benchmarks:
        bench_data = data[data['benchmark'] == benchmark]
        speedup = bench_data['speedup'].values[0]
        best_framework = bench_data['best_framework'].values[0]

        results.append({
            'benchmark': benchmark,
            'speedup': speedup,
            'best_framework': best_framework
        })

    results_df = pd.DataFrame(results)

    # Sort by speedup (highest first)
    results_df = results_df.sort_values('speedup', ascending=False).reset_index(drop=True)
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

    # Add horizontal line at y=1 (baseline: triton == best non-triton)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Triton Speedup vs Best Non-Triton Framework', fontsize=12)

    # Update title
    title = f'Triton Performance vs Best Alternative (Preset: {preset})'
    ax.set_title(title, fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')

    # Create legend with framework colors
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=framework_colors[fw], label=fw)
                       for fw in ['cupy', 'dace_gpu', 'jax', 'numpy']]
    ax.legend(handles=legend_elements, loc='upper left')

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
