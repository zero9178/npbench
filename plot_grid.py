import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from npbench.infrastructure import utilities as util

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rows", type=int, default=6, help="Number of rows in the grid")
parser.add_argument("-c", "--cols", type=int, default=9, help="Number of columns in the grid")
parser.add_argument("--latex", action="store_true", default=True,
                    help="Generate figures sized for LaTeX two-column paper")
args = parser.parse_args()


def bootstrap_ci(data, statfunction=np.median, alpha=0.05, n_samples=300):
    def bootstrap_ids(data, n_samples=100):
        for _ in range(n_samples):
            yield np.random.randint(data.shape[0], size=(data.shape[0],))

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


database = r"npbench_2070.db"
conn = util.create_connection(database)
data = pd.read_sql_query("SELECT * FROM results", conn)

data = data.drop(['timestamp', 'kind', 'dwarf', 'version'],
                 axis=1).reset_index(drop=True)

data = data[data["domain"] != ""]

data = data[data['preset'] == 'paper']
data = data.drop(['preset'], axis=1).reset_index(drop=True)

aggdata = data.groupby(["benchmark", "domain", "framework", "mode", "details"],
                       dropna=False).agg({
    "time": "median",
    "validated": "first"
}).reset_index()
best = aggdata.sort_values("time").groupby(
    ["benchmark", "domain", "framework", "mode"],
    dropna=False).first().reset_index()
bestgroup = best.drop(["time", "validated"], axis=1)
data = pd.merge(left=bestgroup,
                right=data,
                on=["benchmark", "domain", "framework", "mode", "details"],
                how="inner")
data = data.drop(['mode', 'details'], axis=1).reset_index(drop=True)

data = data[data['framework'] != 'dace_cpu']
data = data[data['framework'] != 'pythran']
data = data[data['framework'] != 'numba']

frmwrks = list(data['framework'].unique())
assert ('numpy' in frmwrks)
frmwrks.remove('numpy')
frmwrks.sort()

benchmarks_unsorted = list(data['benchmark'].unique())
results = []

for benchmark in benchmarks_unsorted:
    bench_data = data[data['benchmark'] == benchmark]

    numpy_times = bench_data[bench_data['framework'] == 'numpy']['time'].values
    if len(numpy_times) == 0:
        continue
    numpy_median = np.median(numpy_times)

    for framework in frmwrks:
        frmwrk_data = bench_data[bench_data['framework'] == framework]
        frmwrk_times = frmwrk_data['time'].values

        if len(frmwrk_times) == 0:
            continue

        validated = frmwrk_data['validated'].all()

        speedups = numpy_median / frmwrk_times
        median_speedup = np.median(speedups)

        ci = bootstrap_ci(speedups, statfunction=np.median, alpha=0.05, n_samples=300)
        ci_lower = ci[0]
        ci_upper = ci[1]

        error_lower = median_speedup - ci_lower
        error_upper = ci_upper - median_speedup

        results.append({
            'benchmark': benchmark,
            'framework': framework,
            'speedup': median_speedup,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'error_lower': error_lower,
            'error_upper': error_upper,
            'validated': validated,
        })

results_df = pd.DataFrame(results)

triton_speedups = results_df[results_df['framework'] == 'triton'][['benchmark', 'speedup']]
triton_speedups = triton_speedups.sort_values('speedup', ascending=False)
benchmarks = triton_speedups['benchmark'].tolist()

missing_benchmarks = [b for b in benchmarks_unsorted if b not in benchmarks]
benchmarks.extend(sorted(missing_benchmarks))
n_benchmarks = len(benchmarks)

# LaTeX two-column paper settings
# Single column width: ~3.5 inches, text height: ~9 inches
if args.latex:
    n_cols = 4
    n_rows = 7
    fig_width = 3.5  # Single column width in inches
    fig_height = 9.0  # Fits within text height
    benchmarks_per_fig = n_cols * n_rows
    n_figures = math.ceil(n_benchmarks / benchmarks_per_fig)
else:
    n_cols = args.cols
    n_rows = args.rows
    fig_width = 2.5 * n_cols
    fig_height = 2.5 * n_rows
    n_figures = 1
    benchmarks_per_fig = n_cols * n_rows

framework_colors = {
    'cupy': '#17becf',
    'dace_cpu': '#1f77b4',
    'dace_gpu': '#9467bd',
    'numba': '#1f77b4',
    'pythran': '#2ca02c',
    'triton': '#ff7f0e',
    'jax': '#d62728',
}
color_map = {fw: framework_colors.get(fw, '#808080') for fw in frmwrks}

for fig_num in range(n_figures):
    start_idx = fig_num * benchmarks_per_fig
    end_idx = min(start_idx + benchmarks_per_fig, n_benchmarks)
    fig_benchmarks = benchmarks[start_idx:end_idx]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Calculate row y-limits for this figure
    row_y_limits = []
    for row in range(n_rows):
        row_min = float('inf')
        row_max = float('-inf')
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= len(fig_benchmarks):
                continue
            benchmark = fig_benchmarks[idx]
            bench_results = results_df[results_df['benchmark'] == benchmark]
            for fw in frmwrks:
                fw_data = bench_results[bench_results['framework'] == fw]
                if len(fw_data) > 0:
                    speedup = fw_data['speedup'].values[0]
                    err_low = fw_data['error_lower'].values[0]
                    err_up = fw_data['error_upper'].values[0]
                    row_min = min(row_min, speedup - err_low)
                    row_max = max(row_max, speedup + err_up)
        if row_min == float('inf'):
            row_min = 0.1
        if row_max == float('-inf'):
            row_max = 10
        y_max = 10 ** math.ceil(math.log10(row_max * 3))
        y_min = 10 ** math.floor(math.log10(max(row_min * 0.3, 0.001)))
        row_y_limits.append((y_min, y_max))

    axes_flat = axes.flatten()

    did_not_validate_colour = 'gray'
    for idx, benchmark in enumerate(fig_benchmarks):
        row = idx // n_cols
        ax = axes_flat[idx]
        bench_results = results_df[results_df['benchmark'] == benchmark]

        x_positions = []
        speedups = []
        bar_colors = []
        labels = []
        errors_lower = []
        errors_upper = []
        validated_list = []
        failed_list = []

        for i, fw in enumerate(frmwrks):
            fw_data = bench_results[bench_results['framework'] == fw]
            if len(fw_data) > 0:
                x_positions.append(len(labels))
                speedups.append(fw_data['speedup'].values[0])
                bar_colors.append(color_map[fw])
                labels.append(fw)
                errors_lower.append(fw_data['error_lower'].values[0])
                errors_upper.append(fw_data['error_upper'].values[0])
                validated_list.append(fw_data['validated'].values[0])
                failed_list.append(False)
            else:
                x_positions.append(len(labels))
                bar_colors.append(color_map[fw])
                labels.append(fw)
                # Fake data
                speedups.append(1.0)
                errors_lower.append(0)
                errors_upper.append(0)
                validated_list.append(True)
                failed_list.append(True)

        if speedups:
            bottoms = []
            heights = []
            for s in speedups:
                if s >= 1:
                    bottoms.append(1)
                    heights.append(s - 1)
                else:
                    bottoms.append(s)
                    heights.append(1 - s)

            bars = ax.bar(x_positions, heights, bottom=bottoms, color=bar_colors, width=0.7)
            ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_yscale('log')

            if args.latex:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(np.log10(y))}' if y >= 1 else ''))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(np.log10(y))}' if y > 0 else ''))

            for j, (pos, speedup, validated, failed) in enumerate(
                    zip(x_positions, speedups, validated_list, failed_list)):
                if failed:
                    ax.scatter(pos, speedup, marker='x', s=50, c='red', linewidths=1.5, zorder=5)
                    continue
                if not validated:
                    ax.scatter(pos, speedup, marker='x', s=50, c=did_not_validate_colour, linewidths=1.5, zorder=5)

            for j, (pos, speedup, err_low, err_up, failed) in enumerate(
                    zip(x_positions, speedups, errors_lower, errors_upper, failed_list)):
                if failed:
                    continue
                ax.errorbar(pos, speedup, yerr=[[err_low], [err_up]], fmt='none',
                            ecolor='black', capsize=2, capthick=1, elinewidth=1, zorder=10)

            for j, (pos, speedup, failed) in enumerate(zip(x_positions, speedups, failed_list)):
                if failed:
                    continue
                label_y = speedup * 1.3 if speedup >= 1 else speedup * 0.7
                if args.latex:
                    # Compact scientific notation without 'x': e.g., "1.2e1"
                    exp = int(np.floor(np.log10(abs(speedup)))) if speedup != 0 else 0
                    mantissa = speedup / (10 ** exp)
                    label = f'{mantissa:.1f}e{exp}'
                    fontsize = 4
                else:
                    if speedup < 0.1:
                        label = f'{speedup:.0e}x'
                    else:
                        label = f'{speedup:.1f}x'
                    fontsize = 5
                ax.text(pos, label_y, label, ha='left', va='bottom' if speedup >= 1 else 'top',
                        fontsize=fontsize, fontweight='bold', rotation=45)

            ax.set_ylim(row_y_limits[row])

        title_fontsize = 9 if not args.latex else 7
        ax.set_title(benchmark, fontsize=title_fontsize, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        tick_fontsize = 7 if not args.latex else 5
        ax.set_xticklabels([fw[:3] for fw in labels], fontsize=tick_fontsize, rotation=45)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        if not args.latex:
            ax.set_ylabel('log10(Speedup)', fontsize=6)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.3)

    for idx in range(len(fig_benchmarks), n_rows * n_cols):
        axes_flat[idx].axis('off')

    if not args.latex:
        fig.suptitle('log10(Speedup) over NumPy by Benchmark', fontsize=16, fontweight='bold', y=0.995)

    legend_fontsize = 12 if not args.latex else 6

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[fw], label=fw) for fw in frmwrks]
    legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor=did_not_validate_colour,
                                  markeredgecolor=did_not_validate_colour, markersize=8, markeredgewidth=2,
                                  label='Did not validate'))
    legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                                  markeredgecolor='red', markersize=8, markeredgewidth=2, label='Failed to run'))
    legend_elements.append(Line2D([0], [0], color='black', marker='_', markersize=10,
                                  markeredgewidth=1.5, label='95% CI'))
    if args.latex:
        ncol = min(len(frmwrks) + 2, 4)
        legend = fig.legend(handles=legend_elements, loc='lower center', ncol=ncol,
                            fontsize=legend_fontsize, bbox_to_anchor=(0.5, -0.02), frameon=True,
                            fancybox=True, shadow=False, borderpad=0.5, handlelength=1, handletextpad=0.3)
    else:
        legend = fig.legend(handles=legend_elements, loc='lower center', ncol=len(frmwrks) + 2,
                            fontsize=legend_fontsize, bbox_to_anchor=(0.5, 0.01), frameon=True,
                            fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_linewidth(1.5 if not args.latex else 0.5)
    legend.get_frame().set_edgecolor('gray')

    if args.latex:
        plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=0.5, w_pad=0.3)
    else:
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])

    if args.latex:
        if n_figures > 1:
            filename_base = f'benchmark_grid_latex_{fig_num + 1}'
        else:
            filename_base = 'benchmark_grid_latex'
    else:
        filename_base = f'benchmark_grid_{n_rows}x{n_cols}'

    plt.savefig(f'{filename_base}.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved {filename_base}.pdf")
    if not args.latex:
        plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
        print(f"Saved {filename_base}.png")
    plt.close(fig)
