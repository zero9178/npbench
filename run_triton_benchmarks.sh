#!/bin/bash
# Run all benchmarks with Triton implementations using triton, dace_gpu, and jax frameworks
# Uses the paper dataset with float32 precision

# All benchmarks with Triton implementations
BENCHMARKS=(
    "adi"
    "arc_distance"
    "atax"
    "azimint_hist"
    "azimint_naive"
    "bicg"
    "cavity_flow"
    "channel_flow"
    "cholesky"
    "cholesky2"
    "compute"
    "contour_integral"
    "conv2d_bias"
    "correlation"
    "covariance"
    "covariance2"
    "crc16"
    "deriche"
    "doitgen"
    "durbin"
    "fdtd_2d"
    "floyd_warshall"
    "gemm"
    "gemver"
    "gesummv"
    "go_fast"
    "gramschmidt"
    "hdiff"
    "heat_3d"
    "jacobi_1d"
    "jacobi_2d"
    "k2mm"
    "k3mm"
    "lenet"
    "lu"
    "ludcmp"
    "mandelbrot1"
    "mandelbrot2"
    "mlp"
    "mvt"
    "nbody"
    "nussinov"
    "resnet"
    "scattering_self_energies"
    "seidel_2d"
    "softmax"
    "spmv"
    "stockham_fft"
    "symm"
    "syr2k"
    "syrk"
    "trisolv"
    "trmm"
    "vadv"
)

FRAMEWORKS=("triton" "dace_gpu" "jax" "cupy")
PRESET="paper"
DATATYPE="float32"
REPEAT=10
TIMEOUT=200.0

# Output file for logging
LOG_FILE="triton_benchmarks_$(date +%Y%m%d_%H%M%S).log"

echo "Found ${#BENCHMARKS[@]} benchmarks with Triton implementations" | tee -a "$LOG_FILE"
echo "Running with frameworks: ${FRAMEWORKS[*]}" | tee -a "$LOG_FILE"
echo "Dataset: $PRESET, Datatype: $DATATYPE" | tee -a "$LOG_FILE"
echo "Repeat: $REPEAT, Timeout: ${TIMEOUT}s" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counters for summary
declare -A success_count
declare -A failed_count
for fw in "${FRAMEWORKS[@]}"; do
    success_count[$fw]=0
    failed_count[$fw]=0
done

# Run benchmarks
total=${#BENCHMARKS[@]}
current=0

for benchmark in "${BENCHMARKS[@]}"; do
    ((current++))
    for framework in "${FRAMEWORKS[@]}"; do
        echo "========================================" | tee -a "$LOG_FILE"
        echo "[$current/$total] Running $benchmark with $framework" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        # Run with uv for all frameworks
        uv run python run_benchmark.py \
            -b "$benchmark" \
            -f "$framework" \
            -p "$PRESET" \
            -d "$DATATYPE" \
            -r "$REPEAT" \
            -t "$TIMEOUT" \
            >> "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            ((success_count[$framework]++))
            echo "✓ SUCCESS: $benchmark with $framework" | tee -a "$LOG_FILE"
        else
            ((failed_count[$framework]++))
            echo "✗ FAILED: $benchmark with $framework" | tee -a "$LOG_FILE"
        fi
        echo "" | tee -a "$LOG_FILE"
    done
done

# Print summary
echo "================================================================================" | tee -a "$LOG_FILE"
echo "SUMMARY" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for framework in "${FRAMEWORKS[@]}"; do
    success=${success_count[$framework]}
    failed=${failed_count[$framework]}
    echo "${framework^^}:" | tee -a "$LOG_FILE"
    echo "  Success: $success/$total" | tee -a "$LOG_FILE"
    echo "  Failed:  $failed/$total" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Full logs saved to: $LOG_FILE" | tee -a "$LOG_FILE"
