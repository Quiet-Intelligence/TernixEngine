#!/usr/bin/env bash
# Linux/macOS-free equivalent of run_all.bat (GCC or Clang, Ninja optional).
#   ./run_all.sh               full pipeline
#   ./run_all.sh --build-only  configure + build only
# Env: PYTHON_EXE (default python3), TERNIX_BENCH_CPU (logical CPU to pin the benchmark to, default 2).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_EXE="${PYTHON_EXE:-python3}"
export TERNIX_BENCH_CPU="${TERNIX_BENCH_CPU:-2}"

echo "=== TernixEngine Dev Pipeline ==="
"$PYTHON_EXE" -m pip install --quiet numpy matplotlib pytest

echo "[1/6] Configure";   cmake -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v "$PYTHON_EXE")"
echo "[2/6] Build";       cmake --build build --parallel
[ "${1:-}" = "--build-only" ] && exit 0

echo "[3/6] C++ and CUDA unit tests"; ctest --test-dir build --output-on-failure
echo "[4/6] Python binding tests";    PYTHONPATH="$PWD/build" "$PYTHON_EXE" -m pytest tests/test_python_binding.py -q

echo "[5/6] Benchmarks (median of 5 repetitions)"
mkdir -p benchmarks
./build/bench_simd_math --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
    --benchmark_format=json --benchmark_out=benchmarks/results.json > /dev/null
[ -x build/bench_cuda ] && ./build/bench_cuda benchmarks/results_cuda.json

echo "[6/6] Charts and README tables"
"$PYTHON_EXE" scripts/plot_results.py --input benchmarks/results.json \
    --cuda-input benchmarks/results_cuda.json --output benchmarks --readme README.md
echo "Pipeline successful."
