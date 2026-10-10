// CPU microbenchmarks for the ternary GEMM kernels.
// Inputs are seeded random int8 activations and random ternary weights packed in the
// K x (N/4) layout, so every implementation sees the same data. Square M=K=N problems,
// single thread. Each iteration also touches the output so nothing is optimised away.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <cstdlib>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif
#include <cstdint>
#include <random>
#include <vector>
#include "ternix/bitpack.hpp"
#include "ternix/mul_mat_ternary.h"

namespace {

struct Operands {
    int8_t* A;
    uint8_t* B;
    int32_t* C;
    size_t n;
    explicit Operands(size_t n_) : n(n_) {
        A = static_cast<int8_t*>(_mm_malloc(n * n, 32));
        B = static_cast<uint8_t*>(_mm_malloc(n * n / 4, 32));
        C = static_cast<int32_t*>(_mm_malloc(n * n * sizeof(int32_t), 32));
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> act(-127, 127), wt(-1, 1);
        for (size_t i = 0; i < n * n; ++i) A[i] = static_cast<int8_t>(act(rng));
        std::vector<int8_t> w(n * n);
        for (auto& x : w) x = static_cast<int8_t>(wt(rng));
        ternix::pack_ternary_weights(B, w.data(), w.size());
    }
    ~Operands() { _mm_free(A); _mm_free(B); _mm_free(C); }
};

template <void (*Kernel)(const int8_t*, const uint8_t*, int32_t*, size_t, size_t, size_t)>
void BM_Ternary(benchmark::State& state) {
    Operands op(static_cast<size_t>(state.range(0)));
    for (auto _ : state) {
        Kernel(op.A, op.B, op.C, op.n, op.n, op.n);
        benchmark::DoNotOptimize(op.C);
        benchmark::ClobberMemory();
    }
    // 2 ops (multiply + add) per ternary MAC
    state.SetItemsProcessed(state.iterations() * 2 * op.n * op.n * op.n);
}

} // namespace

BENCHMARK_TEMPLATE(BM_Ternary, ternix::mul_mat_ternary_naive)->Name("BM_NaiveTernary")->RangeMultiplier(2)->Range(128, 512)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Ternary, ternix::mul_mat_ternary_scalar)->Name("BM_ScalarTernary")->RangeMultiplier(2)->Range(128, 512)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Ternary, ternix::mul_mat_ternary_avx2)->Name("BM_AVX2Ternary")->RangeMultiplier(2)->Range(128, 512)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Ternary, ternix::mul_mat_ternary_avx2_blocked)->Name("BM_AVX2BlockedTernary")->RangeMultiplier(2)->Range(128, 512)->Unit(benchmark::kMillisecond);

// Hybrid CPUs (P + E cores) make single-thread timings depend on where the scheduler puts the
// thread: set TERNIX_BENCH_CPU=<logical cpu> to pin the benchmark thread to a P-core.
static void pin_if_requested() {
    const char* env = std::getenv("TERNIX_BENCH_CPU");
    if (!env) return;
    const int cpu = std::atoi(env);
#if defined(_WIN32)
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#endif
}

int main(int argc, char** argv) {
    pin_if_requested();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
