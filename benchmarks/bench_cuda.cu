// GPU benchmark for the ternary GEMM kernels, timed with CUDA events (kernel time only,
// data resident on device, median of timed launches after warm-up). Each result is
// verified against the CPU reference before it is reported.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "ternix/bitpack.hpp"
#include "ternix/cuda_kernels.cuh"
#include "ternix/mul_mat_ternary.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1);} } while (0)

template <typename F>
static double median_ms(F launch, int warmup, int iters) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s));
    CK(cudaEventCreate(&e));
    for (int i = 0; i < warmup; ++i) launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(s));
        launch();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        CK(cudaEventElapsedTime(&t[i], s, e));
    }
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(s));
    CK(cudaEventDestroy(e));
    return t[iters / 2];
}

int main(int argc, char** argv) {
    // optional: argv[1] = path of a JSON file to write the results to
    std::FILE* json = (argc > 1) ? std::fopen(argv[1], "w") : nullptr;
    bool first_row = true;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    if (json) std::fprintf(json, "{\"device\": \"%s\", \"sm\": \"%d%d\", \"results\": [", p.name, p.major, p.minor);
    std::printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
    std::printf("%6s %12s %12s %9s %12s %s\n", "N", "naive ms", "tiled ms", "speedup", "tiled TOPS", "verified");
    const int sizes[] = {512, 1024, 2048, 4096};
    for (int n : sizes) {
        const int M = n, K = n, N = n;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> act(-128, 127), wt(-1, 1);
        std::vector<int8_t> A(static_cast<size_t>(M) * K), W(static_cast<size_t>(K) * N);
        for (auto& a : A) a = static_cast<int8_t>(act(rng));
        for (auto& w : W) w = static_cast<int8_t>(wt(rng));
        std::vector<uint8_t> Wp(W.size() / 4);
        ternix::pack_ternary_weights(Wp.data(), W.data(), W.size());

        int8_t* dA; uint8_t* dB; int32_t *dC1, *dC2;
        CK(cudaMalloc(&dA, A.size()));
        CK(cudaMalloc(&dB, Wp.size()));
        CK(cudaMalloc(&dC1, sizeof(int32_t) * M * N));
        CK(cudaMalloc(&dC2, sizeof(int32_t) * M * N));
        CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, Wp.data(), Wp.size(), cudaMemcpyHostToDevice));

        const double t_naive = median_ms([&] { CK(ternix::mul_mat_ternary_cuda_naive(dA, dB, dC1, M, K, N)); }, 3, 11);
        const double t_tiled = median_ms([&] { CK(ternix::mul_mat_ternary_cuda_tiled(dA, dB, dC2, M, K, N)); }, 5, 31);

        // exactness vs the CPU blocked AVX2 kernel (itself unit-tested against a plain integer GEMM)
        std::vector<int32_t> ref(static_cast<size_t>(M) * N), g1(ref.size()), g2(ref.size());
        ternix::mul_mat_ternary_avx2_blocked(A.data(), Wp.data(), ref.data(), M, K, N);
        CK(cudaMemcpy(g1.data(), dC1, g1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g2.data(), dC2, g2.size() * 4, cudaMemcpyDeviceToHost));
        const bool ok = (ref == g1) && (ref == g2);
        const double tops = 2.0 * M * N * K / (t_tiled * 1e-3) / 1e12;
        std::printf("%6d %12.3f %12.3f %8.1fx %12.2f %s\n", n, t_naive, t_tiled, t_naive / t_tiled, tops, ok ? "yes" : "NO");
        if (json) {
            std::fprintf(json, "%s{\"n\": %d, \"naive_ms\": %.4f, \"tiled_ms\": %.4f, \"tiled_tops\": %.3f, \"verified\": %s}",
                         first_row ? "" : ", ", n, t_naive, t_tiled, tops, ok ? "true" : "false");
            first_row = false;
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC1); cudaFree(dC2);
        if (!ok) return 2;
    }
    if (json) { std::fprintf(json, "]}\n"); std::fclose(json); }
    return 0;
}
