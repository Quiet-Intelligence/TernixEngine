// GPU-vs-CPU exactness: both CUDA kernels must reproduce the integer reference bit-for-bit.
#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <vector>
#include "ternix/bitpack.hpp"
#include "ternix/cuda_kernels.cuh"
#include "ternix/mul_mat_ternary.h"

namespace {
void run_case(int M, int K, int N, bool tiled, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> act(-128, 127), wt(-1, 1);
    std::vector<int8_t> A(static_cast<size_t>(M) * K), W(static_cast<size_t>(K) * N);
    for (auto& a : A) a = static_cast<int8_t>(act(rng));
    for (auto& w : W) w = static_cast<int8_t>(wt(rng));
    std::vector<uint8_t> Wp(W.size() / 4);
    ternix::pack_ternary_weights(Wp.data(), W.data(), W.size());

    std::vector<int32_t> ref(static_cast<size_t>(M) * N, 0), got(ref.size(), -1);
    // CPU reference (itself unit-tested against a plain integer GEMM on unpacked weights)
    ternix::mul_mat_ternary_scalar(A.data(), Wp.data(), ref.data(), M, K, N);
    ASSERT_EQ(cudaSuccess, ternix::mul_mat_ternary_cuda_host(A.data(), Wp.data(), got.data(), M, K, N, tiled));
    EXPECT_EQ(ref, got) << (tiled ? "tiled" : "naive") << " M=" << M << " K=" << K << " N=" << N;
}
} // namespace

TEST(CudaTernaryTest, NaiveMatchesCpu) {
    run_case(64, 64, 64, false, 1);
    run_case(37, 100, 52, false, 2);          // ragged shapes are fine for the naive kernel
    run_case(128, 512, 256, false, 3);
}

TEST(CudaTernaryTest, TiledMatchesCpu) {
    run_case(64, 64, 64, true, 4);
    run_case(128, 256, 192, true, 5);
    run_case(256, 1024, 512, true, 6);
    run_case(64, 4096, 64, true, 7);          // deep K: many pipeline iterations
}

TEST(CudaTernaryTest, TiledRejectsUnsupportedShape) {
    EXPECT_EQ(cudaErrorInvalidValue, ternix::mul_mat_ternary_cuda_tiled(nullptr, nullptr, nullptr, 65, 64, 64));
}
