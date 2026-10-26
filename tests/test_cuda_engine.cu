// Kernel-level checks of the CUDA engine against the CPU kernels (exact for integer paths).
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "ternix/cuda_engine.hpp"
#include "ternix/i2k.hpp"

using namespace ternix;

TEST(CudaEngine, TernaryGemvMatchesCpuKernel) {
    std::mt19937 rng(11);
    for (int nt : {1, 2, 5, 8})
        for (auto [N, K] : {std::pair<int, int>{16, 128}, {37, 256}, {640, 2560}, {96, 6912}}) {
            std::vector<int8_t> w(size_t(N) * K), q(size_t(nt) * K);
            for (auto& v : w) v = static_cast<int8_t>(int(rng() % 3) - 1);
            for (auto& v : q) v = static_cast<int8_t>(int(rng() % 256) - 128);
            std::vector<uint8_t> wp(size_t(N) * K / 4);
            i2k::pack(w.data(), wp.data(), N, K);
            std::vector<int32_t> sumq(nt);
            std::vector<float> deq(nt);
            for (int t = 0; t < nt; ++t) {
                for (int k = 0; k < K; ++k) sumq[t] += q[size_t(t) * K + k];
                deq[t] = 0.01f * float(t + 1);
            }
            const float scale = 0.037f;
            std::vector<float> y(size_t(nt) * N);
            cuda_test_ternary_gemv(wp.data(), q.data(), deq.data(), sumq.data(), scale, y.data(), nt, N, K);
            for (int t = 0; t < nt; ++t) {
                std::vector<int32_t> ref(N);
                i2k::gemv(wp.data(), q.data() + size_t(t) * K, sumq[t], ref.data(), 0, N, K, Isa::Scalar);
                for (int n = 0; n < N; ++n)
                    ASSERT_FLOAT_EQ(y[size_t(t) * N + n], static_cast<float>(ref[n]) * (deq[t] * scale)) << "nt=" << nt << " N=" << N << " K=" << K << " t=" << t << " n=" << n;
            }
        }
}

TEST(CudaEngine, RmsnormQuantMatchesCpu) {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 2.f);
    for (int K : {128, 2560, 6912}) {
        const int nt = 3;
        std::vector<float> x(size_t(nt) * K), w(K);
        for (auto& v : x) v = nd(rng);
        for (auto& v : w) v = 0.5f + std::fabs(nd(rng));
        std::vector<int8_t> q(size_t(nt) * K);
        std::vector<float> deq(nt);
        std::vector<int32_t> sumq(nt);
        cuda_test_rmsnorm_quant(x.data(), w.data(), 1e-5f, nt, K, q.data(), deq.data(), sumq.data());
        for (int t = 0; t < nt; ++t) {
            std::vector<float> xn(K);
            float ss = 0;
            for (int k = 0; k < K; ++k) ss += x[size_t(t) * K + k] * x[size_t(t) * K + k];
            const float inv = 1.0f / std::sqrt(ss / float(K) + 1e-5f);
            for (int k = 0; k < K; ++k) xn[k] = (x[size_t(t) * K + k] * inv) * w[k];
            std::vector<int8_t> qc(K);
            float dc;
            int32_t sc;
            i2k::quantize_act(xn.data(), qc.data(), K, &dc, &sc, Isa::Scalar);
            int mism = 0;
            for (int k = 0; k < K; ++k) mism += qc[k] != q[size_t(t) * K + k];
            EXPECT_LE(mism, 2) << "K=" << K << " t=" << t;   // rounding-boundary ties only (summation order of the norm)
            EXPECT_NEAR(deq[t], dc, 1e-4f * dc);
        }
    }
}

TEST(CudaEngine, TensorCoreGemmMatchesCpuKernelExactly) {
    std::mt19937 rng(31);
    for (int M : {9, 16, 17, 33, 48, 64})
        for (auto [N, K] : {std::pair<int, int>{64, 128}, {128, 256}, {640, 2560}, {192, 6912}}) {
            std::vector<int8_t> w(size_t(N) * K), q(size_t(M) * K);
            for (auto& v : w) v = static_cast<int8_t>(int(rng() % 3) - 1);
            for (auto& v : q) v = static_cast<int8_t>(int(rng() % 256) - 128);
            std::vector<uint8_t> wp(size_t(N) * K / 4);
            i2k::pack(w.data(), wp.data(), N, K);
            std::vector<int32_t> sumq(M);
            std::vector<float> deq(M);
            for (int t = 0; t < M; ++t) {
                for (int k = 0; k < K; ++k) sumq[t] += q[size_t(t) * K + k];
                deq[t] = 0.01f * float(t + 1);
            }
            const float scale = 0.037f;
            std::vector<float> y(size_t(M) * N);
            cuda_test_ternary_mma(wp.data(), q.data(), deq.data(), sumq.data(), scale, y.data(), M, N, K);
            for (int t = 0; t < M; ++t) {
                std::vector<int32_t> ref(N);
                i2k::gemv(wp.data(), q.data() + size_t(t) * K, sumq[t], ref.data(), 0, N, K, Isa::Scalar);
                for (int n = 0; n < N; ++n)
                    ASSERT_FLOAT_EQ(y[size_t(t) * N + n], static_cast<float>(ref[n]) * (deq[t] * scale)) << "M=" << M << " N=" << N << " K=" << K << " t=" << t << " n=" << n;
            }
        }
}
