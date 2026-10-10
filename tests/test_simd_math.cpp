#include <gtest/gtest.h>
#include "ternix/simd_math.hpp"
#include <cmath>
#include <random>
#include <vector>

TEST(SimdMathTest, RmsNormKnownValue) {
    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4);
    ternix::rms_norm_avx2(out.data(), in.data(), weight.data(), 4, 1e-5f);
    const float expected_rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-5f);
    EXPECT_NEAR(out[0], 1.0f / expected_rms, 1e-4);
    EXPECT_NEAR(out[3], 4.0f / expected_rms, 1e-4);
}

TEST(SimdMathTest, RmsNormAvx2MatchesScalarAcrossSizes) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (size_t n : {1u, 7u, 8u, 9u, 64u, 100u, 4096u}) {
        std::vector<float> in(n), w(n), a(n), b(n);
        for (size_t i = 0; i < n; ++i) { in[i] = nd(rng); w[i] = nd(rng); }
        ternix::rms_norm_avx2(a.data(), in.data(), w.data(), n, 1e-5f);
        ternix::rms_norm_scalar(b.data(), in.data(), w.data(), n, 1e-5f);
        for (size_t i = 0; i < n; ++i) EXPECT_NEAR(a[i], b[i], 1e-4f * (1.0f + std::fabs(b[i]))) << "n=" << n << " i=" << i;
    }
}

TEST(SimdMathTest, Swiglu) {
    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(1);
    ternix::swiglu(out.data(), in.data(), 2);
    const float silu = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_NEAR(out[0], silu * 2.0f, 1e-4);
}
