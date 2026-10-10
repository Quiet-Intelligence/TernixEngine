#include <gtest/gtest.h>
#include <immintrin.h>
#include <cstdint>
#include <random>
#include <vector>
#include "ternix/bitpack.hpp"
#include "ternix/mul_mat_ternary.h"

namespace {

struct Problem {
    size_t M, K, N;
    std::vector<int8_t> A;       // M x K activations
    std::vector<int8_t> W;       // K x N ternary weights {-1,0,1}, unpacked
    std::vector<uint8_t> Wp;     // K x (N/4) packed
};

Problem make_problem(size_t M, size_t K, size_t N, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> act(-127, 127);
    std::uniform_int_distribution<int> wt(-1, 1);
    Problem p{M, K, N, std::vector<int8_t>(M * K), std::vector<int8_t>(K * N), std::vector<uint8_t>(K * N / 4)};
    for (auto& a : p.A) a = static_cast<int8_t>(act(rng));
    for (auto& w : p.W) w = static_cast<int8_t>(wt(rng));
    ternix::pack_ternary_weights(p.Wp.data(), p.W.data(), p.W.size());
    return p;
}

// Independent reference: plain integer GEMM on the *unpacked* weights, so it
// does not share the packed-weight decode path with the kernels under test.
std::vector<int32_t> reference(const Problem& p) {
    std::vector<int32_t> C(p.M * p.N, 0);
    for (size_t i = 0; i < p.M; ++i)
        for (size_t k = 0; k < p.K; ++k) {
            const int32_t a = p.A[i * p.K + k];
            for (size_t j = 0; j < p.N; ++j) C[i * p.N + j] += a * p.W[k * p.N + j];
        }
    return C;
}

using Kernel = void (*)(const int8_t*, const uint8_t*, int32_t*, size_t, size_t, size_t);

void check(Kernel f, const char* name) {
    const size_t shapes[][3] = {{1, 8, 8}, {4, 16, 8}, {5, 33, 24}, {8, 64, 64}, {13, 100, 40}, {64, 256, 128}};
    uint32_t seed = 1;
    for (const auto& s : shapes) {
        Problem p = make_problem(s[0], s[1], s[2], seed++);
        std::vector<int32_t> expect = reference(p), got(p.M * p.N, -999);
        f(p.A.data(), p.Wp.data(), got.data(), p.M, p.K, p.N);
        EXPECT_EQ(expect, got) << name << " M=" << s[0] << " K=" << s[1] << " N=" << s[2];
    }
}

} // namespace

TEST(TernaryMatmulTest, NaiveMatchesReference) { check(ternix::mul_mat_ternary_naive, "naive"); }
TEST(TernaryMatmulTest, ScalarMatchesReference) { check(ternix::mul_mat_ternary_scalar, "scalar"); }
TEST(TernaryMatmulTest, Avx2MatchesReference) { check(ternix::mul_mat_ternary_avx2, "avx2"); }
TEST(TernaryMatmulTest, Avx2BlockedMatchesReference) { check(ternix::mul_mat_ternary_avx2_blocked, "avx2_blocked"); }

TEST(TernaryMatmulTest, ExtremeActivationsDoNotOverflow) {
    // int8 extremes accumulated over K=4096 stay inside int32.
    const size_t M = 4, K = 4096, N = 16;
    Problem p = make_problem(M, K, N, 99);
    for (auto& a : p.A) a = -128;
    std::vector<int32_t> expect = reference(p), got(M * N);
    ternix::mul_mat_ternary_avx2_blocked(p.A.data(), p.Wp.data(), got.data(), M, K, N);
    EXPECT_EQ(expect, got);
}
