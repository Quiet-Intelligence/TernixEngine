#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "ternix/cpu_features.hpp"
#include "ternix/i2k.hpp"

using namespace ternix;

namespace {

std::vector<Isa> available_isas() {
    std::vector<Isa> v{Isa::Scalar};
    if (cpu_info().avx2) v.push_back(Isa::Avx2);
    if (cpu_info().avx_vnni) v.push_back(Isa::AvxVnni);
    if (cpu_info().avx512_vnni) v.push_back(Isa::Avx512Vnni);   // absent on the development machine: these cases run only on AVX-512 hardware
    return v;
}

std::vector<int8_t> random_ternary(size_t n, uint32_t seed, int zero_pct = 33) {
    std::mt19937 rng(seed);
    std::vector<int8_t> w(n);
    for (auto& x : w) {
        const int r = static_cast<int>(rng() % 100);
        x = r < zero_pct ? 0 : ((rng() & 1) ? 1 : -1);
    }
    return w;
}

std::vector<int8_t> random_i8(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int8_t> a(n);
    for (auto& x : a) x = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
    return a;
}

// Independent reference on the unpacked ternary matrix.
std::vector<int32_t> ref(const std::vector<int8_t>& w, const std::vector<int8_t>& q, size_t MR, size_t N, size_t K) {
    std::vector<int32_t> y(MR * N);
    for (size_t m = 0; m < MR; ++m)
        for (size_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) acc += q[m * K + k] * w[n * K + k];
            y[m * N + n] = acc;
        }
    return y;
}

int32_t sum_row(const int8_t* q, size_t K) {
    int32_t s = 0;
    for (size_t k = 0; k < K; ++k) s += q[k];
    return s;
}

} // namespace

TEST(I2K, PackKnownAnswer) {
    // column 5 of block 0 is stored in byte 5, bit pair 0; column 32 + 5 in byte 5, bit pair 1; etc.
    const size_t N = 1, K = 128;
    std::vector<int8_t> w(K, 0);
    w[5] = 1;            // code 2 at byte 5, pair 0
    w[32 + 5] = -1;      // code 0 at byte 5, pair 1
    w[64 + 5] = 1;       // code 2 at byte 5, pair 2
    w[96 + 5] = -1;      // code 0 at byte 5, pair 3
    std::vector<uint8_t> p(K / 4);
    i2k::pack(w.data(), p.data(), N, K);
    EXPECT_EQ(p[5], (2u << 0) | (0u << 2) | (2u << 4) | (0u << 6));
    EXPECT_EQ(p[4], (1u << 0) | (1u << 2) | (1u << 4) | (1u << 6));   // all-zero weights are code 1
}

TEST(I2K, PackUnpackRoundTrip) {
    const size_t N = 7, K = 384;
    auto w = random_ternary(N * K, 1);
    std::vector<uint8_t> p(N * K / 4);
    i2k::pack(w.data(), p.data(), N, K);
    std::vector<int8_t> u(K);
    for (size_t n = 0; n < N; ++n) {
        i2k::unpack_row(p.data() + n * K / 4, u.data(), K);
        for (size_t k = 0; k < K; ++k) ASSERT_EQ(u[k], w[n * K + k]) << n << "," << k;
    }
}

TEST(I2K, GemmRowsMatchesReferenceAllIsas) {
    const size_t Ks[] = {128, 256, 640, 2560};
    const size_t Ns[] = {1, 2, 3, 4, 5, 8, 17, 64};
    uint32_t seed = 10;
    for (Isa isa : available_isas())
        for (size_t K : Ks)
            for (size_t N : Ns)
                for (size_t MR = 1; MR <= 4; ++MR) {
                    auto w = random_ternary(N * K, seed++);
                    auto q = random_i8(MR * K, seed++);
                    std::vector<uint8_t> p(N * K / 4);
                    i2k::pack(w.data(), p.data(), N, K);
                    std::vector<int32_t> sums(MR);
                    for (size_t m = 0; m < MR; ++m) sums[m] = sum_row(&q[m * K], K);
                    std::vector<int32_t> got(MR * N, -7);
                    i2k::gemm_rows(p.data(), q.data(), sums.data(), MR, got.data(), N, 0, N, K, isa);
                    ASSERT_EQ(ref(w, q, MR, N, K), got) << isa_name(isa) << " K=" << K << " N=" << N << " MR=" << MR;
                }
}

TEST(I2K, GemvSubRangeAndStride) {
    // n0..n1 sub-range: output index is relative to n0
    const size_t N = 40, K = 256;
    auto w = random_ternary(N * K, 5);
    auto q = random_i8(K, 6);
    std::vector<uint8_t> p(N * K / 4);
    i2k::pack(w.data(), p.data(), N, K);
    const auto full = ref(w, q, 1, N, K);
    for (Isa isa : available_isas()) {
        std::vector<int32_t> got(13);
        i2k::gemv(p.data(), q.data(), sum_row(q.data(), K), got.data(), 9, 22, K, isa);
        for (size_t i = 0; i < 13; ++i) ASSERT_EQ(got[i], full[9 + i]) << isa_name(isa) << " i=" << i;
    }
}

TEST(I2K, ExtremeActivationsAndWeights) {
    // worst case magnitudes: q = -128 everywhere, W = -1 everywhere (code 0) and W = +1 everywhere (code 2)
    const size_t N = 8, K = 6912;
    for (int8_t wv : {int8_t(-1), int8_t(1), int8_t(0)})
        for (int8_t qv : {int8_t(-128), int8_t(127)}) {
            std::vector<int8_t> w(N * K, wv), q(K, qv);
            std::vector<uint8_t> p(N * K / 4);
            i2k::pack(w.data(), p.data(), N, K);
            for (Isa isa : available_isas()) {
                std::vector<int32_t> got(N);
                i2k::gemv(p.data(), q.data(), sum_row(q.data(), K), got.data(), 0, N, K, isa);
                for (auto g : got) ASSERT_EQ(g, static_cast<int32_t>(K) * wv * qv) << isa_name(isa);
            }
        }
}

TEST(I2K, QuantizeActMatchesScalarAndRoundsLikeBitLinear) {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (size_t K : {1u, 7u, 31u, 32u, 33u, 128u, 2560u, 6912u}) {
        std::vector<float> x(K);
        for (auto& v : x) v = nd(rng);
        std::vector<int8_t> qs(K), qa(K);
        float ds, da;
        int32_t ss, sa;
        i2k::quantize_act(x.data(), qs.data(), K, &ds, &ss, Isa::Scalar);
        if (cpu_info().avx2) {
            i2k::quantize_act(x.data(), qa.data(), K, &da, &sa, Isa::Avx2);
            EXPECT_EQ(qs, qa) << "K=" << K;
            EXPECT_EQ(ds, da);
            EXPECT_EQ(ss, sa);
        }
        // independent re-derivation
        float amax = 0;
        for (float v : x) amax = std::max(amax, std::fabs(v));
        amax = std::max(amax, 1e-5f);
        const float s = 127.0f / amax;
        int32_t sum = 0;
        for (size_t i = 0; i < K; ++i) {
            const int e = static_cast<int>(std::max(-128.f, std::min(127.f, std::nearbyint(x[i] * s))));
            ASSERT_EQ(qs[i], e);
            sum += e;
        }
        EXPECT_EQ(ss, sum);
        // the largest-magnitude element maps to +/-127
        EXPECT_EQ(*std::max_element(qs.begin(), qs.end(), [](int8_t a, int8_t b) { return std::abs(a) < std::abs(b); }) != 0, true);
    }
}

TEST(I2K, QuantizeZerosAndTinyInputs) {
    std::vector<float> x(64, 0.f);
    std::vector<int8_t> q(64, 55);
    float d;
    int32_t s;
    for (Isa isa : available_isas()) {
        i2k::quantize_act(x.data(), q.data(), 64, &d, &s, isa);
        for (auto v : q) ASSERT_EQ(v, 0);
        EXPECT_EQ(s, 0);
    }
}

TEST(I2K, SparseKMajorMatchesDenseExactly) {
    std::mt19937 rng(21);
    for (size_t K : {128u, 640u, 6912u})
        for (size_t N : {128u, 256u, 640u, 2560u})
            for (int zero_pct : {0, 50, 80, 95, 100}) {
                auto w = random_ternary(N * K, static_cast<uint32_t>(rng()));
                std::vector<int8_t> q(K);
                for (auto& v : q) v = (static_cast<int>(rng() % 100) < zero_pct) ? 0 : static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
                std::vector<uint8_t> wp(N * K / 4), wk(K * N / 4);
                i2k::pack(w.data(), wp.data(), N, K);
                i2k::pack_kmajor_from_i2k(wp.data(), wk.data(), N, K);
                std::vector<int32_t> dense(N);
                i2k::gemv(wp.data(), q.data(), sum_row(q.data(), K), dense.data(), 0, N, K, Isa::Scalar);
                std::vector<uint32_t> idx(K);
                int32_t sq = 0;
                const size_t cnt = i2k::nonzero_indices(q.data(), K, idx.data(), &sq);
                EXPECT_EQ(sq, sum_row(q.data(), K));
                for (Isa isa : available_isas()) {
                    std::vector<int32_t> acc(N, 0);
                    i2k::gemv_sparse_accumulate(wk.data(), N, q.data(), idx.data(), cnt, acc.data(), isa);
                    for (size_t n = 0; n < N; ++n) acc[n] -= sq;
                    ASSERT_EQ(acc, dense) << isa_name(isa) << " K=" << K << " N=" << N << " zeros=" << zero_pct << "% nnz=" << cnt;
                }
            }
}

TEST(I2K, GroupScaledGemmMatchesDoubleReference) {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.5f);
    for (size_t K : {128u, 384u, 2048u})
        for (size_t N : {1u, 3u, 8u})
            for (size_t MR = 1; MR <= 4; ++MR) {
                const size_t nb = K / 128;
                auto w = random_ternary(N * K, static_cast<uint32_t>(rng()));
                std::vector<uint8_t> wp(N * K / 4);
                i2k::pack(w.data(), wp.data(), N, K);
                std::vector<float> scale_f(N * nb);
                std::vector<uint16_t> gs(N * nb);
                for (size_t i = 0; i < gs.size(); ++i) {
                    const float s = 0.01f + 0.001f * static_cast<float>(rng() % 50);
                    gs[i] = static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(s), _MM_FROUND_TO_NEAREST_INT), 0));
                    scale_f[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(gs[i])));
                }
                std::vector<float> x(MR * K);
                for (auto& v : x) v = nd(rng);
                std::vector<int8_t> q(MR * K);
                std::vector<float> deq(MR * nb);
                std::vector<int32_t> sq(MR * nb);
                for (size_t m = 0; m < MR; ++m) i2k::quantize_act_blocks(x.data() + m * K, q.data() + m * K, K, deq.data() + m * nb, sq.data() + m * nb, Isa::Scalar);
                std::vector<double> ref(MR * N, 0.0);
                for (size_t m = 0; m < MR; ++m)
                    for (size_t n = 0; n < N; ++n)
                        for (size_t b = 0; b < nb; ++b) {
                            int64_t d = 0;
                            for (size_t k = b * 128; k < (b + 1) * 128; ++k) d += static_cast<int64_t>(q[m * K + k]) * w[n * K + k];
                            ref[m * N + n] += static_cast<double>(scale_f[n * nb + b]) * deq[m * nb + b] * static_cast<double>(d);
                        }
                for (Isa isa : available_isas()) {
                    std::vector<float> y(MR * N, -1.f);
                    i2k::gemm_rows_gs(wp.data(), gs.data(), q.data(), deq.data(), sq.data(), MR, y.data(), N, 0, N, K, isa);
                    for (size_t i = 0; i < y.size(); ++i)
                        ASSERT_NEAR(y[i], ref[i], 2e-4 * (1.0 + std::fabs(ref[i]))) << isa_name(isa) << " K=" << K << " N=" << N << " MR=" << MR << " i=" << i;
                }
            }
}

TEST(I2K, BlockQuantizeMatchesPerBlockTokenQuantize) {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 2.f);
    const size_t K = 512;
    std::vector<float> x(K);
    for (auto& v : x) v = nd(rng);
    x[200] = 40.f;                                   // an outlier only affects its own block
    std::vector<int8_t> q(K);
    std::vector<float> deq(K / 128);
    std::vector<int32_t> sum(K / 128);
    for (Isa isa : available_isas()) {
        i2k::quantize_act_blocks(x.data(), q.data(), K, deq.data(), sum.data(), isa);
        for (size_t b = 0; b < K / 128; ++b) {
            std::vector<int8_t> qb(128);
            float d;
            int32_t s;
            i2k::quantize_act(x.data() + b * 128, qb.data(), 128, &d, &s, Isa::Scalar);
            EXPECT_EQ(deq[b], d);
            EXPECT_EQ(sum[b], s);
            for (size_t i = 0; i < 128; ++i) ASSERT_EQ(q[b * 128 + i], qb[i]);
        }
        EXPECT_GT(deq[1], 5 * deq[0]);               // block 1 holds the outlier
    }
}

namespace ternix { namespace i2k {
// src/i2k_avx512.cpp compiled against scalar-emulated AVX-512 intrinsics (tests/i2k_avx512_emulated.cpp)
void gemm_rows_avx512_emulated(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride, size_t n0, size_t n1, size_t K);
} }

// The AVX-512 kernel's indexing, block pairing (two 128-column blocks per 64-byte load), odd-block tail and reduction, checked on any CPU.
// The intrinsics themselves are emulated here, so this does not exercise the hardware instructions.
TEST(I2K, EmulatedAvx512KernelMatchesScalar) {
    uint32_t seed = 77;
    for (size_t K : {128, 256, 384, 640, 2560, 6912})
        for (size_t N : {1, 2, 3, 4, 5, 8, 17, 37})
            for (size_t MR = 1; MR <= 4; ++MR) {
                auto w = random_ternary(N * K, seed++);
                auto q = random_i8(MR * K, seed++);
                std::vector<uint8_t> p(N * K / 4);
                i2k::pack(w.data(), p.data(), N, K);
                std::vector<int32_t> sums(MR);
                for (size_t m = 0; m < MR; ++m) sums[m] = sum_row(&q[m * K], K);
                std::vector<int32_t> got(MR * N, -7), want(MR * N, -9);
                i2k::gemm_rows_avx512_emulated(p.data(), q.data(), sums.data(), MR, got.data(), N, 0, N, K);
                i2k::gemm_rows(p.data(), q.data(), sums.data(), MR, want.data(), N, 0, N, K, Isa::Scalar);
                ASSERT_EQ(want, got) << "K=" << K << " N=" << N << " MR=" << MR;
            }
    // a column range that does not start at row 0
    const size_t K = 640, N = 13;
    auto w = random_ternary(N * K, 5);
    auto q = random_i8(K, 6);
    std::vector<uint8_t> p(N * K / 4);
    i2k::pack(w.data(), p.data(), N, K);
    const int32_t sum = sum_row(q.data(), K);
    std::vector<int32_t> got(7, -1), want(7, -2);
    i2k::gemm_rows_avx512_emulated(p.data(), q.data(), &sum, 1, got.data(), 7, 4, 11, K);
    i2k::gemm_rows(p.data(), q.data(), &sum, 1, want.data(), 7, 4, 11, K, Isa::Scalar);
    EXPECT_EQ(want, got);
}

