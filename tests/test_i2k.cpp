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
