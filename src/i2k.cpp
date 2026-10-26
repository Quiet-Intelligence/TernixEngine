#include "ternix/i2k.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ternix {
namespace i2k {

// ---------------------------------------------------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------------------------------------------------
void pack(const int8_t* w, uint8_t* out, size_t N, size_t K) {
    for (size_t n = 0; n < N; ++n) {
        uint8_t* row = out + n * (K / 4);
        for (size_t b = 0; b < K / kBlock; ++b)
            for (size_t l = 0; l < 32; ++l) {
                unsigned byte = 0;
                for (size_t j = 0; j < 4; ++j) byte |= static_cast<unsigned>(w[n * K + b * kBlock + 32 * j + l] + 1) << (2 * j);
                row[b * 32 + l] = static_cast<uint8_t>(byte);
            }
    }
}

void unpack_row(const uint8_t* row, int8_t* out, size_t K) {
    for (size_t b = 0; b < K / kBlock; ++b)
        for (size_t l = 0; l < 32; ++l)
            for (size_t j = 0; j < 4; ++j)
                out[b * kBlock + 32 * j + l] = static_cast<int8_t>(((row[b * 32 + l] >> (2 * j)) & 3) - 1);
}

// ---------------------------------------------------------------------------------------------------------------
// activation quantisation
// ---------------------------------------------------------------------------------------------------------------
static void quantize_act_scalar(const float* x, int8_t* q, size_t K, float* deq, int32_t* sum) {
    float amax = 0.f;
    for (size_t i = 0; i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 1e-5f);
    const float s = 127.0f / amax;
    int32_t acc = 0;
    for (size_t i = 0; i < K; ++i) {
        float v = std::nearbyint(x[i] * s);
        v = std::min(127.f, std::max(-128.f, v));
        q[i] = static_cast<int8_t>(v);
        acc += q[i];
    }
    *deq = 1.0f / s;
    *sum = acc;
}

static void quantize_act_avx2(const float* x, int8_t* q, size_t K, float* deq, int32_t* sum) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 vmax = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= K; i += 8) vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), abs_mask));
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    float amax = _mm_cvtss_f32(m);
    for (; i < K; ++i) amax = std::max(amax, std::fabs(x[i]));
    amax = std::max(amax, 1e-5f);
    const float s = 127.0f / amax;
    const __m256 vs = _mm256_set1_ps(s);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i sad = _mm256_setzero_si256();
    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));
    size_t j = 0;
    for (; j + 32 <= K; j += 32) {
        __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j), vs));
        __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j + 8), vs));
        __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j + 16), vs));
        __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j + 24), vs));
        __m256i ab = _mm256_packs_epi32(a, b);              // saturating int32 -> int16
        __m256i cd = _mm256_packs_epi32(c, d);
        __m256i p = _mm256_packs_epi16(ab, cd);             // saturating int16 -> int8 == clamp(-128, 127)
        p = _mm256_permutevar8x32_epi32(p, perm);           // undo the in-lane interleave of the pack instructions
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + j), p);
        sad = _mm256_add_epi64(sad, _mm256_sad_epu8(_mm256_xor_si256(p, bias), _mm256_setzero_si256()));
    }
    alignas(32) int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sad);
    int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3] - 128 * static_cast<int64_t>(j);  // (q + 128) summed
    for (; j < K; ++j) {
        float v = std::nearbyint(x[j] * s);
        v = std::min(127.f, std::max(-128.f, v));
        q[j] = static_cast<int8_t>(v);
        total += q[j];
    }
    *deq = 1.0f / s;
    *sum = static_cast<int32_t>(total);
}

void quantize_act(const float* x, int8_t* q, size_t K, float* deq_step, int32_t* sum_q, Isa isa) {
    if (isa == Isa::Scalar) quantize_act_scalar(x, q, K, deq_step, sum_q);
    else quantize_act_avx2(x, q, K, deq_step, sum_q);
}

// ---------------------------------------------------------------------------------------------------------------
// scalar kernels (reference semantics, also the fallback for CPUs without AVX2)
// ---------------------------------------------------------------------------------------------------------------
static void gemm_rows_scalar(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y,
                             size_t y_stride, size_t n0, size_t n1, size_t K) {
    for (size_t n = n0; n < n1; ++n) {
        const uint8_t* row = W + n * (K / 4);
        for (size_t m = 0; m < MR; ++m) {
            int32_t acc = 0;
            const int8_t* a = q + m * K;
            for (size_t b = 0; b < K / kBlock; ++b)
                for (size_t l = 0; l < 32; ++l) {
                    const unsigned byte = row[b * 32 + l];
                    const int8_t* ab = a + b * kBlock + l;
                    acc += ab[0] * static_cast<int>(byte & 3) + ab[32] * static_cast<int>((byte >> 2) & 3) +
                           ab[64] * static_cast<int>((byte >> 4) & 3) + ab[96] * static_cast<int>(byte >> 6);
                }
            y[m * y_stride + (n - n0)] = acc - sum_q[m];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// SIMD kernels. V selects vpdpbusd (AVX-VNNI) over the vpmaddubsw / vpmaddwd sequence (AVX2).
// ---------------------------------------------------------------------------------------------------------------
template <bool V>
static inline __m256i decode_dot(__m256i acc, __m256i x, const int8_t* ap) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i w0 = _mm256_and_si256(x, m3);
    const __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(x, 2), m3);
    const __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(x, 4), m3);
    const __m256i w3 = _mm256_and_si256(_mm256_srli_epi16(x, 6), m3);
    const __m256i* a = reinterpret_cast<const __m256i*>(ap);
    if constexpr (V) {
#if defined(__AVXVNNI__) || defined(_MSC_VER)
        acc = _mm256_dpbusd_avx_epi32(acc, w0, _mm256_loadu_si256(a));
        acc = _mm256_dpbusd_avx_epi32(acc, w1, _mm256_loadu_si256(a + 1));
        acc = _mm256_dpbusd_avx_epi32(acc, w2, _mm256_loadu_si256(a + 2));
        acc = _mm256_dpbusd_avx_epi32(acc, w3, _mm256_loadu_si256(a + 3));
        return acc;
#else
#error "AVX-VNNI intrinsics unavailable: compile src/i2k.cpp with -mavxvnni"
#endif
    } else {
        // each maddubs lane is at most 2 * (2 * 128) = 512 in magnitude, so four of them cannot overflow int16
        __m256i p = _mm256_maddubs_epi16(w0, _mm256_loadu_si256(a));
        p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w1, _mm256_loadu_si256(a + 1)));
        p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w2, _mm256_loadu_si256(a + 2)));
        p = _mm256_add_epi16(p, _mm256_maddubs_epi16(w3, _mm256_loadu_si256(a + 3)));
        return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
    }
}

static inline __m128i hsum4(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab = _mm256_hadd_epi32(a, b);
    const __m256i cd = _mm256_hadd_epi32(c, d);
    const __m256i s = _mm256_hadd_epi32(ab, cd);   // lane-wise partial sums: [a0 b0 c0 d0 | a1 b1 c1 d1]
    return _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
}

static inline int32_t hsum1(__m256i a) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// MR activation rows x NR weight rows per pass; the weight vector is decoded once per activation row set.
template <bool V, int MR, int NR>
static inline void tile(const uint8_t* W, const int8_t* q, const int32_t* sum_q, int32_t* y, size_t y_stride,
                        size_t n, size_t n0, size_t K) {
    __m256i acc[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) acc[m][r] = _mm256_setzero_si256();
    const size_t rb = K / 4, nb = K / kBlock;
    const uint8_t* wr[NR];
    for (int r = 0; r < NR; ++r) wr[r] = W + (n + r) * rb;
    for (size_t b = 0; b < nb; ++b) {
        for (int r = 0; r < NR; ++r) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr[r] + b * 32));
            for (int m = 0; m < MR; ++m) acc[m][r] = decode_dot<V>(acc[m][r], x, q + m * K + b * kBlock);
        }
    }
    for (int m = 0; m < MR; ++m) {
        if constexpr (NR == 4) {
            __m128i s = hsum4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
            s = _mm_sub_epi32(s, _mm_set1_epi32(sum_q[m]));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(y + m * y_stride + (n - n0)), s);
        } else {
            for (int r = 0; r < NR; ++r) y[m * y_stride + (n - n0) + r] = hsum1(acc[m][r]) - sum_q[m];
        }
    }
}

template <bool V, int MR>
static void gemm_rows_simd(const uint8_t* W, const int8_t* q, const int32_t* sum_q, int32_t* y, size_t y_stride,
                           size_t n0, size_t n1, size_t K) {
    constexpr int NR = (MR == 1) ? 4 : 2;
    size_t n = n0;
    for (; n + NR <= n1; n += NR) tile<V, MR, NR>(W, q, sum_q, y, y_stride, n, n0, K);
    for (; n < n1; ++n) tile<V, MR, 1>(W, q, sum_q, y, y_stride, n, n0, K);
}

template <bool V>
static void dispatch_mr(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride,
                        size_t n0, size_t n1, size_t K) {
    switch (MR) {
        case 1: gemm_rows_simd<V, 1>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        case 2: gemm_rows_simd<V, 2>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        case 3: gemm_rows_simd<V, 3>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        default: gemm_rows_simd<V, 4>(W, q, sum_q, y, y_stride, n0, n1, K); break;
    }
}

void gemm_rows(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride,
               size_t n0, size_t n1, size_t K, Isa isa) {
    switch (isa) {
        case Isa::Scalar: gemm_rows_scalar(W, q, sum_q, MR, y, y_stride, n0, n1, K); break;
        case Isa::Avx2: dispatch_mr<false>(W, q, sum_q, MR, y, y_stride, n0, n1, K); break;
        case Isa::AvxVnni: dispatch_mr<true>(W, q, sum_q, MR, y, y_stride, n0, n1, K); break;
    }
}

void gemv(const uint8_t* W, const int8_t* q, int32_t sum_q, int32_t* y, size_t n0, size_t n1, size_t K, Isa isa) {
    gemm_rows(W, q, &sum_q, 1, y, n1 - n0, n0, n1, K, isa);
}

} // namespace i2k
} // namespace ternix
