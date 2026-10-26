#include "ternix/i2k.hpp"
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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

static inline void quantize_act_scalar_or_avx(const float* x, int8_t* q, size_t K, float* d, int32_t* s, Isa isa) {
    if (isa == Isa::Scalar) quantize_act_scalar(x, q, K, d, s);
    else quantize_act_avx2(x, q, K, d, s);
}

void quantize_act_blocks(const float* x, int8_t* q, size_t K, float* deq, int32_t* sum, Isa isa) {
    (void)isa;
    for (size_t b = 0; b < K / kBlock; ++b) {
        int32_t s = 0;
        float d;
        // reuse the per-token routine on a 128-element block: identical rounding (rint) and clamp
        quantize_act_scalar_or_avx(x + b * kBlock, q + b * kBlock, kBlock, &d, &s, isa);
        deq[b] = d;
        sum[b] = s;
    }
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

// ---------------------------------------------------------------------------------------------------------------
// group-scaled GEMM: one 128-column block per scale
// ---------------------------------------------------------------------------------------------------------------
static inline float f16_to_f32_bits(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

static void gemm_rows_gs_scalar(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, size_t MR, float* y,
                                size_t y_stride, size_t n0, size_t n1, size_t K) {
    const size_t nb = K / kBlock;
    for (size_t n = n0; n < n1; ++n) {
        const uint8_t* row = W + n * (K / 4);
        for (size_t m = 0; m < MR; ++m) {
            double acc = 0;
            for (size_t b = 0; b < nb; ++b) {
                int32_t d = 0;
                for (size_t l = 0; l < 32; ++l) {
                    const unsigned byte = row[b * 32 + l];
                    const int8_t* a = q + m * K + b * kBlock + l;
                    d += a[0] * static_cast<int>(byte & 3) + a[32] * static_cast<int>((byte >> 2) & 3) + a[64] * static_cast<int>((byte >> 4) & 3) +
                         a[96] * static_cast<int>(byte >> 6);
                }
                acc += static_cast<double>(f16_to_f32_bits(gs[n * nb + b])) * deq[m * nb + b] * (d - sum_q[m * nb + b]);
            }
            y[m * y_stride + (n - n0)] = static_cast<float>(acc);
        }
    }
}

// MR activation rows x NR weight rows per tile: the weight bytes of NR rows and the activations of MR rows stay in registers / L1 across the K loop.
template <bool V, int MR, int NR>
static void gs_tile(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, float* y, size_t y_stride, size_t n,
                    size_t n0, size_t K) {
    const size_t nb = K / kBlock, rb = K / 4;
    const uint8_t* wr[NR];
    for (int r = 0; r < NR; ++r) wr[r] = W + (n + r) * rb;
    __m256 lanes[MR][NR];
    float corr[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) { lanes[m][r] = _mm256_setzero_ps(); corr[m][r] = 0.f; }
    for (size_t b = 0; b < nb; ++b) {
        __m256i x[NR];
        float s[NR];
        for (int r = 0; r < NR; ++r) {
            x[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr[r] + b * 32));
            s[r] = f16_to_f32_bits(gs[(n + r) * nb + b]);
        }
        for (int m = 0; m < MR; ++m) {
            const float d = deq[m * nb + b];
            const float qs = static_cast<float>(sum_q[m * nb + b]);
            for (int r = 0; r < NR; ++r) {
                const __m256i acc = decode_dot<V>(_mm256_setzero_si256(), x[r], q + m * K + b * kBlock);
                const float c = s[r] * d;
                lanes[m][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(c), lanes[m][r]);
                corr[m][r] += c * qs;
            }
        }
    }
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) {
            __m128 h = _mm_add_ps(_mm256_castps256_ps128(lanes[m][r]), _mm256_extractf128_ps(lanes[m][r], 1));
            h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
            y[m * y_stride + (n - n0) + r] = _mm_cvtss_f32(h) - corr[m][r];
        }
}

template <bool V, int MR>
static void gs_rows(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, float* y, size_t y_stride, size_t n0,
                    size_t n1, size_t K) {
    constexpr int NR = (MR == 1) ? 4 : 2;
    size_t n = n0;
    for (; n + NR <= n1; n += NR) gs_tile<V, MR, NR>(W, gs, q, deq, sum_q, y, y_stride, n, n0, K);
    for (; n < n1; ++n) gs_tile<V, MR, 1>(W, gs, q, deq, sum_q, y, y_stride, n, n0, K);
}

template <bool V>
static void gemm_rows_gs_simd(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, size_t MR, float* y,
                              size_t y_stride, size_t n0, size_t n1, size_t K) {
    switch (MR) {
        case 1: gs_rows<V, 1>(W, gs, q, deq, sum_q, y, y_stride, n0, n1, K); break;
        case 2: gs_rows<V, 2>(W, gs, q, deq, sum_q, y, y_stride, n0, n1, K); break;
        case 3: gs_rows<V, 3>(W, gs, q, deq, sum_q, y, y_stride, n0, n1, K); break;
        default: gs_rows<V, 4>(W, gs, q, deq, sum_q, y, y_stride, n0, n1, K); break;
    }
}

void gemm_rows_gs(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, size_t MR, float* y, size_t y_stride,
                  size_t n0, size_t n1, size_t K, Isa isa) {
    switch (isa) {
        case Isa::Scalar: gemm_rows_gs_scalar(W, gs, q, deq, sum_q, MR, y, y_stride, n0, n1, K); break;
        case Isa::Avx2: gemm_rows_gs_simd<false>(W, gs, q, deq, sum_q, MR, y, y_stride, n0, n1, K); break;
        case Isa::AvxVnni: gemm_rows_gs_simd<true>(W, gs, q, deq, sum_q, MR, y, y_stride, n0, n1, K); break;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// exact activation-sparse GEMV
// ---------------------------------------------------------------------------------------------------------------
void pack_kmajor_from_i2k(const uint8_t* i2k_rows, uint8_t* out, size_t N, size_t K) {
    // row k: 32-byte blocks of 128 outputs; byte l, bit pair j of block b holds output 128 b + 32 j + l (same trick as I2K128)
    std::memset(out, 0, K * N / 4);
    std::vector<int8_t> row(K);
    for (size_t n = 0; n < N; ++n) {
        unpack_row(i2k_rows + n * (K / 4), row.data(), K);
        const size_t byte = (n / 128) * 32 + (n % 32);
        const unsigned sh = static_cast<unsigned>((n % 128) / 32) * 2;
        for (size_t k = 0; k < K; ++k) out[k * (N / 4) + byte] |= static_cast<uint8_t>((row[k] + 1) << sh);
    }
}

#ifdef _MSC_VER
static inline unsigned ctz32(uint32_t v) { unsigned long r; _BitScanForward(&r, v); return static_cast<unsigned>(r); }
#else
static inline unsigned ctz32(uint32_t v) { return static_cast<unsigned>(__builtin_ctz(v)); }
#endif

size_t nonzero_indices(const int8_t* q, size_t K, uint32_t* idx, int32_t* sum_q) {
    size_t cnt = 0;
    int32_t sum = 0;
    size_t k = 0;
    for (; k + 32 <= K; k += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + k));
        uint32_t m = ~static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_setzero_si256())));
        while (m) {
            const unsigned b = ctz32(m);
            idx[cnt++] = static_cast<uint32_t>(k + b);
            sum += q[k + b];
            m &= m - 1;
        }
    }
    for (; k < K; ++k)
        if (q[k]) { idx[cnt++] = static_cast<uint32_t>(k); sum += q[k]; }
    *sum_q = sum;
    return cnt;
}

static void sparse_accumulate_scalar(const uint8_t* kmajor, size_t N, const int8_t* q, const uint32_t* idx, size_t count, int32_t* acc) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* row = kmajor + static_cast<size_t>(idx[i]) * (N / 4);
        const int a = q[idx[i]];
        for (size_t n = 0; n < N; ++n) acc[n] += a * ((row[(n / 128) * 32 + (n % 32)] >> (((n % 128) / 32) * 2)) & 3);
    }
}

// Rows are handled two at a time: the 2-bit codes of both rows are decoded (shift + mask), interleaved byte-wise and multiplied
// against the broadcast activation pair with vpmaddubsw (|pair sum| <= 508). int16 accumulation is safe for up to 64 pairs, so each
// chunk of 64 rows keeps eight int16 vectors per 128 outputs in registers and widens to int32 once per chunk and block.
static void sparse_accumulate_avx2(const uint8_t* kmajor, size_t N, const int8_t* q, const uint32_t* idx, size_t count, int32_t* acc) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const size_t nb = N / 128, rb = N / 4;
    alignas(32) __m256i avs[32];
    const uint8_t* rows[64];
    for (size_t c0 = 0; c0 < count; c0 += 64) {
        const size_t c1 = std::min(count, c0 + 64), n = c1 - c0, npairs = (n + 1) / 2;
        for (size_t i = 0; i < n; ++i) {
            rows[i] = kmajor + static_cast<size_t>(idx[c0 + i]) * rb;
            for (size_t off = 0; off < rb; off += 64) _mm_prefetch(reinterpret_cast<const char*>(rows[i] + off), _MM_HINT_T1);
        }
        if (n & 1) rows[n] = rows[n - 1];
        for (size_t p = 0; p < npairs; ++p) {
            const int a1 = q[idx[c0 + 2 * p]], a2 = (2 * p + 1 < n) ? q[idx[c0 + 2 * p + 1]] : 0;
            avs[p] = _mm256_set1_epi16(static_cast<int16_t>(((a2 & 0xFF) << 8) | (a1 & 0xFF)));
        }
        for (size_t b = 0; b < nb; ++b) {
            __m256i lo[4], hi[4];
            for (int j = 0; j < 4; ++j) lo[j] = hi[j] = _mm256_setzero_si256();
            for (size_t p = 0; p < npairs; ++p) {
                const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[2 * p] + 32 * b));
                const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[2 * p + 1] + 32 * b));
                const __m256i av = avs[p];
                __m256i x1 = _mm256_and_si256(v1, m3), x2 = _mm256_and_si256(v2, m3);
                lo[0] = _mm256_add_epi16(lo[0], _mm256_maddubs_epi16(_mm256_unpacklo_epi8(x1, x2), av));
                hi[0] = _mm256_add_epi16(hi[0], _mm256_maddubs_epi16(_mm256_unpackhi_epi8(x1, x2), av));
                x1 = _mm256_and_si256(_mm256_srli_epi16(v1, 2), m3); x2 = _mm256_and_si256(_mm256_srli_epi16(v2, 2), m3);
                lo[1] = _mm256_add_epi16(lo[1], _mm256_maddubs_epi16(_mm256_unpacklo_epi8(x1, x2), av));
                hi[1] = _mm256_add_epi16(hi[1], _mm256_maddubs_epi16(_mm256_unpackhi_epi8(x1, x2), av));
                x1 = _mm256_and_si256(_mm256_srli_epi16(v1, 4), m3); x2 = _mm256_and_si256(_mm256_srli_epi16(v2, 4), m3);
                lo[2] = _mm256_add_epi16(lo[2], _mm256_maddubs_epi16(_mm256_unpacklo_epi8(x1, x2), av));
                hi[2] = _mm256_add_epi16(hi[2], _mm256_maddubs_epi16(_mm256_unpackhi_epi8(x1, x2), av));
                x1 = _mm256_and_si256(_mm256_srli_epi16(v1, 6), m3); x2 = _mm256_and_si256(_mm256_srli_epi16(v2, 6), m3);
                lo[3] = _mm256_add_epi16(lo[3], _mm256_maddubs_epi16(_mm256_unpacklo_epi8(x1, x2), av));
                hi[3] = _mm256_add_epi16(hi[3], _mm256_maddubs_epi16(_mm256_unpackhi_epi8(x1, x2), av));
            }
            for (int j = 0; j < 4; ++j) {
                int32_t* o = acc + 128 * b + 32 * j;
                auto add = [&](int off, __m128i h16) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + off), _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(o + off)), _mm256_cvtepi16_epi32(h16)));
                };
                add(0, _mm256_castsi256_si128(lo[j]));        // outputs 0..7
                add(8, _mm256_castsi256_si128(hi[j]));        // outputs 8..15
                add(16, _mm256_extracti128_si256(lo[j], 1));  // outputs 16..23
                add(24, _mm256_extracti128_si256(hi[j], 1));  // outputs 24..31
            }
        }
    }
}

void gemv_sparse_accumulate(const uint8_t* kmajor, size_t N, const int8_t* q, const uint32_t* idx, size_t count, int32_t* acc, Isa isa) {
    if (isa == Isa::Scalar) sparse_accumulate_scalar(kmajor, N, q, idx, count, acc);
    else sparse_accumulate_avx2(kmajor, N, q, idx, count, acc);
}

void gemv_sparse(ThreadPool& pool, const uint8_t* kmajor, size_t N, size_t K, const int8_t* q, int32_t* y, SparseScratch& sc, Isa isa) {
    const size_t T = static_cast<size_t>(pool.size());
    if (sc.idx.size() < K) sc.idx.resize(K);
    if (sc.acc.size() < T) sc.acc.resize(T);
    for (size_t t = 0; t < T; ++t) {
        if (sc.acc[t].size() < N) sc.acc[t].reset(N);
        std::memset(sc.acc[t].data(), 0, N * sizeof(int32_t));
    }
    int32_t sum = 0;
    const size_t cnt = nonzero_indices(q, K, sc.idx.data(), &sum);
    sc.nnz = cnt;
    constexpr size_t kChunk = 48;
    pool.parallel_for((cnt + kChunk - 1) / kChunk, [&](size_t c, int tid) {
        const size_t i0 = c * kChunk;
        gemv_sparse_accumulate(kmajor, N, q, sc.idx.data() + i0, std::min(kChunk, cnt - i0), sc.acc[static_cast<size_t>(tid)].data(), isa);
    });
    constexpr size_t kRed = 256;
    pool.parallel_for((N + kRed - 1) / kRed, [&](size_t c, int) {
        const size_t n0 = c * kRed, n1 = std::min(N, n0 + kRed);
        for (size_t n = n0; n < n1; ++n) {
            int32_t s = -sum;
            for (size_t t = 0; t < T; ++t) s += sc.acc[t][n];
            y[n] = s;
        }
    });
}

} // namespace i2k
} // namespace ternix
