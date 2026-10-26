// AVX-512 VNNI kernel for the I2K128 layout. Compiled as its own translation unit with AVX-512 flags (CMakeLists.txt) so that no
// AVX-512 instruction can leak into code that runs on CPUs without it. Selected only when TERNIX_ISA=avx512_vnni and cpuid reports
// AVX512F/BW/VL/VNNI with OS-enabled ZMM state.
//
// Status: the algorithm (indexing, block pairing, tail, reduction) is tested against the scalar reference on every machine by compiling this file against a scalar
// emulation of the intrinsics (tests/i2k_avx512_emulated.cpp, EmulatedAvx512KernelMatchesScalar). The instructions themselves have NOT been executed or timed: the
// development machine (i7-14650HX) has no AVX-512. On AVX-512 hardware the generic ISA tests in tests/test_i2k.cpp run this kernel too (TERNIX_ISA is not needed there).
//
// Layout recap: block b is 32 bytes, byte l holds codes of columns 128b + 32j + l in bit pair j. A 64-byte load covers two blocks
// (b, b+1); bit pair j of that register pairs with the activation vector [q[128b + 32j, +32) | q[128(b+1) + 32j, +32)], which costs one
// 256->512 insert per (activation row, j) and is shared by all weight rows of the tile.
#include <immintrin.h>
#include "ternix/i2k.hpp"
#ifdef TERNIX_AVX512_EMULATE   // tests/i2k_avx512_emulated.cpp: same source on scalar-emulated intrinsics, under another symbol name
#include "emu_avx512.hpp"
#define gemm_rows_avx512 gemm_rows_avx512_emulated
#endif

namespace ternix {
namespace i2k {
namespace {

inline __m512i act_pair(const int8_t* a, int j) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * j));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 128 + 32 * j));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

template <int MR, int NR>
void tile512(const uint8_t* W, const int8_t* q, const int32_t* sum_q, int32_t* y, size_t y_stride, size_t n, size_t n0, size_t K) {
    __m512i acc[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) acc[m][r] = _mm512_setzero_si512();
    const size_t rb = K / 4, nb = K / kBlock, npair = nb / 2;
    const uint8_t* wr[NR];
    for (int r = 0; r < NR; ++r) wr[r] = W + (n + r) * rb;
    const __m512i m3 = _mm512_set1_epi8(3);
    for (size_t p = 0; p < npair; ++p) {
        __m512i w[NR];
        for (int r = 0; r < NR; ++r) w[r] = _mm512_loadu_si512(wr[r] + p * 64);
        for (int j = 0; j < 4; ++j) {
            __m512i wj[NR];
            for (int r = 0; r < NR; ++r) wj[r] = _mm512_and_si512(_mm512_srli_epi16(w[r], 2 * j), m3);
            for (int m = 0; m < MR; ++m) {
                const __m512i a = act_pair(q + static_cast<size_t>(m) * K + p * 256, j);
                for (int r = 0; r < NR; ++r) acc[m][r] = _mm512_dpbusd_epi32(acc[m][r], wj[r], a);
            }
        }
    }
    int32_t s[MR][NR];
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) s[m][r] = _mm512_reduce_add_epi32(acc[m][r]);
    if (nb & 1) {   // one trailing 128-column block: 256-bit VNNI (AVX512VL form)
        const size_t b = nb - 1;
        const __m256i m3y = _mm256_set1_epi8(3);
        for (int r = 0; r < NR; ++r) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr[r] + b * 32));
            for (int m = 0; m < MR; ++m) {
                const int8_t* a = q + static_cast<size_t>(m) * K + b * kBlock;
                __m256i t = _mm256_setzero_si256();
                t = _mm256_dpbusd_epi32(t, _mm256_and_si256(w, m3y), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                t = _mm256_dpbusd_epi32(t, _mm256_and_si256(_mm256_srli_epi16(w, 2), m3y), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32)));
                t = _mm256_dpbusd_epi32(t, _mm256_and_si256(_mm256_srli_epi16(w, 4), m3y), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 64)));
                t = _mm256_dpbusd_epi32(t, _mm256_and_si256(_mm256_srli_epi16(w, 6), m3y), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 96)));
                __m128i h = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
                h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4E));
                h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xB1));
                s[m][r] += _mm_cvtsi128_si32(h);
            }
        }
    }
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NR; ++r) y[m * y_stride + (n - n0) + r] = s[m][r] - sum_q[m];
}

template <int MR>
void rows512(const uint8_t* W, const int8_t* q, const int32_t* sum_q, int32_t* y, size_t y_stride, size_t n0, size_t n1, size_t K) {
    constexpr int NR = (MR == 1) ? 4 : 2;
    size_t n = n0;
    for (; n + NR <= n1; n += NR) tile512<MR, NR>(W, q, sum_q, y, y_stride, n, n0, K);
    for (; n < n1; ++n) tile512<MR, 1>(W, q, sum_q, y, y_stride, n, n0, K);
}

} // namespace

void gemm_rows_avx512(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride, size_t n0, size_t n1, size_t K) {
    switch (MR) {
        case 1: rows512<1>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        case 2: rows512<2>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        case 3: rows512<3>(W, q, sum_q, y, y_stride, n0, n1, K); break;
        default: rows512<4>(W, q, sum_q, y, y_stride, n0, n1, K); break;
    }
}

} // namespace i2k
} // namespace ternix
