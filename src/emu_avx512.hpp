#pragma once
// Scalar emulation of the few AVX-512 intrinsics used by src/i2k_avx512.cpp. Included only when TERNIX_AVX512_EMULATE is defined (tests/i2k_avx512_emulated.cpp),
// so the kernel's indexing, block pairing, tail handling and reduction can be checked on CPUs without AVX-512. It does not test the hardware behaviour of the real
// instructions; the semantics below follow the Intel intrinsics guide (vpdpbusd: per 32-bit lane acc += sum of four u8 * s8 products, no saturation).
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace emu512 {

struct M512 { uint8_t b[64]; };

inline M512 zero() { M512 r; std::memset(r.b, 0, 64); return r; }
inline M512 loadu(const void* p) { M512 r; std::memcpy(r.b, p, 64); return r; }
inline M512 set1_epi8(int v) { M512 r; std::memset(r.b, v, 64); return r; }
inline M512 and_(M512 a, M512 b) { for (int i = 0; i < 64; ++i) a.b[i] &= b.b[i]; return a; }
inline M512 srli_epi16(M512 a, int n) {
    for (int i = 0; i < 32; ++i) {
        uint16_t v;
        std::memcpy(&v, a.b + 2 * i, 2);
        v = static_cast<uint16_t>(v >> n);
        std::memcpy(a.b + 2 * i, &v, 2);
    }
    return a;
}
inline M512 cast256(__m256i lo) { M512 r = zero(); std::memcpy(r.b, &lo, 32); return r; }
inline M512 insert256(M512 a, __m256i hi, int imm) { std::memcpy(a.b + 32 * (imm & 1), &hi, 32); return a; }
inline M512 dpbusd(M512 acc, M512 u, M512 s) {
    for (int i = 0; i < 16; ++i) {
        int32_t a;
        std::memcpy(&a, acc.b + 4 * i, 4);
        for (int j = 0; j < 4; ++j) a += static_cast<int32_t>(u.b[4 * i + j]) * static_cast<int32_t>(static_cast<int8_t>(s.b[4 * i + j]));
        std::memcpy(acc.b + 4 * i, &a, 4);
    }
    return acc;
}
inline int32_t reduce_add(M512 a) {
    int32_t s = 0;
    for (int i = 0; i < 16; ++i) { int32_t v; std::memcpy(&v, a.b + 4 * i, 4); s += v; }
    return s;
}
inline __m256i dpbusd256(__m256i acc, __m256i u, __m256i s) {
    alignas(32) uint8_t ub[32], sb[32];
    alignas(32) int32_t ab[8];
    std::memcpy(ub, &u, 32); std::memcpy(sb, &s, 32); std::memcpy(ab, &acc, 32);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) ab[i] += static_cast<int32_t>(ub[4 * i + j]) * static_cast<int32_t>(static_cast<int8_t>(sb[4 * i + j]));
    __m256i r;
    std::memcpy(&r, ab, 32);
    return r;
}

} // namespace emu512

#undef __m512i
#undef _mm512_setzero_si512
#undef _mm512_loadu_si512
#undef _mm512_set1_epi8
#undef _mm512_and_si512
#undef _mm512_srli_epi16
#undef _mm512_castsi256_si512
#undef _mm512_inserti64x4
#undef _mm512_dpbusd_epi32
#undef _mm512_reduce_add_epi32
#undef _mm256_dpbusd_epi32
#define __m512i emu512::M512
#define _mm512_setzero_si512 emu512::zero
#define _mm512_loadu_si512 emu512::loadu
#define _mm512_set1_epi8 emu512::set1_epi8
#define _mm512_and_si512 emu512::and_
#define _mm512_srli_epi16 emu512::srli_epi16
#define _mm512_castsi256_si512 emu512::cast256
#define _mm512_inserti64x4 emu512::insert256
#define _mm512_dpbusd_epi32 emu512::dpbusd
#define _mm512_reduce_add_epi32 emu512::reduce_add
#define _mm256_dpbusd_epi32 emu512::dpbusd256
