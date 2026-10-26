#include "ternix/ops.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ternix/cpu_features.hpp"

namespace ternix {
namespace ops {

void add_inplace(float* x, const float* y, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) x[i] += y[i];
}

void relu2_mul(float* out, const float* gate, const float* up, size_t n) {
    const __m256 z = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 g = _mm256_max_ps(_mm256_loadu_ps(gate + i), z);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(g, g), _mm256_loadu_ps(up + i)));
    }
    for (; i < n; ++i) {
        const float g = std::max(gate[i], 0.f);
        out[i] = g * g * up[i];
    }
}

void silu_mul(float* out, const float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = gate[i] / (1.f + std::exp(-gate[i])) * up[i];
}

static inline __m256 bf16x8_to_f32(const uint16_t* p) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

void dot_bf16_rows(const uint16_t* row, const float* vecs, size_t nvec, size_t n, float* out) {
    if (nvec == 1) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        size_t i = 0;
        for (; i + 16 <= n; i += 16) {
            a0 = _mm256_fmadd_ps(bf16x8_to_f32(row + i), _mm256_loadu_ps(vecs + i), a0);
            a1 = _mm256_fmadd_ps(bf16x8_to_f32(row + i + 8), _mm256_loadu_ps(vecs + i + 8), a1);
        }
        float s = hsum(_mm256_add_ps(a0, a1));
        for (; i < n; ++i) {
            uint32_t b = static_cast<uint32_t>(row[i]) << 16;
            float w;
            std::memcpy(&w, &b, 4);
            s += w * vecs[i];
        }
        out[0] = s;
        return;
    }
    for (size_t v = 0; v < nvec; ++v) out[v] = 0.f;
    // decode each 8-wide bf16 slice once, reuse it for every vector
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 w = bf16x8_to_f32(row + i);
        for (size_t v = 0; v < nvec; ++v) {
            __m256 p = _mm256_mul_ps(w, _mm256_loadu_ps(vecs + v * n + i));
            out[v] += hsum(p);
        }
    }
    for (; i < n; ++i) {
        uint32_t b = static_cast<uint32_t>(row[i]) << 16;
        float w;
        std::memcpy(&w, &b, 4);
        for (size_t v = 0; v < nvec; ++v) out[v] += w * vecs[v * n + i];
    }
}

float f16_to_f32_scalar(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void f32_to_f16(uint16_t* dst, const float* src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; ++i) {
        __m128i h = _mm_cvtps_ph(_mm_set_ss(src[i]), _MM_FROUND_TO_NEAREST_INT);
        dst[i] = static_cast<uint16_t>(_mm_extract_epi16(h, 0));
    }
}

void rope_apply(float* v, const float* c, const float* s, size_t d) {
    const size_t h = d / 2;
    size_t i = 0;
    for (; i + 8 <= h; i += 8) {
        const __m256 a = _mm256_loadu_ps(v + i), b = _mm256_loadu_ps(v + i + h);
        const __m256 cc = _mm256_loadu_ps(c + i), ss = _mm256_loadu_ps(s + i);
        _mm256_storeu_ps(v + i, _mm256_sub_ps(_mm256_mul_ps(a, cc), _mm256_mul_ps(b, ss)));
        _mm256_storeu_ps(v + i + h, _mm256_add_ps(_mm256_mul_ps(b, cc), _mm256_mul_ps(a, ss)));
    }
    for (; i < h; ++i) {
        const float a = v[i], b = v[i + h];
        v[i] = a * c[i] - b * s[i];
        v[i + h] = b * c[i] + a * s[i];
    }
}

void attend_head(const float* q, const uint16_t* k, const uint16_t* v, size_t len, size_t d, float scale, float* scores,
                 float* out) {
    float mx = -INFINITY;
    for (size_t j = 0; j < len; ++j) {
        const uint16_t* kj = k + j * d;
        __m256 acc = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= d; i += 8)
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kj + i))), _mm256_loadu_ps(q + i), acc);
        float s = hsum(acc);
        for (; i < d; ++i) s += f16_to_f32_scalar(kj[i]) * q[i];
        s *= scale;
        scores[j] = s;
        mx = std::max(mx, s);
    }
    float sum = 0.f;
    for (size_t j = 0; j < len; ++j) {
        scores[j] = std::exp(scores[j] - mx);
        sum += scores[j];
    }
    const float inv = 1.f / sum;
    // accumulate V rows; d is a multiple of 8 for every supported head size, tail handled scalar
    constexpr size_t kMaxVec = 32;   // up to head_dim 256
    __m256 acc[kMaxVec];
    const size_t nv = d / 8;
    for (size_t t = 0; t < nv; ++t) acc[t] = _mm256_setzero_ps();
    for (size_t j = 0; j < len; ++j) {
        const __m256 p = _mm256_set1_ps(scores[j] * inv);
        const uint16_t* vj = v + j * d;
        for (size_t t = 0; t < nv; ++t)
            acc[t] = _mm256_fmadd_ps(p, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vj + 8 * t))), acc[t]);
    }
    for (size_t t = 0; t < nv; ++t) _mm256_storeu_ps(out + 8 * t, acc[t]);
    for (size_t i = nv * 8; i < d; ++i) {
        float s = 0.f;
        for (size_t j = 0; j < len; ++j) s += scores[j] * inv * f16_to_f32_scalar(v[j * d + i]);
        out[i] = s;
    }
}

} // namespace ops
} // namespace ternix
