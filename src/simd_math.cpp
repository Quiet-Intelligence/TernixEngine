#include "ternix/simd_math.hpp"
#include <immintrin.h>
#include <cmath>

namespace ternix {

void rms_norm_scalar(float* out, const float* in, const float* weight, size_t size, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) sum_sq += in[i] * in[i];
    const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    for (size_t i = 0; i < size; ++i) out[i] = in[i] * inv * weight[i];
}

void rms_norm_avx2(float* out, const float* in, const float* weight, size_t size, float eps) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    // horizontal sum of the 8 lanes
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    float sum_sq = _mm_cvtss_f32(s);
    for (size_t t = i; t < size; ++t) sum_sq += in[t] * in[t];

    const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(size) + eps);
    const __m256 vinv = _mm256_set1_ps(inv);
    size_t j = 0;
    for (; j + 8 <= size; j += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(in + j), vinv);
        _mm256_storeu_ps(out + j, _mm256_mul_ps(v, _mm256_loadu_ps(weight + j)));
    }
    for (; j < size; ++j) out[j] = in[j] * inv * weight[j];
}

void swiglu(float* out, const float* in, size_t size) {
    for (size_t i = 0; i + 1 < size; i += 2) {
        const float x = in[i];
        const float y = in[i + 1];
        out[i / 2] = (x / (1.0f + std::exp(-x))) * y;
    }
}

} // namespace ternix
