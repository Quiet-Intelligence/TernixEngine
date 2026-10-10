#pragma once
#include <cstdint>
#include <cstddef>

namespace ternix {
    // RMSNorm: out[i] = in[i] / sqrt(mean(in^2) + eps) * weight[i].
    // AVX2 + FMA: 8-lane sum-of-squares reduction and 8-lane scale pass, scalar tail.
    void rms_norm_avx2(float* out, const float* in, const float* weight, size_t size, float eps);
    // Scalar reference for rms_norm_avx2 (used by tests).
    void rms_norm_scalar(float* out, const float* in, const float* weight, size_t size, float eps);
    // SwiGLU over interleaved (gate, up) pairs: out[i] = silu(in[2i]) * in[2i+1]. Scalar.
    void swiglu(float* out, const float* in, size_t size);
}
