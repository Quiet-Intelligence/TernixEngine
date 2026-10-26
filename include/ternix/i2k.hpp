#pragma once
// Ternary x int8 kernels on the I2K128 weight layout (see docs/FORMAT.md).
//
//   W[n][k], n = output row, k = input column, K % 128 == 0. Row n is K/4 bytes: per 128-column block 32 bytes, byte l
//   holding the 2-bit codes (w + 1) of columns 128b + 32j + l in bit pair j. One 32-byte load therefore yields four
//   32-lane weight vectors whose activations are four contiguous 32-byte loads.
//
// Dot product trick: with codes c = w + 1 in {0,1,2}, sum_k q_k w_k = sum_k q_k c_k - sum_k q_k, so the multiply is an
// unsigned x signed byte dot product (vpdpbusd / vpmaddubsw) and the correction term is one scalar per activation row.
#include <cstddef>
#include <cstdint>
#include "ternix/cpu_features.hpp"

namespace ternix {
namespace i2k {

constexpr size_t kBlock = 128;                       // columns per 32-byte block
inline size_t row_bytes(size_t K) { return K / 4; }

// Per-token absmax int8 quantisation exactly as BitLinear does it:
//   s = 127 / max(amax, 1e-5);  q = clamp(rint(x * s), -128, 127).
// Returns the dequantisation step amax'/127 in *deq_step and the sum of q (needed by the dot-product correction).
void quantize_act(const float* x, int8_t* q, size_t K, float* deq_step, int32_t* sum_q, Isa isa);

// y[i] = sum_k q[k] * W[n0 + i][k] for i in [0, n1 - n0): exact int32.
void gemv(const uint8_t* W, const int8_t* q, int32_t sum_q, int32_t* y, size_t n0, size_t n1, size_t K, Isa isa);

// Same for MR activation rows (1 <= MR <= 4, rows contiguous with stride K): y[m * y_stride + i].
void gemm_rows(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride,
               size_t n0, size_t n1, size_t K, Isa isa);

// Reference decode of one row (tests, tooling): out[k] in {-1, 0, +1}.
void unpack_row(const uint8_t* row, int8_t* out, size_t K);
// Packs int8 ternary W[N][K] into the layout (tests, tooling).
void pack(const int8_t* w, uint8_t* out, size_t N, size_t K);

} // namespace i2k
} // namespace ternix
