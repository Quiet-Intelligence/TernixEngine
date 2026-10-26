#pragma once
// Element-wise and attention operators for the CPU engine (float32 activations, fp16 KV cache, bf16 embedding rows).
#include <cstddef>
#include <cstdint>

namespace ternix {
namespace ops {

// x[i] += y[i]
void add_inplace(float* x, const float* y, size_t n);
// out[i] = max(gate[i], 0)^2 * up[i]   (squared-ReLU gated feed-forward)
void relu2_mul(float* out, const float* gate, const float* up, size_t n);
// out[i] = silu(gate[i]) * up[i]
void silu_mul(float* out, const float* gate, const float* up, size_t n);

// Dot products of one bf16 row with `nvec` float vectors of length n (vectors contiguous, stride n): out[v].
void dot_bf16_rows(const uint16_t* row, const float* vecs, size_t nvec, size_t n, float* out);

// fp16 helpers (F16C when present, scalar fallback otherwise)
void f32_to_f16(uint16_t* dst, const float* src, size_t n);
float f16_to_f32_scalar(uint16_t h);

// Rotary embedding, rotate-half convention (pairs (i, i + d/2)); cos/sin are the tables for this position, length d/2.
void rope_apply(float* vec, const float* cos_t, const float* sin_t, size_t head_dim);

// Single-query attention over a contiguous fp16 K/V history of `len` positions (row stride head_dim).
//   scores: scratch of at least `len` floats. out: head_dim floats.
void attend_head(const float* q, const uint16_t* k, const uint16_t* v, size_t len, size_t head_dim, float scale,
                 float* scores, float* out);

// int8 K/V variant: each cached vector has one float scale (per token and kv head).
//   k_scale[j], v_scale[j]: dequantisation steps of position j.
void quantize_kv(int8_t* dst, float* scale, const float* src, size_t n);
void attend_head_i8(const float* q, const int8_t* k, const float* k_scale, const int8_t* v, const float* v_scale, size_t len, size_t head_dim,
                    float scale, float* scores, float* out);

// Decode attention over one kv head and a position chunk [j0, j1), shared by the gs query heads of its group: every K and V row is
// converted once and used for all gs heads. out[h * (d + 2) + {0: chunk max, 1: chunk sum of exp, 2..: sum of exp * V}].
// q: gs heads, row stride d. K/V are fp16 (`_f16`) or int8 with per-position scales (`_i8`). gs <= 8, j1 - j0 <= 256.
constexpr size_t kAttChunk = 256;
void attend_chunk_f16(const float* q, size_t gs, const uint16_t* k, const uint16_t* v, size_t j0, size_t j1, size_t d, float scale, float* out);
void attend_chunk_i8(const float* q, size_t gs, const int8_t* k, const float* k_scale, const int8_t* v, const float* v_scale, size_t j0, size_t j1, size_t d,
                     float scale, float* out);
// Merges `chunks` partial results of one head (stride d + 2 floats each) into out[d].
void attend_combine(const float* partial, size_t chunks, size_t d, float* out);

} // namespace ops
} // namespace ternix
