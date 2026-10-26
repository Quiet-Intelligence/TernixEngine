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
#include <vector>
#include "ternix/cpu_features.hpp"
#include "ternix/threadpool.hpp"
#include "ternix/util.hpp"

namespace ternix {
namespace i2k {

constexpr size_t kBlock = 128;                       // columns per 32-byte block
inline size_t row_bytes(size_t K) { return K / 4; }

// Per-token absmax int8 quantisation exactly as BitLinear does it:
//   s = 127 / max(amax, 1e-5);  q = clamp(rint(x * s), -128, 127).
// Returns the dequantisation step amax'/127 in *deq_step and the sum of q (needed by the dot-product correction).
void quantize_act(const float* x, int8_t* q, size_t K, float* deq_step, int32_t* sum_q, Isa isa);

// Per-128-column block activation quantisation (for group-scaled weights): block b of the row gets its own step
// deq[b] = max(amax_b, 1e-5) / 127 and sum[b] = sum of its int8 values.
void quantize_act_blocks(const float* x, int8_t* q, size_t K, float* deq, int32_t* sum, Isa isa);

// Group-scaled ternary GEMM (one fp16 scale per 128 input columns and output row, gs[n][K/128]) on block-quantised activations:
//   y[m][i] = sum_b gs[n][b] * deq[m][b] * (sum_{k in b} q[m][k] * W[n][k]) , computed as sum_b c_b * (dot_b - sum_q_b) in float.
// q rows have stride K; deq / sum_q have stride K/128. Output float, y[m * y_stride + (n - n0)].
void gemm_rows_gs(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sum_q, size_t MR, float* y, size_t y_stride,
                  size_t n0, size_t n1, size_t K, Isa isa);

// y[i] = sum_k q[k] * W[n0 + i][k] for i in [0, n1 - n0): exact int32.
void gemv(const uint8_t* W, const int8_t* q, int32_t sum_q, int32_t* y, size_t n0, size_t n1, size_t K, Isa isa);

// Same for MR activation rows (1 <= MR <= 4, rows contiguous with stride K): y[m * y_stride + i].
void gemm_rows(const uint8_t* W, const int8_t* q, const int32_t* sum_q, size_t MR, int32_t* y, size_t y_stride,
               size_t n0, size_t n1, size_t K, Isa isa);

// --- exact activation-sparse GEMV (batch 1) -----------------------------------------------------------------------------
// K-major copy of a ternary matrix: row k holds the N output codes for input column k, four per byte (output n in bit pair
// n % 4 of byte n / 4), code = w + 1. Skipping an input column whose int8 activation is exactly 0 changes nothing, so
//   y[n] = sum_{k: q_k != 0} q_k * W[n][k]
// is bit-identical to the dense kernel while reading only the rows of nonzero activations (80% of the down-projection input of
// BitNet 2B4T is exactly zero, see docs/research/EXPERIMENTS.md EXP-002).
// Builds the K-major copy from the I2K128 matrix (N rows, K columns); out has K * N/4 bytes. N % 128 == 0.
void pack_kmajor_from_i2k(const uint8_t* i2k_rows, uint8_t* out, size_t N, size_t K);
// Collects the nonzero positions of q (length K) in ascending order; returns their count; sum_q receives their sum.
size_t nonzero_indices(const int8_t* q, size_t K, uint32_t* idx, int32_t* sum_q);
// Adds sum_k q_k * code[k][n] over the listed columns into acc (N int32, caller zero-initialises); the caller subtracts sum_q.
void gemv_sparse_accumulate(const uint8_t* kmajor, size_t N, const int8_t* q, const uint32_t* idx, size_t count, int32_t* acc, Isa isa);

// Parallel batch-1 driver: nonzero columns are split into chunks claimed dynamically; every thread accumulates into a private
// int32 vector, then the vectors are summed and the sum_q correction subtracted: y[n] = exact dense GEMV result.
struct SparseScratch {
    std::vector<uint32_t> idx;
    std::vector<AlignedBuf<int32_t>> acc;
    size_t nnz = 0;   // nonzero count of the last call (for statistics)
};
void gemv_sparse(ThreadPool& pool, const uint8_t* kmajor, size_t N, size_t K, const int8_t* q, int32_t* y, SparseScratch& scratch, Isa isa);

// Reference decode of one row (tests, tooling): out[k] in {-1, 0, +1}.
void unpack_row(const uint8_t* row, int8_t* out, size_t K);
// Packs int8 ternary W[N][K] into the layout (tests, tooling).
void pack(const int8_t* w, uint8_t* out, size_t N, size_t K);

} // namespace i2k
} // namespace ternix
