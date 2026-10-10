#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ternix {

// C[M,N] (int32) = A[M,K] (int8) x W[K,N] (ternary, packed 4 per byte as K x N/4).
// Device-pointer variants are asynchronous on `stream`; they return cudaGetLastError().

// Baseline: one thread per output element. Any M, K; N % 4 == 0.
cudaError_t mul_mat_ternary_cuda_naive(const int8_t* dA, const uint8_t* dB, int32_t* dC,
                                       int M, int K, int N, cudaStream_t stream = 0);

// Tiled kernel: 64x64x64 block tile, cp.async double buffering. Requires M, N, K multiples of 64
// (returns cudaErrorInvalidValue otherwise) and 16-byte aligned device pointers.
cudaError_t mul_mat_ternary_cuda_tiled(const int8_t* dA, const uint8_t* dB, int32_t* dC,
                                       int M, int K, int N, cudaStream_t stream = 0);

// Convenience wrapper on host pointers (allocates, copies, runs, copies back, frees).
cudaError_t mul_mat_ternary_cuda_host(const int8_t* A, const uint8_t* B_packed, int32_t* C,
                                      int M, int K, int N, bool tiled);

} // namespace ternix
