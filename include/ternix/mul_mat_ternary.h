#pragma once
#include <cstddef>
#include <cstdint>

namespace ternix {

/**
 * @brief Naive, unoptimized implementation of ternary matrix multiplication
 * 
 * Computes C = A * B, where:
 * @param A        M x K matrix of INT8 activations
 * @param B_packed K x N matrix of 1.58-bit (packed to 2-bit) weights
 * @param C        M x N matrix of INT32 accumulated results
 * @param M        Number of rows in A and C
 * @param K        Number of columns in A, rows in B
 * @param N        Number of columns in B and C
 */
void mul_mat_ternary_naive(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

/**
 * @brief Strong scalar baseline: i-k-j loop order and a branch-free 2-bit decode.
 * Used to measure the SIMD kernels against something other than the cache-hostile
 * naive loop order.
 */
void mul_mat_ternary_scalar(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

/**
 * @brief AVX2 SIMD-native implementation of ternary matrix multiplication
 * 
 * Bypasses floating point math. Uses in-register unpacking and accumulation.
 * Precondition: N % 8 == 0 (eight output columns per 256-bit register);
 * B_packed is the row-major K x (N/4) byte layout produced by pack_ternary_weights.
 */
void mul_mat_ternary_avx2(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

/**
 * @brief AVX2 kernel with 4-row register blocking: each decoded 8-weight vector
 * is applied to four activation rows before moving on (same result as above).
 */
void mul_mat_ternary_avx2_blocked(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N
);

} // namespace ternix
