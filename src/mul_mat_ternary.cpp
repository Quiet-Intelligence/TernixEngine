#include "ternix/mul_mat_ternary.h"
#include <memory>
#include <immintrin.h>
#include <cstring>

namespace ternix {

void mul_mat_ternary_naive(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N) 
{
    const int8_t* aligned_A = std::assume_aligned<32>(A);
    const uint8_t* aligned_B = std::assume_aligned<32>(B_packed);
    int32_t* aligned_C = std::assume_aligned<32>(C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // B is packed along N: shape K x (N/4) bytes
                size_t byte_idx = k * (N / 4) + (j / 4);
                uint8_t packed_val = aligned_B[byte_idx];
                uint8_t weight_bits = (packed_val >> ((j % 4) * 2)) & 0x03;
                
                int8_t weight = 0;
                if (weight_bits == 1) weight = 1;
                else if (weight_bits == 3) weight = -1;
                
                sum += aligned_A[i * K + k] * weight;
            }
            aligned_C[i * N + j] = sum;
        }
    }
}

void mul_mat_ternary_scalar(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N)
{
    // Strong scalar baseline: i-k-j loop order (streams B and C rows contiguously),
    // branch-free decode w = (bits & 1) - (bits & 2): 00->0, 01->+1, 11->-1.
    for (size_t i = 0; i < M; ++i) {
        int32_t* c = C + i * N;
        for (size_t j = 0; j < N; ++j) c[j] = 0;
        for (size_t k = 0; k < K; ++k) {
            const int32_t a = A[i * K + k];
            const uint8_t* b = B_packed + k * (N / 4);
            for (size_t j = 0; j < N; ++j) {
                const int bits = (b[j >> 2] >> ((j & 3) * 2)) & 3;
                c[j] += a * ((bits & 1) - (bits & 2));
            }
        }
    }
}

void mul_mat_ternary_avx2(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N)
{
    const int8_t* aligned_A = std::assume_aligned<32>(A);
    const uint8_t* aligned_B = std::assume_aligned<32>(B_packed);
    int32_t* aligned_C = std::assume_aligned<32>(C);

    // AVX2 registers hold 8 32-bit integers. We tile N by 8.
    const __m256i shift_vec = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i mask_3 = _mm256_set1_epi32(3);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += 8) {
            __m256i c_vec = _mm256_setzero_si256();
            
            for (size_t k = 0; k < K; ++k) {
                // Scalar broadcast of activation A[i, k]
                int8_t a_val = aligned_A[i * K + k];
                __m256i a_vec = _mm256_set1_epi32(a_val);
                
                // Load 2 bytes (8 weights) from B_packed
                // B_packed layout is K x (N/4).
                uint16_t packed_w;
                std::memcpy(&packed_w, &aligned_B[k * (N / 4) + (j / 4)], sizeof(packed_w));
                __m256i w_packed = _mm256_set1_epi32(packed_w);
                
                // SIMD-native extraction of 8 2-bit weights
                __m256i w_shifted = _mm256_srlv_epi32(w_packed, shift_vec);
                __m256i w_2bit = _mm256_and_si256(w_shifted, mask_3);
                
                // Branchless masking: Map (0, 1, 3) to (0, 1, -1)
                __m256i sign_control = _mm256_srai_epi32(_mm256_slli_epi32(w_2bit, 30), 30);
                
                // Accumulation bypassing float
                __m256i product = _mm256_sign_epi32(a_vec, sign_control);
                c_vec = _mm256_add_epi32(c_vec, product);
            }
            
            // Store accumulated results back to C
            _mm256_storeu_si256((__m256i*)&aligned_C[i * N + j], c_vec);
        }
    }
}

void mul_mat_ternary_avx2_blocked(
    const int8_t* __restrict A,
    const uint8_t* __restrict B_packed,
    int32_t* __restrict C,
    size_t M, size_t K, size_t N)
{
    // Register-blocked variant: the 2-bit -> {0,+1,-1} decode of eight weights is
    // done once per (k, j) and reused for MR activation rows, amortising the
    // shift/mask/sign-extend sequence that dominates the single-row kernel.
    constexpr size_t MR = 4;
    const __m256i shift_vec = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    const __m256i mask_3 = _mm256_set1_epi32(3);

    size_t i = 0;
    for (; i + MR <= M; i += MR) {
        for (size_t j = 0; j < N; j += 8) {
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();
            const int8_t* a0 = A + (i + 0) * K;
            const int8_t* a1 = A + (i + 1) * K;
            const int8_t* a2 = A + (i + 2) * K;
            const int8_t* a3 = A + (i + 3) * K;
            const uint8_t* b = B_packed + (j / 4);
            for (size_t k = 0; k < K; ++k) {
                uint16_t packed_w;
                std::memcpy(&packed_w, b + k * (N / 4), sizeof(packed_w));
                __m256i w = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set1_epi32(packed_w), shift_vec), mask_3);
                w = _mm256_srai_epi32(_mm256_slli_epi32(w, 30), 30);
                c0 = _mm256_add_epi32(c0, _mm256_sign_epi32(_mm256_set1_epi32(a0[k]), w));
                c1 = _mm256_add_epi32(c1, _mm256_sign_epi32(_mm256_set1_epi32(a1[k]), w));
                c2 = _mm256_add_epi32(c2, _mm256_sign_epi32(_mm256_set1_epi32(a2[k]), w));
                c3 = _mm256_add_epi32(c3, _mm256_sign_epi32(_mm256_set1_epi32(a3[k]), w));
            }
            _mm256_storeu_si256((__m256i*)&C[(i + 0) * N + j], c0);
            _mm256_storeu_si256((__m256i*)&C[(i + 1) * N + j], c1);
            _mm256_storeu_si256((__m256i*)&C[(i + 2) * N + j], c2);
            _mm256_storeu_si256((__m256i*)&C[(i + 3) * N + j], c3);
        }
    }
    if (i < M) {  // remaining rows
        mul_mat_ternary_avx2(A + i * K, B_packed, C + i * N, M - i, K, N);
    }
}

} // namespace ternix
