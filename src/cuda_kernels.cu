// CUDA ternary GEMM: C[M,N] (int32) = A[M,K] (int8) x W[K,N] (ternary, 2-bit packed K x N/4).
//
// Two kernels share one contract so they can be checked against the CPU reference:
//   * ternary_gemm_naive_kernel : one thread per output element, weights decoded from
//     global memory inside the K loop. Baseline.
//   * ternary_gemm_tiled_kernel : 64x64x64 block tile, 256 threads, each thread owns a
//     4x4 output micro-tile (one packed weight byte = the four columns it owns).
//     Tiles are streamed global->shared with cp.async (16 B per copy) through a
//     two-stage double buffer, so the copy of tile t+1 overlaps the math on tile t.
//     The int8 tile uses a padded row pitch (80 B) so the threads of a warp that read
//     different rows hit different shared-memory banks.
//
// Shape requirements of the tiled kernel (checked by the host wrapper):
//   M % 64 == 0, N % 64 == 0, K % 64 == 0 and 16-byte aligned device pointers
//   (cudaMalloc guarantees this). The naive kernel accepts any M, K with N % 4 == 0.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "ternix/cuda_kernels.cuh"

namespace ternix {
namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int A_PITCH = BK + 16;          // bytes per int8 row in shared memory (16 B aligned)
constexpr int B_COLS = BN / 4;            // packed bytes per k-row of the weight tile
constexpr int THREADS = 256;              // 16 x 16 threads, 4 x 4 outputs each

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// 00 -> 0, 01 -> +1, 11 -> -1 (branch-free)
__device__ __forceinline__ int decode2(unsigned bits) { return static_cast<int>(bits & 1u) - static_cast<int>(bits & 2u); }

__global__ void ternary_gemm_naive_kernel(const int8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
                                          int32_t* __restrict__ C, int M, int K, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    int32_t acc = 0;
    const int shift = (col & 3) * 2;
    for (int k = 0; k < K; ++k) {
        const unsigned byte = Bp[static_cast<size_t>(k) * (N / 4) + (col >> 2)];
        acc += static_cast<int>(A[static_cast<size_t>(row) * K + k]) * decode2((byte >> shift) & 3u);
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

__device__ __forceinline__ void load_tile_async(int8_t (*sA)[A_PITCH], uint8_t (*sB)[B_COLS],
                                                const int8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
                                                int K, int N, int row0, int col0_bytes, int k0, int tid) {
    // A tile: 64 rows x 64 B = 256 chunks of 16 B -> one per thread.
    {
        const int r = tid >> 2;            // 0..63
        const int c16 = (tid & 3) * 16;    // 0,16,32,48
        cp_async_16(&sA[r][c16], A + static_cast<size_t>(row0 + r) * K + k0 + c16);
    }
    // W tile: 64 k-rows x 16 B = 64 chunks of 16 B -> first 64 threads.
    if (tid < BK) {
        cp_async_16(&sB[tid][0], Bp + static_cast<size_t>(k0 + tid) * (N / 4) + col0_bytes);
    }
}

__global__ void __launch_bounds__(THREADS)
ternary_gemm_tiled_kernel(const int8_t* __restrict__ A, const uint8_t* __restrict__ Bp,
                          int32_t* __restrict__ C, int M, int K, int N) {
    __shared__ __align__(16) int8_t sA[2][BM][A_PITCH];
    __shared__ __align__(16) uint8_t sB[2][BK][B_COLS];

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int col0_bytes = col0 / 4;
    const int ktiles = K / BK;

    int32_t acc[4][4] = {};

    load_tile_async(sA[0], sB[0], A, Bp, K, N, row0, col0_bytes, 0, tid);
    cp_async_commit();

    for (int t = 0; t < ktiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < ktiles) {
            load_tile_async(sA[cur ^ 1], sB[cur ^ 1], A, Bp, K, N, row0, col0_bytes, (t + 1) * BK, tid);
            cp_async_commit();
            cp_async_wait<1>();            // tile t has landed; tile t+1 may still be in flight
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

#pragma unroll 8
        for (int kk = 0; kk < BK; ++kk) {
            const unsigned wb = sB[cur][kk][tx];   // four ternary weights for this thread's columns
            const int w0 = decode2(wb & 3u), w1 = decode2((wb >> 2) & 3u);
            const int w2 = decode2((wb >> 4) & 3u), w3 = decode2((wb >> 6) & 3u);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int a = sA[cur][ty * 4 + r][kk];
                acc[r][0] += a * w0;
                acc[r][1] += a * w1;
                acc[r][2] += a * w2;
                acc[r][3] += a * w3;
            }
        }
        __syncthreads();                   // buffer `cur` is refilled by the next iteration's prefetch
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        int32_t* out = C + static_cast<size_t>(row0 + ty * 4 + r) * N + col0 + tx * 4;
        out[0] = acc[r][0]; out[1] = acc[r][1]; out[2] = acc[r][2]; out[3] = acc[r][3];
    }
}

} // namespace

cudaError_t mul_mat_ternary_cuda_naive(const int8_t* dA, const uint8_t* dB, int32_t* dC, int M, int K, int N, cudaStream_t s) {
    if (N % 4) return cudaErrorInvalidValue;
    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    ternary_gemm_naive_kernel<<<grid, block, 0, s>>>(dA, dB, dC, M, K, N);
    return cudaGetLastError();
}

cudaError_t mul_mat_ternary_cuda_tiled(const int8_t* dA, const uint8_t* dB, int32_t* dC, int M, int K, int N, cudaStream_t s) {
    if (M % BM || N % BN || K % BK) return cudaErrorInvalidValue;
    const dim3 block(16, 16);
    const dim3 grid(N / BN, M / BM);
    ternary_gemm_tiled_kernel<<<grid, block, 0, s>>>(dA, dB, dC, M, K, N);
    return cudaGetLastError();
}

cudaError_t mul_mat_ternary_cuda_host(const int8_t* A, const uint8_t* Bp, int32_t* C, int M, int K, int N, bool tiled) {
    int8_t* dA = nullptr; uint8_t* dB = nullptr; int32_t* dC = nullptr;
    const size_t a_bytes = static_cast<size_t>(M) * K, b_bytes = static_cast<size_t>(K) * N / 4,
                 c_bytes = static_cast<size_t>(M) * N * sizeof(int32_t);
    cudaError_t e = cudaSuccess;
    if ((e = cudaMalloc(&dA, a_bytes)) || (e = cudaMalloc(&dB, b_bytes)) || (e = cudaMalloc(&dC, c_bytes))) goto done;
    if ((e = cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice)) || (e = cudaMemcpy(dB, Bp, b_bytes, cudaMemcpyHostToDevice))) goto done;
    e = tiled ? mul_mat_ternary_cuda_tiled(dA, dB, dC, M, K, N, 0) : mul_mat_ternary_cuda_naive(dA, dB, dC, M, K, N, 0);
    if (e) goto done;
    if ((e = cudaDeviceSynchronize())) goto done;
    e = cudaMemcpy(C, dC, c_bytes, cudaMemcpyDeviceToHost);
done:
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return e;
}

} // namespace ternix
