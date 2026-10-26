#pragma once
// GPU forward pass for BitNet-style models: same arithmetic as the CPU engine (int8 activations x ternary weights with
// int32 accumulation, fp16 KV cache, bf16 tied head), executed entirely on the device.
#ifdef TERNIX_HAS_CUDA
#include <cstdint>
#include <memory>
#include "ternix/format.hpp"

namespace ternix {

class CudaModel {
public:
    CudaModel(const ModelFile& file, int max_ctx = 4096);
    ~CudaModel();
    CudaModel(const CudaModel&) = delete;
    CudaModel& operator=(const CudaModel&) = delete;

    const ModelConfig& config() const;
    int position() const;
    int max_ctx() const;
    void reset();
    void truncate(int pos);
    // Same contract as Model::forward. logits must hold vocab floats (mode 1) or n * vocab floats (mode 2).
    bool forward(const int32_t* tokens, int n, float* logits, int logits_mode = 1);
    // Teacher-forced scoring: nll[i] = -log softmax(logits after tokens[i])[targets[i]] (targets[i] < 0 skips). The logits stay on
    // the device, so perplexity runs do not pay for a 128k-wide host transfer and softmax per token.
    bool forward_nll(const int32_t* tokens, int n, const int32_t* targets, float* nll, int32_t* argmax = nullptr);
    // Greedy decode fast path: feed one token, return argmax of the next-token logits without copying them to the host.
    int32_t step_argmax(int32_t token);
    // Certified-exact int8 output head for batch-1 decode (same scheme as the CPU OutputHead); builds the int8 copy on first use.
    void set_head_int8(bool on);
    // CUDA-graph replay of the n == 1 decode step (default on).
    void set_graph(bool on);
    // int8 tensor-core GEMM (mma.sync m16n8k32) for chunks of more than 8 tokens (default on); off = dp4a kernels only.
    void set_tensor_cores(bool on);
    static constexpr int kMaxChunk = 64;
    size_t device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace ternix
#endif

#ifdef TERNIX_HAS_CUDA
namespace ternix {
// Test hooks: run single engine kernels on host data and return device results (see tests/test_cuda_engine.cu).
// y[t][n] = (sum_k q[t][k] W[n][k] - sum_q[t]) * deq[t] * scale for nt tokens (1..8), W in I2K128.
void cuda_test_ternary_gemv(const uint8_t* W, const int8_t* q, const float* deq, const int32_t* sumq, float scale, float* y, int nt, int N, int K);
// RMSNorm + absmax int8 quantisation of nt rows (mode 0) -> q (nt*K), deq (nt), sumq (nt)
void cuda_test_ternary_mma(const uint8_t* W, const int8_t* q, const float* deq, const int32_t* sumq, float scale, float* y, int M, int N, int K);
void cuda_test_ternary_gs(const uint8_t* W, const uint16_t* gs, const int8_t* q, const float* deq, const int32_t* sumq, float* y, int M, int N, int K, bool tc);
void cuda_test_rmsnorm_quant(const float* x, const float* w, float eps, int nt, int K, int8_t* q, float* deq, int32_t* sumq);
}
#endif
