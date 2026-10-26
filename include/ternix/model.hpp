#pragma once
// CPU forward pass for BitNet-style decoder-only transformers stored in a .ternix file.
//
//   x <- x + o_proj( attn_sub_norm( attention( rope(q_proj(n1)), rope(k_proj(n1)), v_proj(n1) ) ) ),  n1 = quant(rmsnorm(x))
//   x <- x + down_proj( ffn_sub_norm( act(gate_proj(n2)) * up_proj(n2) ) ),                           n2 = quant(rmsnorm(x))
//
// Every projection is a BitLinear: per-token absmax int8 activations x ternary weights (I2K128), int32 accumulation,
// result scaled by (activation step) * (per-tensor weight scale). Norms, rotary embedding, softmax and the residual
// stream are float32; the KV cache is fp16; the tied embedding / output head is bf16.
#include <chrono>
#include <memory>
#include <vector>
#include "ternix/cpu_features.hpp"
#include "ternix/format.hpp"
#include "ternix/head.hpp"
#include "ternix/i2k.hpp"
#include "ternix/threadpool.hpp"
#include "ternix/util.hpp"

namespace ternix {

struct Linear {
    const uint8_t* w = nullptr;
    size_t N = 0, K = 0;
    float scale = 1.f;
};

struct LayerWeights {
    Linear q, k, v, o, gate, up, down;
    const float *in_norm = nullptr, *post_norm = nullptr, *attn_sub = nullptr, *ffn_sub = nullptr;
};

struct Profile {
    double quant_ms = 0, linear_ms = 0, attn_ms = 0, norm_ms = 0, head_ms = 0, total_ms = 0;
    uint64_t tokens = 0;
};

class Model {
public:
    // threads <= 0: one per performance core. max_ctx is capped at the model's trained context.
    Model(const ModelFile& file, int threads = 0, int max_ctx = 4096, Isa isa = best_isa());

    const ModelConfig& config() const { return cfg_; }
    int position() const { return pos_; }
    int max_ctx() const { return max_ctx_; }
    void reset() { pos_ = 0; }
    // Rewinds the sequence to `pos` (KV entries beyond it are simply overwritten later).
    void truncate(int pos) { pos_ = pos; }
    ThreadPool& pool() { return pool_; }
    Isa isa() const { return isa_; }
    Profile& profile() { return prof_; }

    // Feeds n tokens at the current position. logits_mode: 0 = none, 1 = last token only -> logits[vocab],
    // 2 = every token -> logits[n][vocab]. Returns false if the context window would overflow.
    bool forward(const int32_t* tokens, int n, float* logits, int logits_mode = 1);

    // Output head precision for batch-1 decode (HeadMode::Bf16 is the reference; Int8/Int4 are certified-exact for greedy).
    // Builds the compressed copy on first use (a second or two for 128k x 2560).
    void set_head(HeadMode mode);
    HeadMode head_mode() const { return head_mode_; }
    const uint16_t* embedding() const { return embed_; }
    OutputHead* head() { return head_; }
    // Exact activation-sparse down projection at batch 1 (bit-identical results; builds a K-major copy of every down matrix, +25%
    // of the ternary weight memory, on first use).
    void set_sparse_down(bool on);
    bool sparse_down() const { return sparse_down_; }
    double mean_down_nnz_fraction() const { return sparse_calls_ ? double(sparse_nnz_) / (double(sparse_calls_) * cfg_.ffn) : 0.0; }
    // logits_mode 3 of forward(): final-norm hidden vectors, hidden[n][H] (input of the head).

    // Teacher-forced scoring (see CudaModel::forward_nll): nll[i] for the logits after tokens[i] against targets[i].
    bool forward_nll(const int32_t* tokens, int n, const int32_t* targets, float* nll, int32_t* argmax = nullptr);

    static constexpr int kMaxChunk = 64;   // tokens per internal forward chunk

private:
    struct Job { const Linear* lin; float* out; bool accumulate; };
    void quantize_rows(const float* x, int n, size_t K);
    void run_linears(const Job* jobs, int njobs, int n);
    void chunk_forward(const int32_t* tokens, int n, float* logits, int logits_mode);
    void attention(int layer, int n);
    void lm_head(const float* xn, int n, float* logits, bool decode);

    const ModelFile& file_;
    ModelConfig cfg_;
    Isa isa_;
    ThreadPool pool_;
    int max_ctx_, pos_ = 0;
    const uint16_t* embed_ = nullptr;
    std::unique_ptr<OutputHead> heads_[3];        // built lazily per mode and kept, so modes can be switched cheaply
    OutputHead* head_ = nullptr;                  // active compressed head (nullptr = bf16 reference path)
    HeadMode head_mode_ = HeadMode::Bf16;
    const float* final_norm_ = nullptr;
    std::vector<LayerWeights> layers_;
    bool sparse_down_ = false;
    std::vector<AlignedBuf<uint8_t>> down_k_;      // per layer: K-major copy of the down projection
    i2k::SparseScratch sparse_scratch_;
    AlignedBuf<int32_t> sparse_y_;
    uint64_t sparse_nnz_ = 0, sparse_calls_ = 0;

    // scratch (sized for kMaxChunk tokens)
    AlignedBuf<float> x_, xn_, q_, k_, v_, ctx_, gate_, up_, hid_, tmp_;
    AlignedBuf<int8_t> q8_;
    AlignedBuf<float> deq_;
    AlignedBuf<int32_t> sumq_;
    std::vector<AlignedBuf<float>> scores_;     // per thread
    AlignedBuf<float> attn_out_;
    AlignedBuf<float> rope_cos_, rope_sin_;      // [max_ctx][head_dim/2]
    AlignedBuf<uint16_t> kcache_, vcache_;       // [layer][kv_head][max_ctx][head_dim]
    Profile prof_;
};

} // namespace ternix
