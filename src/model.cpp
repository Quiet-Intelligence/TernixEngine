#include "ternix/model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "ternix/i2k.hpp"
#include "ternix/ops.hpp"
#include "ternix/simd_math.hpp"

namespace ternix {
namespace {

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

Linear load_linear(const ModelFile& f, const std::string& name) {
    const TensorView& t = f.get(name);
    if (t.dtype != DType::I2K128 || t.dims.size() != 2) throw std::runtime_error("ternix: '" + name + "' is not a ternary matrix");
    Linear l;
    l.w = t.data;
    l.N = t.dims[0];
    l.K = t.dims[1];
    l.scale = t.scale;
    if (l.K % i2k::kBlock) throw std::runtime_error("ternix: K of '" + name + "' is not a multiple of 128");
    return l;
}

const float* load_f32(const ModelFile& f, const std::string& name, size_t n) {
    const TensorView& t = f.get(name);
    if (t.dtype != DType::F32 || t.nbytes != n * 4) throw std::runtime_error("ternix: bad norm tensor '" + name + "'");
    return reinterpret_cast<const float*>(t.data);
}

} // namespace

Model::Model(const ModelFile& file, int threads, int max_ctx, Isa isa)
    : file_(file), cfg_(file.config()), isa_(isa), pool_(threads, true),
      max_ctx_(std::min<int>(max_ctx, static_cast<int>(file.config().max_pos))) {
    const size_t H = cfg_.hidden, F = cfg_.ffn, hd = cfg_.head_dim;
    const size_t qdim = size_t(cfg_.heads) * hd, kvdim = size_t(cfg_.kv_heads) * hd;
    if (hd % 8 != 0 || hd > 256) throw std::runtime_error("ternix: unsupported head_dim");
    if (cfg_.act != ModelConfig::Act::Relu2Glu || !cfg_.subln) throw std::runtime_error("ternix: only the BitNet block (relu2 + sub-norms) is implemented");

    const TensorView& emb = file.get("model.embed_tokens.weight");
    if (emb.dtype != DType::BF16) throw std::runtime_error("ternix: embedding must be bf16");
    embed_ = reinterpret_cast<const uint16_t*>(emb.data);
    final_norm_ = load_f32(file, "model.norm.weight", H);
    for (uint32_t l = 0; l < cfg_.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        LayerWeights w;
        w.q = load_linear(file, p + "self_attn.q_proj.weight");
        w.k = load_linear(file, p + "self_attn.k_proj.weight");
        w.v = load_linear(file, p + "self_attn.v_proj.weight");
        w.o = load_linear(file, p + "self_attn.o_proj.weight");
        w.gate = load_linear(file, p + "mlp.gate_proj.weight");
        w.up = load_linear(file, p + "mlp.up_proj.weight");
        w.down = load_linear(file, p + "mlp.down_proj.weight");
        w.in_norm = load_f32(file, p + "input_layernorm.weight", H);
        w.post_norm = load_f32(file, p + "post_attention_layernorm.weight", H);
        w.attn_sub = load_f32(file, p + "self_attn.attn_sub_norm.weight", qdim);
        w.ffn_sub = load_f32(file, p + "mlp.ffn_sub_norm.weight", F);
        layers_.push_back(w);
    }
    const size_t B = kMaxChunk;
    x_.reset(B * H); xn_.reset(B * H); q_.reset(B * qdim); k_.reset(B * kvdim); v_.reset(B * kvdim);
    ctx_.reset(B * qdim); gate_.reset(B * F); up_.reset(B * F); hid_.reset(B * F); tmp_.reset(B * H);
    q8_.reset(B * std::max(H, F));
    deq_.reset(B); sumq_.reset(B);
    scores_.resize(static_cast<size_t>(pool_.size()));
    for (auto& s : scores_) s.reset(static_cast<size_t>(max_ctx_) + 16);
    attn_out_.reset(B * qdim);
    const size_t half = hd / 2;
    rope_cos_.reset(size_t(max_ctx_) * half);
    rope_sin_.reset(size_t(max_ctx_) * half);
    for (int p = 0; p < max_ctx_; ++p)
        for (size_t i = 0; i < half; ++i) {
            const double freq = std::pow(static_cast<double>(cfg_.rope_theta), -2.0 * static_cast<double>(i) / static_cast<double>(hd));
            rope_cos_[size_t(p) * half + i] = static_cast<float>(std::cos(p * freq));
            rope_sin_[size_t(p) * half + i] = static_cast<float>(std::sin(p * freq));
        }
    const size_t kv = size_t(cfg_.layers) * cfg_.kv_heads * max_ctx_ * hd;
    kcache_.reset(kv);
    vcache_.reset(kv);
}

void Model::quantize_rows(const float* x, int n, size_t K) {
    for (int t = 0; t < n; ++t)
        i2k::quantize_act(x + t * K, q8_.data() + t * K, K, &deq_[t], &sumq_[t], isa_);
}

// One parallel region over the output-row chunks of every job (jobs share the already-quantised activations).
void Model::run_linears(const Job* jobs, int njobs, int n) {
    constexpr size_t kChunk = 32;
    struct Span { int job; size_t n0, n1; };
    std::vector<Span> spans;
    for (int j = 0; j < njobs; ++j)
        for (size_t n0 = 0; n0 < jobs[j].lin->N; n0 += kChunk) spans.push_back({j, n0, std::min(jobs[j].lin->N, n0 + kChunk)});
    const size_t K = jobs[0].lin->K;
    const Isa isa = isa_;
    pool_.parallel_for(spans.size(), [&](size_t c, int) {
        const Span s = spans[c];
        const Job& job = jobs[s.job];
        const Linear& L = *job.lin;
        alignas(32) int32_t y[4][kChunk];
        for (int m0 = 0; m0 < n; m0 += 4) {
            const int mr = std::min(4, n - m0);
            i2k::gemm_rows(L.w, q8_.data() + size_t(m0) * K, sumq_.data() + m0, static_cast<size_t>(mr), &y[0][0], kChunk, s.n0, s.n1, K, isa);
            for (int m = 0; m < mr; ++m) {
                const float f = deq_[m0 + m] * L.scale;
                float* out = job.out + size_t(m0 + m) * L.N + s.n0;
                if (job.accumulate)
                    for (size_t i = 0; i < s.n1 - s.n0; ++i) out[i] += static_cast<float>(y[m][i]) * f;
                else
                    for (size_t i = 0; i < s.n1 - s.n0; ++i) out[i] = static_cast<float>(y[m][i]) * f;
            }
        }
    });
}

void Model::attention(int layer, int n) {
    const size_t hd = cfg_.head_dim, nh = cfg_.heads, nkv = cfg_.kv_heads, group = nh / nkv;
    const size_t qdim = nh * hd, kvdim = nkv * hd, half = hd / 2;
    const int pos0 = pos_;
    uint16_t* kc = kcache_.data() + size_t(layer) * nkv * max_ctx_ * hd;
    uint16_t* vc = vcache_.data() + size_t(layer) * nkv * max_ctx_ * hd;

    // rotary embedding on q and k, then append k/v (fp16) to the cache
    for (int t = 0; t < n; ++t) {
        const float* c = rope_cos_.data() + size_t(pos0 + t) * half;
        const float* s = rope_sin_.data() + size_t(pos0 + t) * half;
        for (size_t h = 0; h < nh; ++h) ops::rope_apply(q_.data() + t * qdim + h * hd, c, s, hd);
        for (size_t g = 0; g < nkv; ++g) {
            ops::rope_apply(k_.data() + t * kvdim + g * hd, c, s, hd);
            if (kv_int8_) {
                const size_t base = ((size_t(layer) * nkv + g) * max_ctx_ + pos0 + t);
                ops::quantize_kv(kq_.data() + base * hd, kscale_.data() + base, k_.data() + t * kvdim + g * hd, hd);
                ops::quantize_kv(vq_.data() + base * hd, vscale_.data() + base, v_.data() + t * kvdim + g * hd, hd);
            } else {
                ops::f32_to_f16(kc + (g * max_ctx_ + pos0 + t) * hd, k_.data() + t * kvdim + g * hd, hd);
                ops::f32_to_f16(vc + (g * max_ctx_ + pos0 + t) * hd, v_.data() + t * kvdim + g * hd, hd);
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    pool_.parallel_for(size_t(n) * nh, [&](size_t task, int tid) {
        const size_t t = task / nh, h = task % nh, g = h / group;
        const size_t len = size_t(pos0) + t + 1;
        if (kv_int8_) {
            const size_t base = (size_t(layer) * nkv + g) * max_ctx_;
            ops::attend_head_i8(q_.data() + t * qdim + h * hd, kq_.data() + base * hd, kscale_.data() + base, vq_.data() + base * hd, vscale_.data() + base, len, hd,
                                scale, scores_[static_cast<size_t>(tid)].data(), ctx_.data() + t * qdim + h * hd);
            return;
        }
        ops::attend_head(q_.data() + t * qdim + h * hd, kc + g * max_ctx_ * hd, vc + g * max_ctx_ * hd, len, hd, scale,
                         scores_[static_cast<size_t>(tid)].data(), ctx_.data() + t * qdim + h * hd);
    });
}

void Model::set_kv_int8(bool on) {
    kv_int8_ = on;
    pos_ = 0;
    if (!on || kq_.size() != 0) return;
    const size_t kv = size_t(cfg_.layers) * cfg_.kv_heads * max_ctx_ * cfg_.head_dim;
    kq_.reset(kv);
    vq_.reset(kv);
    kscale_.reset(kv / cfg_.head_dim);
    vscale_.reset(kv / cfg_.head_dim);
}

void Model::set_sparse_down(bool on) {
    sparse_down_ = on;
    if (!on || !down_k_.empty()) return;
    down_k_.resize(layers_.size());
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Linear& d = layers_[l].down;
        down_k_[l].reset(d.K * d.N / 4);
        i2k::pack_kmajor_from_i2k(d.w, down_k_[l].data(), d.N, d.K);
    }
    sparse_y_.reset(cfg_.hidden);
}

void Model::set_head(HeadMode mode) {
    head_mode_ = mode;
    head_ = nullptr;
    if (mode == HeadMode::Bf16) return;
    auto& h = heads_[static_cast<int>(mode)];
    if (!h) {
        h = std::make_unique<OutputHead>(embed_, cfg_.vocab, cfg_.hidden, mode, pool_);
        // int4: the worst-case bound would keep ~every row as a candidate; use the 6.5-sigma statistical bound (see head.hpp)
        if (mode == HeadMode::Int4) h->set_bound_scale(6.5 / std::sqrt(static_cast<double>(cfg_.hidden)));
    }
    head_ = h.get();
}

void Model::lm_head(const float* xn, int n, float* logits, bool decode) {
    const size_t H = cfg_.hidden, V = cfg_.vocab;
    if (decode && n == 1 && head_) {
        head_->logits(xn, logits);
        return;
    }
    constexpr size_t kChunk = 256;
    pool_.parallel_for((V + kChunk - 1) / kChunk, [&](size_t c, int) {
        const size_t v0 = c * kChunk, v1 = std::min(V, v0 + kChunk);
        float tmp[kMaxChunk];
        for (size_t v = v0; v < v1; ++v) {
            ops::dot_bf16_rows(embed_ + v * H, xn, static_cast<size_t>(n), H, tmp);
            for (int t = 0; t < n; ++t) logits[size_t(t) * V + v] = tmp[t];
        }
    });
}

void Model::chunk_forward(const int32_t* tokens, int n, float* logits, int logits_mode) {
    const size_t H = cfg_.hidden, F = cfg_.ffn, hd = cfg_.head_dim;
    const size_t qdim = size_t(cfg_.heads) * hd, kvdim = size_t(cfg_.kv_heads) * hd;
    const float eps = cfg_.rms_eps;
    auto t0 = clk::now();
    // embedding lookup (bf16 -> f32)
    for (int t = 0; t < n; ++t) {
        const uint16_t* row = embed_ + size_t(tokens[t]) * H;
        for (size_t i = 0; i < H; ++i) {
            uint32_t b = uint32_t(row[i]) << 16;
            std::memcpy(&x_[size_t(t) * H + i], &b, 4);
        }
    }
    auto norm = [&](float* out, const float* in, const float* w, size_t len, int rows) {
        auto t = clk::now();
        for (int r = 0; r < rows; ++r) rms_norm_avx2(out + size_t(r) * len, in + size_t(r) * len, w, len, eps);
        prof_.norm_ms += ms_since(t);
    };
    auto quant = [&](const float* in, size_t K) {
        auto t = clk::now();
        quantize_rows(in, n, K);
        prof_.quant_ms += ms_since(t);
    };
    auto linears = [&](const Job* jobs, int nj) {
        auto t = clk::now();
        run_linears(jobs, nj, n);
        prof_.linear_ms += ms_since(t);
    };

    for (uint32_t l = 0; l < cfg_.layers; ++l) {
        const LayerWeights& w = layers_[l];
        norm(xn_.data(), x_.data(), w.in_norm, H, n);
        quant(xn_.data(), H);
        const Job qkv[3] = {{&w.q, q_.data(), false}, {&w.k, k_.data(), false}, {&w.v, v_.data(), false}};
        linears(qkv, 3);
        {
            auto t = clk::now();
            attention(static_cast<int>(l), n);
            prof_.attn_ms += ms_since(t);
        }
        norm(ctx_.data(), ctx_.data(), w.attn_sub, qdim, n);
        quant(ctx_.data(), qdim);
        const Job o[1] = {{&w.o, x_.data(), true}};
        linears(o, 1);

        norm(xn_.data(), x_.data(), w.post_norm, H, n);
        quant(xn_.data(), H);
        const Job gu[2] = {{&w.gate, gate_.data(), false}, {&w.up, up_.data(), false}};
        linears(gu, 2);
        for (int t = 0; t < n; ++t) ops::relu2_mul(hid_.data() + t * F, gate_.data() + t * F, up_.data() + t * F, F);
        norm(hid_.data(), hid_.data(), w.ffn_sub, F, n);
        quant(hid_.data(), F);
        if (sparse_down_ && n == 1) {
            auto t = clk::now();
            i2k::gemv_sparse(pool_, down_k_[l].data(), w.down.N, w.down.K, q8_.data(), sparse_y_.data(), sparse_scratch_, isa_);
            const float f = deq_[0] * w.down.scale;
            for (size_t i = 0; i < w.down.N; ++i) x_[i] += static_cast<float>(sparse_y_[i]) * f;
            sparse_nnz_ += sparse_scratch_.nnz;
            ++sparse_calls_;
            prof_.linear_ms += ms_since(t);
        } else {
            const Job dn[1] = {{&w.down, x_.data(), true}};
            linears(dn, 1);
        }
        if (std::getenv("TERNIX_DEBUG")) {
            double ss = 0;
            for (int t = 0; t < n; ++t)
                if (t == n - 1)
                    for (size_t i = 0; i < H; ++i) ss += double(x_[size_t(t) * H + i]) * x_[size_t(t) * H + i];
            std::fprintf(stderr, "[dbg] layer %u rms %.6f first %.5f %.5f %.5f\n", l, std::sqrt(ss / H), x_[size_t(n - 1) * H], x_[size_t(n - 1) * H + 1], x_[size_t(n - 1) * H + 2]);
        }
    }
    pos_ += n;
    if (logits_mode != 0) {
        const int rows = logits_mode == 1 ? 1 : n;
        const float* src = x_.data() + (logits_mode == 1 ? size_t(n - 1) * H : 0);
        norm(xn_.data(), src, final_norm_, H, rows);
        if (logits_mode == 3) {
            std::memcpy(logits, xn_.data(), size_t(rows) * H * sizeof(float));
        } else {
            auto t = clk::now();
            lm_head(xn_.data(), rows, logits, logits_mode == 1);
            prof_.head_ms += ms_since(t);
        }
    }
    prof_.total_ms += ms_since(t0);
    prof_.tokens += static_cast<uint64_t>(n);
    (void)kvdim;
}

bool Model::forward(const int32_t* tokens, int n, float* logits, int logits_mode) {
    if (pos_ + n > max_ctx_) return false;
    const size_t V = cfg_.vocab;
    int done = 0;
    while (done < n) {
        const int c = std::min(kMaxChunk, n - done);
        const bool last = done + c == n;
        int mode = 0;
        float* dst = nullptr;
        if (logits_mode == 2) { mode = 2; dst = logits + size_t(done) * V; }
        else if (logits_mode == 3) { mode = 3; dst = logits + size_t(done) * cfg_.hidden; }
        else if (logits_mode == 1 && last) { mode = 1; dst = logits; }
        chunk_forward(tokens + done, c, dst, mode);
        done += c;
    }
    return true;
}

bool Model::forward_nll(const int32_t* tokens, int n, const int32_t* targets, float* nll, int32_t* argmax) {
    if (pos_ + n > max_ctx_) return false;
    const size_t V = cfg_.vocab;
    std::vector<float> logits(static_cast<size_t>(kMaxChunk) * V);
    for (int done = 0; done < n; done += kMaxChunk) {
        const int c = std::min(kMaxChunk, n - done);
        chunk_forward(tokens + done, c, logits.data(), 2);
        pool_.parallel_for(static_cast<size_t>(c), [&](size_t t, int) {
            const float* l = logits.data() + t * V;
            float mx = l[0];
            size_t am = 0;
            for (size_t v = 1; v < V; ++v)
                if (l[v] > mx) { mx = l[v]; am = v; }
            double z = 0;
            for (size_t v = 0; v < V; ++v) z += std::exp(l[v] - mx);
            if (argmax) argmax[done + static_cast<int>(t)] = static_cast<int32_t>(am);
            const int32_t tg = targets[done + static_cast<int>(t)];
            nll[done + static_cast<int>(t)] = tg < 0 ? 0.f : static_cast<float>(std::log(z) + mx - l[tg]);
        });
    }
    return true;
}

} // namespace ternix
