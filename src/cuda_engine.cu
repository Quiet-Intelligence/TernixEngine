// CUDA forward pass. Kernel inventory (all single-stream, no host synchronisation inside a forward chunk):
//   embed            bf16 row gather -> float residual stream
//   rmsnorm_quant    RMSNorm (+ optional squared-ReLU gating) fused with per-token absmax int8 quantisation
//   ternary_gemv<NT> I2K128 weights x NT int8 activation rows, dp4a, int32 accumulation, BitLinear rescale epilogue
//   rope_append      rotary embedding on q/k, fp16 K/V cache append
//   attention        one block per (token, head): scores in shared memory, softmax, V accumulation
//   lm_head<NT>      bf16 tied embedding rows x float hidden vectors
//   argmax           greedy next token without host round trip
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "ternix/cuda_engine.hpp"
#include "ternix/i2k.hpp"

namespace ternix {
namespace {

#define CK(x) check_cuda((x), #x, __LINE__)
void check_cuda(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error at ") + what + " (line " + std::to_string(line) + "): " + cudaGetErrorString(e));
}

constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
    return v;
}
__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
    return v;
}
// block-wide reductions; `sh` holds at least 32 elements
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float r = ((threadIdx.x & 31) < nw) ? sh[threadIdx.x & 31] : 0.f;
    return warp_sum(r);
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float r = ((threadIdx.x & 31) < nw) ? sh[threadIdx.x & 31] : -INFINITY;
    return warp_max(r);
}
__device__ int block_sum_i(int v, int* sh) {
    v = warp_sum_i(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    int r = ((threadIdx.x & 31) < nw) ? sh[threadIdx.x & 31] : 0;
    return warp_sum_i(r);
}

// unsigned bytes of a x signed bytes of b, accumulated into c (PTX dp4a.u32.s32; CUDA has no mixed-sign intrinsic)
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__global__ void embed_kernel(const uint16_t* __restrict__ emb, const int32_t* __restrict__ tokens, float* __restrict__ x, int H) {
    const size_t t = blockIdx.x;
    const uint16_t* row = emb + static_cast<size_t>(tokens[t]) * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[t * H + i] = bf16_to_f32(row[i]);
}

// MODE 0: v = in[i]; MODE 1: v = relu(gate[i])^2 * up[i]   (in = gate, in2 = up)
// Writes the normalised row (float, optional) and its int8 quantisation with the per-token scale and sum.
template <int MODE>
__global__ void rmsnorm_quant_kernel(const float* __restrict__ in, const float* __restrict__ in2, const float* __restrict__ w, float eps,
                                     int K, float* __restrict__ out_f, int8_t* __restrict__ q, float* __restrict__ deq,
                                     int32_t* __restrict__ sumq) {
    extern __shared__ float sv[];            // K floats: the pre-norm values, then the normalised values
    __shared__ float red[32];
    __shared__ int redi[32];
    const size_t t = blockIdx.x;
    const float* row = in + t * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v;
        if (MODE == 0) v = row[i];
        else {
            const float g = fmaxf(row[i], 0.f);
            v = g * g * in2[t * K + i];
        }
        sv[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(K) + eps);
    float amax = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float v = (sv[i] * inv) * w[i];
        sv[i] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = block_max(amax, red);
    amax = fmaxf(amax, 1e-5f);
    const float s = 127.0f / amax;
    int sum = 0;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float v = sv[i];
        if (out_f) out_f[t * K + i] = v;
        float r = rintf(v * s);
        r = fminf(127.f, fmaxf(-128.f, r));
        q[t * K + i] = static_cast<int8_t>(static_cast<int>(r));
        sum += static_cast<int>(r);
    }
    sum = block_sum_i(sum, redi);
    if (threadIdx.x == 0) {
        deq[t] = 1.0f / s;
        sumq[t] = sum;
    }
}

// Final RMSNorm without quantisation (head input).
__global__ void rmsnorm_kernel(const float* __restrict__ in, const float* __restrict__ w, float eps, int K, float* __restrict__ out) {
    __shared__ float red[32];
    const size_t t = blockIdx.x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) ss += in[t * K + i] * in[t * K + i];
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(K) + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x) out[t * K + i] = (in[t * K + i] * inv) * w[i];
}

// y[t][row] = (sum_k q[t][k] * W[row][k] - sum_q[t]) * deq[t] * scale   (+= when ACC)
// W is I2K128: per 128 columns, 32 bytes, byte l holding the codes of columns 128b + 32j + l in bit pair j.
// One warp handles 2 rows; lane i walks 16-byte chunks i, i+32, ... (a chunk is bytes 16c..16c+15 of block b).
template <int NT, bool ACC>
__global__ void __launch_bounds__(128) ternary_gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ q,
                                                           const float* __restrict__ deq, const int32_t* __restrict__ sumq,
                                                           float scale, float* __restrict__ out, int N, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * 2;
    if (row0 >= N) return;
    const int nchunks = K / 64;
    const uint8_t* w0 = W + static_cast<size_t>(row0) * (K / 4);
    const uint8_t* w1 = (row0 + 1 < N) ? w0 + K / 4 : w0;
    int acc0[NT], acc1[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = 0;
    for (int cidx = lane; cidx < nchunks; cidx += 32) {
        const uint4 a = __ldcs(reinterpret_cast<const uint4*>(w0) + cidx);
        const uint4 b = __ldcs(reinterpret_cast<const uint4*>(w1) + cidx);
        const int blk = cidx >> 1, c = cidx & 1;
        const int base = blk * 128 + c * 16;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t* qt = q + static_cast<size_t>(t) * K + base;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 av = *reinterpret_cast<const uint4*>(qt + 32 * j);
                const int sh = 2 * j;
                const unsigned m = 0x03030303u;
                acc0[t] = dp4a_us((a.x >> sh) & m, static_cast<int>(av.x), acc0[t]);
                acc0[t] = dp4a_us((a.y >> sh) & m, static_cast<int>(av.y), acc0[t]);
                acc0[t] = dp4a_us((a.z >> sh) & m, static_cast<int>(av.z), acc0[t]);
                acc0[t] = dp4a_us((a.w >> sh) & m, static_cast<int>(av.w), acc0[t]);
                acc1[t] = dp4a_us((b.x >> sh) & m, static_cast<int>(av.x), acc1[t]);
                acc1[t] = dp4a_us((b.y >> sh) & m, static_cast<int>(av.y), acc1[t]);
                acc1[t] = dp4a_us((b.z >> sh) & m, static_cast<int>(av.z), acc1[t]);
                acc1[t] = dp4a_us((b.w >> sh) & m, static_cast<int>(av.w), acc1[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int s0 = warp_sum_i(acc0[t]);
        const int s1 = warp_sum_i(acc1[t]);
        if (lane == 0) {
            const float f = deq[t] * scale;
            const float y0 = static_cast<float>(s0 - sumq[t]) * f;
            const float y1 = static_cast<float>(s1 - sumq[t]) * f;
            float* o = out + static_cast<size_t>(t) * N;
            if (ACC) { o[row0] += y0; if (row0 + 1 < N) o[row0 + 1] += y1; }
            else { o[row0] = y0; if (row0 + 1 < N) o[row0 + 1] = y1; }
        }
    }
}

__global__ void rope_append_kernel(float* __restrict__ qv, float* __restrict__ kv, const float* __restrict__ vv,
                                   const float* __restrict__ cos_t, const float* __restrict__ sin_t, uint16_t* __restrict__ kc,
                                   uint16_t* __restrict__ vc, int nh, int nkv, int hd, int max_ctx, int pos0) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int half = hd / 2;
    const float* c = cos_t + static_cast<size_t>(pos0 + t) * half;
    const float* s = sin_t + static_cast<size_t>(pos0 + t) * half;
    if (h < nh) {
        float* v = qv + (static_cast<size_t>(t) * nh + h) * hd;
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const float a = v[i], b = v[i + half];
            v[i] = a * c[i] - b * s[i];
            v[i + half] = b * c[i] + a * s[i];
        }
    } else {
        const int g = h - nh;
        float* k = kv + (static_cast<size_t>(t) * nkv + g) * hd;
        const float* v = vv + (static_cast<size_t>(t) * nkv + g) * hd;
        uint16_t* kd = kc + (static_cast<size_t>(g) * max_ctx + pos0 + t) * hd;
        uint16_t* vd = vc + (static_cast<size_t>(g) * max_ctx + pos0 + t) * hd;
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const float a = k[i], b = k[i + half];
            const float ra = a * c[i] - b * s[i], rb = b * c[i] + a * s[i];
            kd[i] = __half_as_ushort(__float2half_rn(ra));
            kd[i + half] = __half_as_ushort(__float2half_rn(rb));
            vd[i] = __half_as_ushort(__float2half_rn(v[i]));
            vd[i + half] = __half_as_ushort(__float2half_rn(v[i + half]));
        }
    }
}

// grid (n_tokens, n_heads), blockDim = head_dim (<= 256). dynamic smem: (len + head_dim) floats.
__global__ void attention_kernel(const float* __restrict__ qv, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                                 float* __restrict__ ctx, int nh, int nkv, int hd, int max_ctx, int pos0, float scale) {
    extern __shared__ float smem[];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int len = pos0 + t + 1;
    float* scores = smem;
    float* qs = smem + len;
    const float* q = qv + (static_cast<size_t>(t) * nh + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = q[i];
    __syncthreads();
    const uint16_t* K = kc + static_cast<size_t>(g) * max_ctx * hd;
    const uint16_t* V = vc + static_cast<size_t>(g) * max_ctx * hd;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float mx = -INFINITY;
    for (int j = warp; j < len; j += nw) {
        float s = 0.f;
        for (int d = lane; d < hd; d += 32) s += __half2float(__ushort_as_half(K[static_cast<size_t>(j) * hd + d])) * qs[d];
        s = warp_sum(s) * scale;
        if (lane == 0) scores[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float e = __expf(scores[j] - mx);
        scores[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    const float inv = 1.0f / sum;
    if (threadIdx.x < hd) {
        float acc = 0.f;
        const int d = threadIdx.x;
        int j = 0;
        for (; j + 4 <= len; j += 4) {
            acc += scores[j] * __half2float(__ushort_as_half(V[static_cast<size_t>(j) * hd + d]));
            acc += scores[j + 1] * __half2float(__ushort_as_half(V[static_cast<size_t>(j + 1) * hd + d]));
            acc += scores[j + 2] * __half2float(__ushort_as_half(V[static_cast<size_t>(j + 2) * hd + d]));
            acc += scores[j + 3] * __half2float(__ushort_as_half(V[static_cast<size_t>(j + 3) * hd + d]));
        }
        for (; j < len; ++j) acc += scores[j] * __half2float(__ushort_as_half(V[static_cast<size_t>(j) * hd + d]));
        ctx[(static_cast<size_t>(t) * nh + h) * hd + d] = acc * inv;
    }
}

// logits[t][v] = dot(emb[v], x[t]); one warp per vocabulary row.
template <int NT>
__global__ void __launch_bounds__(256) lm_head_kernel(const uint16_t* __restrict__ emb, const float* __restrict__ x, float* __restrict__ logits,
                                                      int V, int H) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.x * 8 + warp;
    if (v >= V) return;
    const uint4* row = reinterpret_cast<const uint4*>(emb + static_cast<size_t>(v) * H);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c = lane; c < H / 8; c += 32) {
        const uint4 r = __ldcs(row + c);
        const uint32_t words[4] = {r.x, r.y, r.z, r.w};
        float w[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[2 * i] = __uint_as_float(words[i] << 16);
            w[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 x0 = *reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * H + c * 8);
            const float4 x1 = *reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * H + c * 8 + 4);
            acc[t] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) logits[static_cast<size_t>(t) * V + v] = s;
    }
}

__global__ void argmax_kernel(const float* __restrict__ logits, int V, int32_t* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int besti = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        if (logits[i] > best) { best = logits[i]; besti = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(kFull, best, o);
        const int oi = __shfl_xor_sync(kFull, besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bi[threadIdx.x >> 5] = besti; }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = blockDim.x >> 5;
        best = threadIdx.x < nw ? bv[threadIdx.x] : -INFINITY;
        besti = threadIdx.x < nw ? bi[threadIdx.x] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(kFull, best, o);
            const int oi = __shfl_xor_sync(kFull, besti, o);
            if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
        }
        if (threadIdx.x == 0) *out = besti;
    }
}

// nll[t] = logsumexp(logits[t]) - logits[t][target[t]]; one block per row.
__global__ void nll_kernel(const float* __restrict__ logits, const int32_t* __restrict__ targets, int V, float* __restrict__ nll,
                           int32_t* __restrict__ argmax_out) {
    __shared__ float red[32];
    __shared__ int best;
    const size_t t = blockIdx.x;
    const float* l = logits + t * V;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, l[i]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) sum += __expf(l[i] - mx);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) {
        nll[t] = targets[t] < 0 ? 0.f : (mx + __logf(sum)) - l[targets[t]];
        best = 0x7fffffff;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < V; i += blockDim.x)
        if (l[i] == mx) atomicMin(&best, i);
    __syncthreads();
    if (threadIdx.x == 0) argmax_out[t] = best;
}

struct DevLinear {
    const uint8_t* w = nullptr;
    int N = 0, K = 0;
    float scale = 1.f;
};
struct DevLayer {
    DevLinear q, k, v, o, gate, up, down;
    const float *in_norm, *post_norm, *attn_sub, *ffn_sub;
};

} // namespace

struct CudaModel::Impl {
    const ModelFile& file;
    ModelConfig cfg;
    int max_ctx, pos = 0;
    size_t bytes = 0;
    std::vector<void*> allocs;
    const uint16_t* emb = nullptr;
    const float* final_norm = nullptr;
    std::vector<DevLayer> layers;
    float *x, *xn, *q, *k, *v, *ctx, *gate, *up, *tmpf, *logits;
    int8_t* q8;
    float* deq;
    int32_t *sumq, *d_tokens, *d_argmax;
    float *cos_t, *sin_t;
    uint16_t *kc, *vc;
    int hmem = 0;
    bool debug = std::getenv("TERNIX_DEBUG") != nullptr;
    void dump(const char* what, uint32_t l, const float* d, int len) {
        std::vector<float> h(len);
        CK(cudaMemcpy(h.data(), d, size_t(len) * 4, cudaMemcpyDeviceToHost));
        double ss = 0;
        for (float v : h) ss += double(v) * v;
        std::fprintf(stderr, "[dbg] %s %u rms %.6f first %.5f %.5f %.5f\n", what, l, std::sqrt(ss / len), h[0], h[1], h[2]);
    }

    explicit Impl(const ModelFile& f, int ctx_len) : file(f), cfg(f.config()), max_ctx(std::min<int>(ctx_len, static_cast<int>(f.config().max_pos))) {
        if (cfg.head_dim > 256 || cfg.head_dim % 32 != 0) throw std::runtime_error("cuda engine: unsupported head_dim");
        const size_t H = cfg.hidden, F = cfg.ffn, hd = cfg.head_dim, B = CudaModel::kMaxChunk;
        const size_t qdim = size_t(cfg.heads) * hd, kvdim = size_t(cfg.kv_heads) * hd;
        auto up_bytes = [&](const void* host, size_t n) {
            void* d = nullptr;
            CK(cudaMalloc(&d, n));
            CK(cudaMemcpy(d, host, n, cudaMemcpyHostToDevice));
            allocs.push_back(d);
            bytes += n;
            return d;
        };
        auto alloc = [&](size_t n) {
            void* d = nullptr;
            CK(cudaMalloc(&d, n));
            CK(cudaMemset(d, 0, n));
            allocs.push_back(d);
            bytes += n;
            return d;
        };
        auto lin = [&](const std::string& name) {
            const TensorView& t = f.get(name);
            DevLinear l;
            l.w = static_cast<const uint8_t*>(up_bytes(t.data, t.nbytes));
            l.N = static_cast<int>(t.dims[0]);
            l.K = static_cast<int>(t.dims[1]);
            l.scale = t.scale;
            return l;
        };
        auto norm = [&](const std::string& name) { const TensorView& t = f.get(name); return static_cast<const float*>(up_bytes(t.data, t.nbytes)); };
        const TensorView& e = f.get("model.embed_tokens.weight");
        emb = static_cast<const uint16_t*>(up_bytes(e.data, e.nbytes));
        final_norm = norm("model.norm.weight");
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            DevLayer L;
            L.q = lin(p + "self_attn.q_proj.weight"); L.k = lin(p + "self_attn.k_proj.weight");
            L.v = lin(p + "self_attn.v_proj.weight"); L.o = lin(p + "self_attn.o_proj.weight");
            L.gate = lin(p + "mlp.gate_proj.weight"); L.up = lin(p + "mlp.up_proj.weight"); L.down = lin(p + "mlp.down_proj.weight");
            L.in_norm = norm(p + "input_layernorm.weight"); L.post_norm = norm(p + "post_attention_layernorm.weight");
            L.attn_sub = norm(p + "self_attn.attn_sub_norm.weight"); L.ffn_sub = norm(p + "mlp.ffn_sub_norm.weight");
            layers.push_back(L);
        }
        x = (float*)alloc(B * H * 4); xn = (float*)alloc(B * H * 4); q = (float*)alloc(B * qdim * 4);
        k = (float*)alloc(B * kvdim * 4); v = (float*)alloc(B * kvdim * 4); ctx = (float*)alloc(B * qdim * 4);
        gate = (float*)alloc(B * F * 4); up = (float*)alloc(B * F * 4); tmpf = (float*)alloc(B * H * 4);
        logits = (float*)alloc(B * size_t(cfg.vocab) * 4);
        q8 = (int8_t*)alloc(B * std::max(H, F)); deq = (float*)alloc(B * 4); sumq = (int32_t*)alloc(B * 4);
        d_tokens = (int32_t*)alloc(B * 4); d_argmax = (int32_t*)alloc(4);
        const size_t half = hd / 2;
        std::vector<float> ct(size_t(max_ctx) * half), st(size_t(max_ctx) * half);
        for (int p = 0; p < max_ctx; ++p)
            for (size_t i = 0; i < half; ++i) {
                const double freq = std::pow(static_cast<double>(cfg.rope_theta), -2.0 * static_cast<double>(i) / static_cast<double>(hd));
                ct[size_t(p) * half + i] = static_cast<float>(std::cos(p * freq));
                st[size_t(p) * half + i] = static_cast<float>(std::sin(p * freq));
            }
        cos_t = (float*)up_bytes(ct.data(), ct.size() * 4);
        sin_t = (float*)up_bytes(st.data(), st.size() * 4);
        const size_t kv = size_t(cfg.layers) * cfg.kv_heads * max_ctx * hd * 2;
        kc = (uint16_t*)alloc(kv); vc = (uint16_t*)alloc(kv);
        const size_t attn_smem = (size_t(max_ctx) + hd) * 4;
        if (attn_smem > 48 * 1024) CK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(attn_smem)));
        const size_t rn_smem = std::max(H, F) * 4;
        if (rn_smem > 48 * 1024) {
            CK(cudaFuncSetAttribute(rmsnorm_quant_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(rn_smem)));
            CK(cudaFuncSetAttribute(rmsnorm_quant_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(rn_smem)));
        }
    }
    ~Impl() { for (void* p : allocs) cudaFree(p); }

    void gemv(const DevLinear& L, int n, float* out, bool acc) {
        for (int t0 = 0; t0 < n; t0 += 8) {
            const int nt = std::min(8, n - t0);
            const dim3 grid((L.N + 7) / 8), block(128);
            const int8_t* qa = q8 + size_t(t0) * L.K;
            float* o = out + size_t(t0) * L.N;
#define LAUNCH(NT)                                                                                                           \
    case NT:                                                                                                                 \
        if (acc) ternary_gemv_kernel<NT, true><<<grid, block>>>(L.w, qa, deq + t0, sumq + t0, L.scale, o, L.N, L.K);         \
        else ternary_gemv_kernel<NT, false><<<grid, block>>>(L.w, qa, deq + t0, sumq + t0, L.scale, o, L.N, L.K);            \
        break;
            switch (nt) { LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(6) LAUNCH(7) LAUNCH(8) }
#undef LAUNCH
        }
    }

    void head(const float* xin, int n) {
        const int H = static_cast<int>(cfg.hidden), V = static_cast<int>(cfg.vocab);
        for (int t0 = 0; t0 < n; t0 += 8) {
            const int nt = std::min(8, n - t0);
            const dim3 grid((V + 7) / 8), block(256);
#define LAUNCH(NT) case NT: lm_head_kernel<NT><<<grid, block>>>(emb, xin + size_t(t0) * H, logits + size_t(t0) * V, V, H); break;
            switch (nt) { LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(6) LAUNCH(7) LAUNCH(8) }
#undef LAUNCH
        }
    }

    // mode: 0 none, 1 last token logits left in d_logits[0..V), 2 all tokens
    void chunk(const int32_t* tokens, int n, int mode) {
        const int H = static_cast<int>(cfg.hidden), F = static_cast<int>(cfg.ffn), hd = static_cast<int>(cfg.head_dim);
        const int nh = static_cast<int>(cfg.heads), nkv = static_cast<int>(cfg.kv_heads);
        const int qdim = nh * hd;
        const float eps = cfg.rms_eps;
        CK(cudaMemcpy(d_tokens, tokens, size_t(n) * 4, cudaMemcpyHostToDevice));
        embed_kernel<<<n, 256>>>(emb, d_tokens, x, H);
        const int rn_threads = 256;
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            const DevLayer& L = layers[l];
            rmsnorm_quant_kernel<0><<<n, rn_threads, size_t(H) * 4>>>(x, nullptr, L.in_norm, eps, H, nullptr, q8, deq, sumq);
            if (debug && l == 0) { dump("x_emb", 0, x, H); std::vector<float> dq(n); std::vector<int> sq(n); CK(cudaMemcpy(dq.data(), deq, n * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(sq.data(), sumq, n * 4, cudaMemcpyDeviceToHost)); std::fprintf(stderr, "[dbg] deq0 %g sumq0 %d\n", dq[0], sq[0]); }
            gemv(L.q, n, q, false);
            gemv(L.k, n, k, false);
            gemv(L.v, n, v, false);
            if (debug && l == 0) { dump("q", 0, q, qdim); dump("k", 0, k, nkv * hd); dump("v", 0, v, nkv * hd); }
            uint16_t* kcl = kc + size_t(l) * nkv * max_ctx * hd;
            uint16_t* vcl = vc + size_t(l) * nkv * max_ctx * hd;
            rope_append_kernel<<<dim3(n, nh + nkv), 64>>>(q, k, v, cos_t, sin_t, kcl, vcl, nh, nkv, hd, max_ctx, pos);
            attention_kernel<<<dim3(n, nh), hd, (size_t(pos + n) + hd) * 4>>>(q, kcl, vcl, ctx, nh, nkv, hd, max_ctx, pos, 1.0f / std::sqrt(static_cast<float>(hd)));
            if (debug && l == 0) { dump("q_rope", 0, q, qdim); dump("ctx", 0, ctx, qdim); }
            rmsnorm_quant_kernel<0><<<n, rn_threads, size_t(qdim) * 4>>>(ctx, nullptr, L.attn_sub, eps, qdim, nullptr, q8, deq, sumq);
            gemv(L.o, n, x, true);
            if (debug && l == 0) dump("x_after_o", 0, x, H);
            rmsnorm_quant_kernel<0><<<n, rn_threads, size_t(H) * 4>>>(x, nullptr, L.post_norm, eps, H, nullptr, q8, deq, sumq);
            gemv(L.gate, n, gate, false);
            gemv(L.up, n, up, false);
            rmsnorm_quant_kernel<1><<<n, rn_threads, size_t(F) * 4>>>(gate, up, L.ffn_sub, eps, F, nullptr, q8, deq, sumq);
            gemv(L.down, n, x, true);
            if (debug) dump("layer", l, x, H);
        }
        pos += n;
        if (mode == 0) return;
        const int rows = mode == 2 ? n : 1;
        const float* src = x + (mode == 2 ? 0 : size_t(n - 1) * H);
        rmsnorm_kernel<<<rows, 256>>>(src, final_norm, eps, H, xn);
        head(xn, rows);
    }
};

CudaModel::CudaModel(const ModelFile& file, int max_ctx) : p_(new Impl(file, max_ctx)) {}
CudaModel::~CudaModel() = default;
const ModelConfig& CudaModel::config() const { return p_->cfg; }
int CudaModel::position() const { return p_->pos; }
int CudaModel::max_ctx() const { return p_->max_ctx; }
void CudaModel::reset() { p_->pos = 0; }
void CudaModel::truncate(int pos) { p_->pos = pos; }
size_t CudaModel::device_bytes() const { return p_->bytes; }

bool CudaModel::forward(const int32_t* tokens, int n, float* logits, int mode) {
    Impl& s = *p_;
    if (s.pos + n > s.max_ctx) return false;
    const size_t V = s.cfg.vocab;
    int done = 0;
    while (done < n) {
        const int c = std::min(kMaxChunk, n - done);
        const bool last = done + c == n;
        int m = 0;
        if (mode == 2) m = 2;
        else if (mode == 1 && last) m = 1;
        s.chunk(tokens + done, c, m);
        CK(cudaGetLastError());
        if (m == 2) CK(cudaMemcpy(logits + size_t(done) * V, s.logits, size_t(c) * V * 4, cudaMemcpyDeviceToHost));
        else if (m == 1) CK(cudaMemcpy(logits, s.logits, V * 4, cudaMemcpyDeviceToHost));
        else CK(cudaDeviceSynchronize());
        done += c;
    }
    return true;
}

bool CudaModel::forward_nll(const int32_t* tokens, int n, const int32_t* targets, float* nll, int32_t* argmax) {
    Impl& s = *p_;
    if (s.pos + n > s.max_ctx) return false;
    int32_t* d_t = nullptr;
    float* d_n = nullptr;
    int32_t* d_a = nullptr;
    CK(cudaMalloc(&d_t, size_t(kMaxChunk) * 4));
    CK(cudaMalloc(&d_n, size_t(kMaxChunk) * 4));
    CK(cudaMalloc(&d_a, size_t(kMaxChunk) * 4));
    for (int done = 0; done < n; done += kMaxChunk) {
        const int c = std::min(kMaxChunk, n - done);
        s.chunk(tokens + done, c, 2);
        CK(cudaMemcpy(d_t, targets + done, size_t(c) * 4, cudaMemcpyHostToDevice));
        nll_kernel<<<c, 1024>>>(s.logits, d_t, static_cast<int>(s.cfg.vocab), d_n, d_a);
        CK(cudaGetLastError());
        CK(cudaMemcpy(nll + done, d_n, size_t(c) * 4, cudaMemcpyDeviceToHost));
        if (argmax) CK(cudaMemcpy(argmax + done, d_a, size_t(c) * 4, cudaMemcpyDeviceToHost));
    }
    cudaFree(d_t);
    cudaFree(d_n);
    cudaFree(d_a);
    return true;
}

int32_t CudaModel::step_argmax(int32_t token) {
    Impl& s = *p_;
    if (s.pos + 1 > s.max_ctx) return -1;
    s.chunk(&token, 1, 1);
    argmax_kernel<<<1, 1024>>>(s.logits, static_cast<int>(s.cfg.vocab), s.d_argmax);
    int32_t r = -1;
    CK(cudaMemcpy(&r, s.d_argmax, 4, cudaMemcpyDeviceToHost));
    return r;
}

} // namespace ternix

namespace ternix {

void cuda_test_ternary_gemv(const uint8_t* W, const int8_t* q, const float* deq, const int32_t* sumq, float scale, float* y, int nt, int N, int K) {
    uint8_t* dW; int8_t* dq; float *dd, *dy; int32_t* ds;
    CK(cudaMalloc(&dW, size_t(N) * K / 4)); CK(cudaMalloc(&dq, size_t(nt) * K)); CK(cudaMalloc(&dd, nt * 4));
    CK(cudaMalloc(&ds, nt * 4)); CK(cudaMalloc(&dy, size_t(nt) * N * 4));
    CK(cudaMemcpy(dW, W, size_t(N) * K / 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dq, q, size_t(nt) * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, deq, nt * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, sumq, nt * 4, cudaMemcpyHostToDevice));
    const dim3 grid((N + 7) / 8), block(128);
#define LAUNCH(NT) case NT: ternary_gemv_kernel<NT, false><<<grid, block>>>(dW, dq, dd, ds, scale, dy, N, K); break;
    switch (nt) { LAUNCH(1) LAUNCH(2) LAUNCH(3) LAUNCH(4) LAUNCH(5) LAUNCH(6) LAUNCH(7) LAUNCH(8) default: throw std::runtime_error("nt"); }
#undef LAUNCH
    CK(cudaGetLastError());
    CK(cudaMemcpy(y, dy, size_t(nt) * N * 4, cudaMemcpyDeviceToHost));
    cudaFree(dW); cudaFree(dq); cudaFree(dd); cudaFree(ds); cudaFree(dy);
}

void cuda_test_rmsnorm_quant(const float* x, const float* w, float eps, int nt, int K, int8_t* q, float* deq, int32_t* sumq) {
    float *dx, *dw, *dd; int8_t* dq; int32_t* ds;
    CK(cudaMalloc(&dx, size_t(nt) * K * 4)); CK(cudaMalloc(&dw, K * 4)); CK(cudaMalloc(&dq, size_t(nt) * K));
    CK(cudaMalloc(&dd, nt * 4)); CK(cudaMalloc(&ds, nt * 4));
    CK(cudaMemcpy(dx, x, size_t(nt) * K * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw, w, K * 4, cudaMemcpyHostToDevice));
    rmsnorm_quant_kernel<0><<<nt, 256, size_t(K) * 4>>>(dx, nullptr, dw, eps, K, nullptr, dq, dd, ds);
    CK(cudaGetLastError());
    CK(cudaMemcpy(q, dq, size_t(nt) * K, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(deq, dd, nt * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sumq, ds, nt * 4, cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dw); cudaFree(dq); cudaFree(dd); cudaFree(ds);
}

} // namespace ternix
