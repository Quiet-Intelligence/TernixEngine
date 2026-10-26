#include "ternix/head.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include "ternix/ops.hpp"

namespace ternix {
namespace {

inline float bf16_to_f(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

inline uint16_t f32_to_f16_bits(float f) {
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
}
inline float f16_bits_to_f32(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

} // namespace

OutputHead::OutputHead(const uint16_t* bf16, size_t V, size_t H, HeadMode mode, ThreadPool& pool, bool f16)
    : bf16_(bf16), f16_(f16), V_(V), H_(H), mode_(mode), pool_(pool) {
    if (mode == HeadMode::Bf16) return;
    if (H % 32 != 0) throw std::runtime_error("head: hidden size must be a multiple of 32");
    resid_.reset(V);
    if (mode == HeadMode::Int8) {
        q8_.reset(V * H);
        rscale_.reset(V);
    } else {
        q4_.reset(V * H / 2);
        gscale_.reset(V * H / 32);
    }
    constexpr size_t kChunk = 256;
    pool.parallel_for((V + kChunk - 1) / kChunk, [&](size_t c, int) {
        std::vector<float> row(H), deq(H);
        for (size_t v = c * kChunk; v < std::min(V, (c + 1) * kChunk); ++v) {
            for (size_t i = 0; i < H; ++i) row[i] = f16 ? ops::f16_to_f32_scalar(bf16[v * H + i]) : bf16_to_f(bf16[v * H + i]);
            if (mode == HeadMode::Int8) {
                float amax = 0.f;
                for (float f : row) amax = std::max(amax, std::fabs(f));
                const float s = amax > 0.f ? amax / 127.f : 1.f;
                rscale_[v] = s;
                for (size_t i = 0; i < H; ++i) {
                    const int q = std::clamp(static_cast<int>(std::lrintf(row[i] / s)), -127, 127);
                    q8_[v * H + i] = static_cast<int8_t>(q);
                    deq[i] = static_cast<float>(q) * s;
                }
            } else {
                for (size_t g = 0; g < H / 32; ++g) {
                    float amax = 0.f;
                    for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(row[g * 32 + j]));
                    const uint16_t sh = f32_to_f16_bits(amax > 0.f ? amax / 7.f : 1.f);
                    const float s = f16_bits_to_f32(sh);
                    gscale_[v * (H / 32) + g] = sh;
                    for (size_t j = 0; j < 16; ++j) {
                        const int lo = std::clamp(static_cast<int>(std::lrintf(row[g * 32 + j] / s)), -7, 7);
                        const int hi = std::clamp(static_cast<int>(std::lrintf(row[g * 32 + j + 16] / s)), -7, 7);
                        q4_[v * (H / 2) + g * 16 + j] = static_cast<uint8_t>((lo + 8) | ((hi + 8) << 4));
                        deq[g * 32 + j] = static_cast<float>(lo) * s;
                        deq[g * 32 + j + 16] = static_cast<float>(hi) * s;
                    }
                }
            }
            double r2 = 0;
            for (size_t i = 0; i < H; ++i) {
                const double d = static_cast<double>(row[i]) - static_cast<double>(deq[i]);
                r2 += d * d;
            }
            resid_[v] = static_cast<float>(std::sqrt(r2)) * 1.0001f;   // slack for float rounding in the bound
        }
    });
}

size_t OutputHead::compressed_bytes() const {
    switch (mode_) {
        case HeadMode::Bf16: return V_ * H_ * 2;
        case HeadMode::Int8: return V_ * H_ + V_ * 8;
        case HeadMode::Int4: return V_ * H_ / 2 + V_ * (H_ / 32) * 2 + V_ * 4;
    }
    return 0;
}

float OutputHead::exact(const float* x, size_t v) const {
    float r;
    if (f16_) ops::dot_f16_rows(bf16_ + v * H_, x, 1, H_, &r);
    else ops::dot_bf16_rows(bf16_ + v * H_, x, 1, H_, &r);
    return r;
}

void OutputHead::score_range(const float* x, size_t v0, size_t v1, float* s) const {
    if (mode_ == HeadMode::Int8) {
        for (size_t v = v0; v < v1; ++v) {
            const int8_t* q = q8_.data() + v * H_;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0;
            for (size_t i = 0; i < H_; i += 16) {
                const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + i))));
                const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + i + 8))));
                a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), a0);
                a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), a1);
            }
            s[v - v0] = hsum8(_mm256_add_ps(a0, a1)) * rscale_[v];
        }
    } else {
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i eight = _mm256_set1_epi32(8);
        for (size_t v = v0; v < v1; ++v) {
            const uint8_t* q = q4_.data() + v * (H_ / 2);
            const uint16_t* gs = gscale_.data() + v * (H_ / 32);
            __m256 total = _mm256_setzero_ps();
            for (size_t g = 0; g < H_ / 32; ++g) {
                const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + g * 16));
                const __m128i lo = _mm_and_si128(b, m4);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
                const float* xg = x + g * 32;
                __m256 p = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo), eight)), _mm256_loadu_ps(xg));
                p = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), eight)), _mm256_loadu_ps(xg + 8), p);
                p = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi), eight)), _mm256_loadu_ps(xg + 16), p);
                p = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), eight)), _mm256_loadu_ps(xg + 24), p);
                total = _mm256_fmadd_ps(p, _mm256_set1_ps(f16_bits_to_f32(gs[g])), total);
            }
            s[v - v0] = hsum8(total);
        }
    }
}

void OutputHead::approx_logits(const float* x, float* out, float* sigma) {
    constexpr size_t kChunk = 256;
    double xx = 0;
    for (size_t i = 0; i < H_; ++i) xx += static_cast<double>(x[i]) * x[i];
    const float xn = static_cast<float>(std::sqrt(xx));
    pool_.parallel_for((V_ + kChunk - 1) / kChunk, [&](size_t c, int) {
        const size_t v0 = c * kChunk, v1 = std::min(V_, v0 + kChunk);
        score_range(x, v0, v1, out + v0);
        if (sigma)
            for (size_t v = v0; v < v1; ++v) sigma[v] = xn * resid_[v] / std::sqrt(static_cast<float>(H_));
    });
}

int32_t OutputHead::logits(const float* x, float* out) {
    constexpr size_t kChunk = 256;
    const size_t nch = (V_ + kChunk - 1) / kChunk;
    if (mode_ == HeadMode::Bf16) {
        pool_.parallel_for(nch, [&](size_t c, int) {
            for (size_t v = c * kChunk; v < std::min(V_, (c + 1) * kChunk); ++v) out[v] = exact(x, v);
        });
        int32_t best = 0;
        for (size_t v = 1; v < V_; ++v)
            if (out[v] > out[best]) best = static_cast<int32_t>(v);
        return best;
    }
    double xx = 0;
    for (size_t i = 0; i < H_; ++i) xx += static_cast<double>(x[i]) * x[i];
    const float xn = static_cast<float>(std::sqrt(xx) * bound_scale_) * 1.0001f;

    // pass 1: approximate scores (stored in out) and the best certified lower bound of each chunk
    std::vector<float> chunk_lb(nch, -std::numeric_limits<float>::infinity());
    pool_.parallel_for(nch, [&](size_t c, int) {
        const size_t v0 = c * kChunk, v1 = std::min(V_, v0 + kChunk);
        score_range(x, v0, v1, out + v0);
        float lb = -std::numeric_limits<float>::infinity();
        for (size_t v = v0; v < v1; ++v) lb = std::max(lb, out[v] - xn * resid_[v]);
        chunk_lb[c] = lb;
    });
    const float L = *std::max_element(chunk_lb.begin(), chunk_lb.end());

    // pass 2: candidates = rows whose upper bound reaches L; rescore them exactly
    std::vector<std::vector<uint32_t>> cand(nch);
    pool_.parallel_for(nch, [&](size_t c, int) {
        for (size_t v = c * kChunk; v < std::min(V_, (c + 1) * kChunk); ++v)
            if (out[v] + xn * resid_[v] >= L) cand[c].push_back(static_cast<uint32_t>(v));
    });
    std::vector<uint32_t> all;
    for (auto& c : cand) all.insert(all.end(), c.begin(), c.end());
    pool_.parallel_for(all.size(), [&](size_t i, int) { out[all[i]] = exact(x, all[i]); });
    int32_t best = static_cast<int32_t>(all[0]);
    for (uint32_t v : all)
        if (out[v] > out[best]) best = static_cast<int32_t>(v);
    ++stats_.calls;
    stats_.candidates += all.size();
    stats_.max_candidates = std::max(stats_.max_candidates, all.size());
    return best;
}

} // namespace ternix
