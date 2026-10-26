#pragma once
// Quantised output head with certified-exact greedy decoding.
//
// The tied bf16 embedding is 56% of the bytes a 2B ternary model reads per decoded token. The head here keeps a compressed copy
// (int8 per row, or int4 per group of 32) next to the bf16 matrix and evaluates a logit as
//     s_v = sum_i x_i * deq(q_v)_i ,           |exact_v - s_v| <= ||x||_2 * r_v ,  r_v = ||e_v - deq(q_v)||_2
// (Cauchy-Schwarz, r_v stored per row at quantisation time). Let L = max_u (s_u - ||x|| r_u). Every row with
// s_v + ||x|| r_v < L is provably not the argmax, so only the remaining candidates are rescored exactly from the bf16 rows.
// The returned argmax therefore equals the argmax of the full bf16 head (up to float summation order) -- the compression is
// lossless for greedy decoding, and the bytes read per token drop from 2 B/weight to ~1 (int8) or ~0.56 (int4) plus the
// candidate rows. For sampling the returned logits are exact for the candidates and approximate elsewhere.
#include <cstddef>
#include <cstdint>
#include <vector>
#include "ternix/threadpool.hpp"
#include "ternix/util.hpp"

namespace ternix {

enum class HeadMode { Bf16 = 0, Int8 = 1, Int4 = 2 };

struct HeadStats {
    size_t calls = 0, candidates = 0;       // candidates summed over calls
    size_t max_candidates = 0;
};

class OutputHead {
public:
    // bf16: V x H row-major bf16 (not copied for Bf16 mode; always used for exact rescoring).
    OutputHead(const uint16_t* rows, size_t V, size_t H, HeadMode mode, ThreadPool& pool, bool f16 = false);

    HeadMode mode() const { return mode_; }
    // Scales the certified bound ||x|| r_v. 1.0 (default) is the Cauchy-Schwarz worst case: provably exact greedy output.
    // A value of k/sqrt(H) is the k-sigma statistical bound (errors modelled as independent per element): far fewer
    // candidates, no proof; the empirical miss rate has to be measured (see headeval).
    void set_bound_scale(double s) { bound_scale_ = s; }
    // Approximate logits only (no rescoring), plus the per-row error scale ||x|| r_v / sqrt(H) in sigma (optional).
    void approx_logits(const float* x, float* out, float* sigma);
    // logits[v] for one hidden vector x (H floats): exact for every row that could be the argmax, approximate for the rest
    // (all rows exact in Bf16 mode). Returns the argmax row.
    int32_t logits(const float* x, float* out);
    // Bytes read per call: compressed matrix + per-row side data (+ the rescored rows, from stats()).
    size_t compressed_bytes() const;
    const HeadStats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }
    double mean_candidates() const { return stats_.calls ? double(stats_.candidates) / double(stats_.calls) : 0.0; }

private:
    float exact(const float* x, size_t v) const;
    void score_range(const float* x, size_t v0, size_t v1, float* s) const;

    const uint16_t* bf16_;      // the reference rows (bf16, or fp16 when f16_)
    bool f16_ = false;
    size_t V_, H_;
    HeadMode mode_;
    ThreadPool& pool_;
    AlignedBuf<int8_t> q8_;          // V x H (int8)
    AlignedBuf<uint8_t> q4_;         // V x H/2 (two nibbles per byte, offset 8), group of 32 = 16 bytes
    AlignedBuf<uint16_t> gscale_;    // int4: V x H/32 fp16 group scales
    AlignedBuf<float> rscale_;       // int8: per-row scale
    AlignedBuf<float> resid_;        // per-row residual L2 norm
    HeadStats stats_;
    double bound_scale_ = 1.0;
};

} // namespace ternix
