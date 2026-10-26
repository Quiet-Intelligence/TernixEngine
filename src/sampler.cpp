#include "ternix/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ternix {

int32_t Sampler::argmax(const float* logits, size_t vocab) {
    size_t best = 0;
    for (size_t i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
    return static_cast<int32_t>(best);
}

int32_t Sampler::sample(float* logits, size_t vocab, const std::vector<int32_t>& history) {
    if (cfg_.repeat_penalty != 1.f) {
        const size_t from = history.size() > static_cast<size_t>(cfg_.repeat_window) ? history.size() - cfg_.repeat_window : 0;
        for (size_t i = from; i < history.size(); ++i) {
            float& l = logits[history[i]];
            l = l > 0 ? l / cfg_.repeat_penalty : l * cfg_.repeat_penalty;
        }
    }
    if (cfg_.temperature <= 0.f) return argmax(logits, vocab);

    // candidate set: top_k (or everything) sorted by logit
    std::vector<int32_t> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    size_t keep = cfg_.top_k > 0 ? std::min<size_t>(static_cast<size_t>(cfg_.top_k), vocab) : vocab;
    // top_p / min_p need a descending order; when no top_k is set cap the sort at 4096 candidates
    if (cfg_.top_k <= 0 && (cfg_.top_p < 1.f || cfg_.min_p > 0.f)) keep = std::min<size_t>(vocab, 4096);
    if (keep < vocab) std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    else std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    idx.resize(keep);

    const float mx = logits[idx[0]];
    std::vector<double> p(keep);
    double sum = 0;
    for (size_t i = 0; i < keep; ++i) {
        p[i] = std::exp((logits[idx[i]] - mx) / cfg_.temperature);
        sum += p[i];
    }
    for (auto& v : p) v /= sum;
    size_t cut = keep;
    if (cfg_.min_p > 0.f) {
        const double thr = p[0] * cfg_.min_p;
        while (cut > 1 && p[cut - 1] < thr) --cut;
    }
    if (cfg_.top_p < 1.f) {
        double c = 0;
        size_t k = 0;
        while (k < cut) {
            c += p[k++];
            if (c >= cfg_.top_p) break;
        }
        cut = std::max<size_t>(1, k);
    }
    double tot = 0;
    for (size_t i = 0; i < cut; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng_), c = 0;
    for (size_t i = 0; i < cut; ++i) {
        c += p[i];
        if (r <= c) return idx[i];
    }
    return idx[cut - 1];
}

} // namespace ternix
