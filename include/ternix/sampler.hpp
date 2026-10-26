#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace ternix {

struct SamplerConfig {
    float temperature = 0.f;      // <= 0: greedy
    int top_k = 0;                // 0: disabled
    float top_p = 1.f;
    float min_p = 0.f;
    float repeat_penalty = 1.f;   // > 1 penalises tokens in the recent window
    int repeat_window = 64;
    uint64_t seed = 1;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}
    // logits is modified in place (penalties, temperature). `history` is the token sequence so far.
    int32_t sample(float* logits, size_t vocab, const std::vector<int32_t>& history);
    static int32_t argmax(const float* logits, size_t vocab);

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
};

} // namespace ternix
