// ternix: command-line front end (CPU engine, and the CUDA engine with --gpu).
//   ternix generate  <model.ternix> --prompt "..." [--n 64] [--temp 0] [--top-k N] [--top-p P] [--seed S] [--chat]
//   ternix chat      <model.ternix>                 interactive, BitNet chat template, streaming output
//   ternix bench     <model.ternix> [--pp 128] [--tg 64] [--reps 3]    prefill and decode throughput
//   ternix ppl       <model.ternix> <text-file> [--ppl-ctx 512] [--max-tokens N]   perplexity over a text file
//   ternix logits    <model.ternix> (--prompt "..." | --ids "1 2 3") --out file.f32    last-token logits
//   ternix tokenize  <model.ternix> --prompt "..."                      token ids
// Common: --threads T  --ctx C  --isa scalar|avx2|avx_vnni  --gpu  --head bf16|int8|int4 (CPU: certified-exact quantised output head)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <sstream>
#include <string>
#include <vector>
#include "ternix/cpu_features.hpp"
#include "ternix/format.hpp"
#include "ternix/model.hpp"
#include "ternix/sampler.hpp"
#include "ternix/tokenizer.hpp"
#ifdef TERNIX_HAS_CUDA
#include "ternix/cuda_engine.hpp"
#endif

using namespace ternix;
using clk = std::chrono::steady_clock;

namespace {

struct Args {
    std::string cmd, model, prompt, out, file, ids_arg, configs = "bf16,int8,int8+sparse";
    int n = 64, threads = 0, ctx = 4096, pp = 128, tg = 64, reps = 3, ppl_ctx = 512, max_tokens = 0;
    bool chat = false, prompt_set = false, special = true, gpu = false;
    HeadMode head = HeadMode::Bf16;
    bool sparse_down = false, no_graph = false, kv_int8 = false, no_tc = false;
    double head_k = 0;   // headeval: k-sigma statistical bound for the int4 head (0 = certified)
    SamplerConfig sampler;
    Isa isa = best_isa();
};

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: ternix <generate|chat|bench|ppl|logits|tokenize> <model.ternix> [options]\n");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    if (argc < 3) usage();
    Args a;
    a.cmd = argv[1];
    a.model = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (s == "--prompt") { a.prompt = next(); a.prompt_set = true; }
        else if (s == "--n") a.n = std::atoi(next());
        else if (s == "--temp") a.sampler.temperature = static_cast<float>(std::atof(next()));
        else if (s == "--top-k") a.sampler.top_k = std::atoi(next());
        else if (s == "--top-p") a.sampler.top_p = static_cast<float>(std::atof(next()));
        else if (s == "--min-p") a.sampler.min_p = static_cast<float>(std::atof(next()));
        else if (s == "--repeat-penalty") a.sampler.repeat_penalty = static_cast<float>(std::atof(next()));
        else if (s == "--seed") a.sampler.seed = static_cast<uint64_t>(std::atoll(next()));
        else if (s == "--threads") a.threads = std::atoi(next());
        else if (s == "--ctx") a.ctx = std::atoi(next());
        else if (s == "--pp") a.pp = std::atoi(next());
        else if (s == "--tg") a.tg = std::atoi(next());
        else if (s == "--reps") a.reps = std::atoi(next());
        else if (s == "--max-tokens") a.max_tokens = std::atoi(next());
        else if (s == "--ppl-ctx") a.ppl_ctx = std::atoi(next());
        else if (s == "--out") a.out = next();
        else if (s == "--ids") a.ids_arg = next();
        else if (s == "--chat") a.chat = true;
        else if (s == "--gpu") a.gpu = true;
        else if (s == "--configs") a.configs = next();
        else if (s == "--no-graph") a.no_graph = true;
        else if (s == "--no-tc") a.no_tc = true;
        else if (s == "--kv-int8") a.kv_int8 = true;
        else if (s == "--sparse-down") a.sparse_down = true;
        else if (s == "--head-k") a.head_k = std::atof(next());
        else if (s == "--head") {
            const std::string v = next();
            a.head = v == "int8" ? HeadMode::Int8 : (v == "int4" ? HeadMode::Int4 : HeadMode::Bf16);
        }
        else if (s == "--no-special") a.special = false;
        else if (s == "--isa") {
            const std::string v = next();
            a.isa = v == "scalar" ? Isa::Scalar : (v == "avx2" ? Isa::Avx2 : Isa::AvxVnni);
        } else if ((a.cmd == "ppl" || a.cmd == "headeval") && a.file.empty()) a.file = s;
        else usage();
    }
    if ((a.cmd == "ppl" || a.cmd == "headeval") && a.file.empty()) usage();
    return a;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
    return std::string(std::istreambuf_iterator<char>(f), {});
}

// Prints UTF-8 text token by token without splitting a multi-byte character across writes.
struct Streamer {
    std::string pending;
    void feed(const std::string& piece) {
        pending += piece;
        size_t cut = pending.size();
        for (size_t back = 1; back <= 3 && back <= pending.size(); ++back) {
            const unsigned char c = static_cast<unsigned char>(pending[pending.size() - back]);
            if ((c & 0xC0) == 0x80) continue;
            const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if (need > back) cut = pending.size() - back;
            break;
        }
        std::fwrite(pending.data(), 1, cut, stdout);
        std::fflush(stdout);
        pending.erase(0, cut);
    }
};

std::string chat_prompt(const std::string& user) { return "User: " + user + "<|eot_id|>Assistant: "; }

std::string backend_desc(Model& m) { return std::string("cpu isa=") + isa_name(m.isa()) + " threads=" + std::to_string(m.pool().size()); }
#ifdef TERNIX_HAS_CUDA
std::string backend_desc(CudaModel&) { return "cuda"; }
#endif

bool is_stop(const ModelConfig& cfg, int32_t t) { return t == static_cast<int32_t>(cfg.eos) || t == static_cast<int32_t>(cfg.eot) || t == 128001; }

template <class M>
int generate(M& model, const Tokenizer& tok, const std::string& text, const Args& a) {
    const ModelConfig& cfg = model.config();
    std::vector<int32_t> ids = tok.encode(text, true, a.special);
    if (static_cast<int>(ids.size()) + a.n > model.max_ctx()) {
        std::fprintf(stderr, "prompt (%zu) + n (%d) exceeds context %d\n", ids.size(), a.n, model.max_ctx());
        return 1;
    }
    std::vector<float> logits(cfg.vocab);
    Sampler sampler(a.sampler);
    Streamer out;
    std::vector<int32_t> history = ids;
    auto t0 = clk::now();
    model.forward(ids.data(), static_cast<int>(ids.size()), logits.data(), 1);
    const double pp_s = std::chrono::duration<double>(clk::now() - t0).count();
    auto t1 = clk::now();
    int produced = 0;
    for (; produced < a.n; ++produced) {
        const int32_t next = sampler.sample(logits.data(), cfg.vocab, history);
        if (is_stop(cfg, next)) break;
        history.push_back(next);
        out.feed(tok.token_bytes(next));
        if (!model.forward(&next, 1, logits.data(), 1)) break;
    }
    const double tg_s = std::chrono::duration<double>(clk::now() - t1).count();
    std::printf("\n");
    std::fprintf(stderr, "[prefill %zu tok: %.1f tok/s | decode %d tok: %.2f tok/s | %s]\n", ids.size(), ids.size() / pp_s, produced,
                 produced / std::max(tg_s, 1e-9), backend_desc(model).c_str());
    return 0;
}

template <class M>
int run(M& model, const Tokenizer& tok, const Args& a) {
    const ModelConfig& cfg = model.config();
    if (a.cmd == "generate") {
        if (!a.prompt_set) usage();
        return generate(model, tok, a.chat ? chat_prompt(a.prompt) : a.prompt, a);
    }
    if (a.cmd == "chat") {
        std::fprintf(stderr, "BitNet chat (%s), multi-turn with a shared KV cache. Empty line or Ctrl-D quits.\n", backend_desc(model).c_str());
        SamplerConfig sc = a.sampler;
        if (sc.temperature == 0.f) { sc.temperature = 0.6f; sc.top_p = 0.9f; }
        Sampler sampler(sc);
        std::vector<float> logits(cfg.vocab);
        std::vector<int32_t> history;
        model.reset();
        for (std::string line;;) {
            std::fprintf(stderr, "\n> ");
            if (!std::getline(std::cin, line) || line.empty()) break;
            std::vector<int32_t> turn = tok.encode(chat_prompt(line), history.empty(), true);
            if (static_cast<int>(history.size() + turn.size()) + 16 > model.max_ctx()) {
                std::fprintf(stderr, "context full (%d tokens); starting a new conversation\n", model.max_ctx());
                history.clear();
                model.reset();
                turn = tok.encode(chat_prompt(line), true, true);
            }
            history.insert(history.end(), turn.begin(), turn.end());
            model.forward(turn.data(), static_cast<int>(turn.size()), logits.data(), 1);
            Streamer out;
            for (int i = 0; i < 512 && static_cast<int>(history.size()) + 1 < model.max_ctx(); ++i) {
                const int32_t next = sampler.sample(logits.data(), cfg.vocab, history);
                if (is_stop(cfg, next)) {
                    history.push_back(static_cast<int32_t>(cfg.eot));   // the turn ends with <|eot_id|> in the context
                    model.forward(&history.back(), 1, logits.data(), 0);
                    break;
                }
                history.push_back(next);
                out.feed(tok.token_bytes(next));
                model.forward(&next, 1, logits.data(), 1);
            }
            std::printf("\n");
        }
        return 0;
    }
    if (a.cmd == "logits") {
        std::vector<int32_t> ids;
        if (!a.ids_arg.empty()) {
            std::istringstream is(a.ids_arg);
            for (std::string t; std::getline(is, t, ' ');)
                if (!t.empty()) ids.push_back(std::atoi(t.c_str()));
        } else {
            ids = tok.encode(a.prompt, true, a.special);
        }
        std::vector<float> logits(cfg.vocab);
        model.forward(ids.data(), static_cast<int>(ids.size()), logits.data(), 1);
        std::ofstream(a.out, std::ios::binary).write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * 4));
        std::printf("%zu prompt tokens, top-1 id %d\n", ids.size(), Sampler::argmax(logits.data(), cfg.vocab));
        return 0;
    }
    if (a.cmd == "bench") {
        std::vector<int32_t> ids(static_cast<size_t>(a.pp));
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(1000 + (i * 7919) % 50000);
        std::vector<float> logits(cfg.vocab);
        std::printf("model %s | %s | cpu: %s\n", a.model.c_str(), backend_desc(model).c_str(), cpu_info().brand.c_str());
        std::vector<double> pp, tg;
        for (int r = -1; r < a.reps; ++r) {          // r = -1 is a warm-up
            model.reset();
            auto t0 = clk::now();
            model.forward(ids.data(), a.pp, logits.data(), 1);
            const double ps = std::chrono::duration<double>(clk::now() - t0).count();
            auto t1 = clk::now();
            for (int i = 0; i < a.tg; ++i) {
                const int32_t next = Sampler::argmax(logits.data(), cfg.vocab);
                model.forward(&next, 1, logits.data(), 1);
            }
            const double ts = std::chrono::duration<double>(clk::now() - t1).count();
            if (r >= 0) { pp.push_back(a.pp / ps); tg.push_back(a.tg / ts); }
        }
        std::sort(pp.begin(), pp.end());
        std::sort(tg.begin(), tg.end());
        std::printf("prefill %d tok: median %.1f tok/s (min %.1f max %.1f)\n", a.pp, pp[pp.size() / 2], pp.front(), pp.back());
        std::printf("decode  %d tok: median %.2f tok/s (min %.2f max %.2f)\n", a.tg, tg[tg.size() / 2], tg.front(), tg.back());
        if constexpr (std::is_same_v<M, Model>) {
            if (model.sparse_down()) std::printf("down-projection nonzero fraction (batch-1 calls): %.3f\n", model.mean_down_nnz_fraction());
            if (model.head() != nullptr) std::printf("head %s: mean candidates %.1f\n", model.head_mode() == HeadMode::Int8 ? "int8" : "int4", model.head()->mean_candidates());
        }
        return 0;
    }
    if constexpr (std::is_same_v<M, Model>) {
        if (a.cmd == "headeval") {
            // Compare the quantised heads with the bf16 head on the hidden vectors of real text (EXP-003).
            const std::string text = read_file(a.file);
            std::vector<int32_t> ids = tok.encode(text, false, false);
            if (a.max_tokens > 0 && static_cast<int>(ids.size()) > a.max_tokens) ids.resize(static_cast<size_t>(a.max_tokens));
            const size_t H = cfg.hidden, V = cfg.vocab;
            OutputHead full(model.embedding(), V, H, HeadMode::Bf16, model.pool());
            OutputHead h8(model.embedding(), V, H, HeadMode::Int8, model.pool());
            OutputHead h4(model.embedding(), V, H, HeadMode::Int4, model.pool());
            struct Acc { size_t arg = 0, top5 = 0, n = 0; double kl = 0, kl_cand = 0, cand = 0; size_t maxc = 0; };
            if (a.head_k > 0) h4.set_bound_scale(a.head_k / std::sqrt(static_cast<double>(H)));
            Acc acc8, acc4;
            std::vector<float> sg(V);
            double zmax8 = 0, zmax4 = 0;
            std::vector<double> z4all;
            std::vector<float> lf(V), la(V), hidden(static_cast<size_t>(Model::kMaxChunk) * H);
            const size_t W = static_cast<size_t>(std::min(a.ctx, a.ppl_ctx));
            auto softmax = [&](const std::vector<float>& l, std::vector<double>& p) {
                float mx = *std::max_element(l.begin(), l.end());
                double z = 0;
                for (size_t v = 0; v < V; ++v) { p[v] = std::exp(static_cast<double>(l[v] - mx)); z += p[v]; }
                for (auto& x : p) x /= z;
            };
            std::vector<double> pf(V), pa(V);
            for (size_t s = 0; s < ids.size(); s += W - 1) {
                std::vector<int32_t> win{static_cast<int32_t>(cfg.bos)};
                win.insert(win.end(), ids.begin() + static_cast<long>(s), ids.begin() + static_cast<long>(std::min(ids.size(), s + W - 1)));
                model.reset();
                for (size_t off = 0; off < win.size(); off += Model::kMaxChunk) {
                    const int n = static_cast<int>(std::min<size_t>(Model::kMaxChunk, win.size() - off));
                    model.forward(win.data() + off, n, hidden.data(), 3);
                    for (int t = 0; t < n; ++t) {
                        const float* x = hidden.data() + static_cast<size_t>(t) * H;
                        const int32_t best = full.logits(x, lf.data());
                        softmax(lf, pf);
                        std::vector<int32_t> top5(V);
                        std::iota(top5.begin(), top5.end(), 0);
                        std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(), [&](int32_t p, int32_t q) { return lf[p] > lf[q]; });
                        for (auto* hd2 : {&h8, &h4}) {   // error-to-sigma ratio of the approximations against the exact logits
                            hd2->approx_logits(x, la.data(), sg.data());
                            double zm = 0;
                            for (size_t v = 0; v < V; ++v) zm = std::max(zm, std::fabs(static_cast<double>(lf[v] - la[v])) / std::max(1e-12, static_cast<double>(sg[v])));
                            (hd2 == &h8 ? zmax8 : zmax4) = std::max(hd2 == &h8 ? zmax8 : zmax4, zm);
                            if (hd2 == &h4) z4all.push_back(zm);
                        }
                        for (auto pr : {std::make_pair(&h8, &acc8), std::make_pair(&h4, &acc4)}) {
                            OutputHead& hd = *pr.first;
                            Acc& ac = *pr.second;
                            const size_t before = hd.stats().candidates;
                            const int32_t b2 = hd.logits(x, la.data());
                            const size_t ncand = hd.stats().candidates - before;
                            ac.n++; ac.arg += (b2 == best); ac.cand += static_cast<double>(ncand); ac.maxc = std::max(ac.maxc, ncand);
                            std::vector<int32_t> t5(V);
                            std::iota(t5.begin(), t5.end(), 0);
                            std::partial_sort(t5.begin(), t5.begin() + 5, t5.end(), [&](int32_t p, int32_t q) { return la[p] > la[q]; });
                            size_t same = 0;
                            for (int i = 0; i < 5; ++i) same += std::find(top5.begin(), top5.begin() + 5, t5[i]) != top5.begin() + 5;
                            ac.top5 += (same == 5);
                            softmax(la, pa);
                            double kl = 0;
                            for (size_t v = 0; v < V; ++v) if (pf[v] > 0) kl += pf[v] * (std::log(pf[v]) - std::log(std::max(pa[v], 1e-300)));
                            ac.kl += kl;
                        }
                    }
                }
            }
            for (auto& [name, ac, hd] : {std::tuple<const char*, Acc&, OutputHead&>{"int8", acc8, h8}, {"int4-g32", acc4, h4}})
                std::printf("%-9s positions %zu | argmax agree %zu (%.4f%%) | top5-set agree %.3f%% | mean KL(full||head) %.3e | candidates mean %.1f max %zu | head bytes %.1f MB (+%.2f MB rescoring) vs bf16 %.1f MB\n",
                            name, ac.n, ac.arg, 100.0 * ac.arg / ac.n, 100.0 * ac.top5 / ac.n, ac.kl / ac.n, ac.cand / ac.n, ac.maxc,
                            hd.compressed_bytes() / 1e6, ac.cand / ac.n * H * 2 / 1e6, V * H * 2 / 1e6);
            std::sort(z4all.begin(), z4all.end());
            std::printf("max |exact-approx|/sigma over all rows and positions: int8 %.2f, int4 %.2f (per-position max: median %.2f, p99 %.2f)\n", zmax8, zmax4,
                        z4all[z4all.size() / 2], z4all[z4all.size() * 99 / 100]);
            return 0;
        }
    }
    if constexpr (std::is_same_v<M, Model>) {
        if (a.cmd == "abtest") {
            // Interleaved A/B decode benchmark: every repetition runs every configuration once (shuffled order), so drift of the
            // machine state (clocks, thermals, background load) hits all configurations alike. Reports medians and paired ratios.
            std::vector<std::string> cfgs;
            for (size_t p = 0, q; p <= a.configs.size(); p = q + 1) {
                q = a.configs.find(',', p);
                if (q == std::string::npos) q = a.configs.size();
                if (q > p) cfgs.push_back(a.configs.substr(p, q - p));
            }
            struct Cfg { HeadMode head; bool sparse; bool kv8; };
            std::vector<Cfg> cv;
            for (const std::string& c : cfgs) {
                Cfg x{c.rfind("int8", 0) == 0 ? HeadMode::Int8 : (c.rfind("int4", 0) == 0 ? HeadMode::Int4 : HeadMode::Bf16), c.find("+sparse") != std::string::npos, c.find("+kv8") != std::string::npos};
                cv.push_back(x);
                model.set_head(x.head);
                if (x.sparse) model.set_sparse_down(true);
            }
            std::vector<int32_t> prompt(static_cast<size_t>(a.pp));
            for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<int32_t>(1000 + (i * 7919) % 50000);
            std::vector<float> logits(cfg.vocab);
            std::vector<std::vector<double>> res(cv.size());
            std::mt19937 rng(12345);
            std::vector<size_t> order(cv.size());
            std::iota(order.begin(), order.end(), 0);
            for (int r = -1; r < a.reps; ++r) {
                std::shuffle(order.begin(), order.end(), rng);
                for (size_t ci : order) {
                    model.set_head(cv[ci].head);
                    model.set_sparse_down(cv[ci].sparse);
                    model.set_kv_int8(cv[ci].kv8);
                    model.reset();
                    model.forward(prompt.data(), a.pp, logits.data(), 1);
                    auto t1 = clk::now();
                    for (int i = 0; i < a.tg; ++i) {
                        const int32_t next = Sampler::argmax(logits.data(), cfg.vocab);
                        model.forward(&next, 1, logits.data(), 1);
                    }
                    const double ts = std::chrono::duration<double>(clk::now() - t1).count();
                    if (r >= 0) res[ci].push_back(a.tg / ts);
                }
            }
            std::printf("A/B decode, %d tokens after a %d-token prompt, %d interleaved repetitions, %d threads\n", a.tg, a.pp, a.reps, model.pool().size());
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const double base = med(res[0]);
            for (size_t ci = 0; ci < cv.size(); ++ci) {
                std::vector<double> ratio;
                for (size_t r = 0; r < res[ci].size(); ++r) ratio.push_back(res[ci][r] / res[0][r]);
                auto sorted = res[ci];
                std::sort(sorted.begin(), sorted.end());
                std::sort(ratio.begin(), ratio.end());
                std::printf("  %-18s median %6.2f tok/s  (min %6.2f max %6.2f)  vs first: x%.3f (paired median x%.3f, p10 x%.3f, p90 x%.3f)\n", cfgs[ci].c_str(), med(res[ci]),
                            sorted.front(), sorted.back(), med(res[ci]) / base, med(ratio), ratio[ratio.size() / 10], ratio[ratio.size() * 9 / 10]);
            }
            return 0;
        }
    }
    if (a.cmd == "ppl") {
        const std::string text = read_file(a.file);
        std::vector<int32_t> ids = tok.encode(text, false, false);
        if (a.max_tokens > 0 && static_cast<int>(ids.size()) > a.max_tokens) ids.resize(static_cast<size_t>(a.max_tokens));
        const size_t W = static_cast<size_t>(std::min(a.ctx, a.ppl_ctx));
        double nll = 0;
        size_t count = 0;
        auto t0 = clk::now();
        // non-overlapping windows: [BOS] + up to W-1 text tokens, context reset per window, every text token scored once
        for (size_t s = 0; s < ids.size(); s += W - 1) {
            std::vector<int32_t> win{static_cast<int32_t>(cfg.bos)};
            win.insert(win.end(), ids.begin() + static_cast<long>(s), ids.begin() + static_cast<long>(std::min(ids.size(), s + W - 1)));
            if (win.size() < 2) break;
            const int n = static_cast<int>(win.size()) - 1;
            std::vector<float> per(static_cast<size_t>(n));
            model.reset();
            model.forward_nll(win.data(), n, win.data() + 1, per.data());
            for (float v : per) nll += v;
            count += per.size();
        }
        const double secs = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("tokens scored %zu | mean NLL %.5f | perplexity %.4f | %.1f s\n", count, nll / static_cast<double>(count),
                    std::exp(nll / static_cast<double>(count)), secs);
        return 0;
    }
    usage();
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        ModelFile file(a.model);
        Tokenizer tok(file.tokenizer_data(), file.tokenizer_size());
        if (a.cmd == "tokenize") {
            for (int32_t id : tok.encode(a.prompt, true, a.special)) std::printf("%d ", id);
            std::printf("\n");
            return 0;
        }
        if (a.gpu) {
#ifdef TERNIX_HAS_CUDA
            CudaModel model(file, a.ctx);
            model.set_head_int8(a.head == HeadMode::Int8);
            if (a.no_graph) model.set_graph(false);
            if (a.no_tc) model.set_tensor_cores(false);
            return run(model, tok, a);
#else
            std::fprintf(stderr, "this build has no CUDA support\n");
            return 2;
#endif
        }
        file.prefault();
        Model model(file, a.threads, a.ctx, a.isa);
        model.set_head(a.head);
        if (a.sparse_down) model.set_sparse_down(true);
        if (a.kv_int8) model.set_kv_int8(true);
        return run(model, tok, a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
