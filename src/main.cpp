// ternix: command-line front end (CPU engine, and the CUDA engine with --gpu).
//   ternix generate  <model.ternix> --prompt "..." [--n 64] [--temp 0] [--top-k N] [--top-p P] [--seed S] [--chat]
//   ternix chat      <model.ternix>                 interactive, BitNet chat template, streaming output
//   ternix bench     <model.ternix> [--pp 128] [--tg 64] [--reps 3]    prefill and decode throughput
//   ternix ppl       <model.ternix> <text-file> [--ppl-ctx 512] [--max-tokens N]   perplexity over a text file
//   ternix logits    <model.ternix> (--prompt "..." | --ids "1 2 3") --out file.f32    last-token logits
//   ternix tokenize  <model.ternix> --prompt "..."                      token ids
// Common: --threads T  --ctx C  --isa scalar|avx2|avx_vnni  --gpu
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
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
    std::string cmd, model, prompt, out, file, ids_arg;
    int n = 64, threads = 0, ctx = 4096, pp = 128, tg = 64, reps = 3, ppl_ctx = 512, max_tokens = 0;
    bool chat = false, prompt_set = false, special = true, gpu = false;
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
        else if (s == "--no-special") a.special = false;
        else if (s == "--isa") {
            const std::string v = next();
            a.isa = v == "scalar" ? Isa::Scalar : (v == "avx2" ? Isa::Avx2 : Isa::AvxVnni);
        } else if (a.cmd == "ppl" && a.file.empty()) a.file = s;
        else usage();
    }
    if (a.cmd == "ppl" && a.file.empty()) usage();
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
        return 0;
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
            return run(model, tok, a);
#else
            std::fprintf(stderr, "this build has no CUDA support\n");
            return 2;
#endif
        }
        file.prefault();
        Model model(file, a.threads, a.ctx, a.isa);
        return run(model, tok, a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
