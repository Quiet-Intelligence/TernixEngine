// Decode-shape GEMV benchmark: legacy K x N/4 kernels vs the I2K128 kernels, single thread and thread pool.
// Weights rotate through enough distinct matrices to exceed the last-level cache, so every call streams from DRAM the
// way a real decode step does (a single 4 MB matrix would stay resident in L2/L3 and report cache bandwidth).
//   bench_i2k [--reps N] [--out file.json] [--threads T]
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "ternix/bitpack.hpp"
#include "ternix/cpu_features.hpp"
#include "ternix/i2k.hpp"
#include "ternix/mul_mat_ternary.h"
#include "ternix/threadpool.hpp"

using namespace ternix;
using clk = std::chrono::steady_clock;

struct Shape { const char* name; size_t N, K; };
static const Shape kShapes[] = {{"qo_2560x2560", 2560, 2560}, {"kv_640x2560", 640, 2560}, {"gateup_6912x2560", 6912, 2560},
                                {"down_2560x6912", 2560, 6912}};

template <class F>
static double median_us(F&& f, int reps, int inner) {
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        auto a = clk::now();
        for (int i = 0; i < inner; ++i) f(i);
        t.push_back(std::chrono::duration<double, std::micro>(clk::now() - a).count() / inner);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    int reps = 15, threads = 0;
    const char* out_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }
    ThreadPool pool(threads);
    std::printf("cpu: %s | isa=%s | pool threads=%d\n", cpu_info().brand.c_str(), isa_name(best_isa()), pool.size());
    std::FILE* jf = out_path ? std::fopen(out_path, "w") : nullptr;
    if (jf) std::fprintf(jf, "{\"cpu\": \"%s\", \"pool_threads\": %d, \"results\": [\n", cpu_info().brand.c_str(), pool.size());
    bool first = true;

    for (const Shape& s : kShapes) {
        const size_t N = s.N, K = s.K, bytes = N * K / 4;
        const size_t copies = std::max<size_t>(4, (320u << 20) / bytes);
        std::mt19937 rng(1);
        std::vector<int8_t> w(N * K);
        for (auto& x : w) x = static_cast<int8_t>(static_cast<int>(rng() % 3) - 1);
        std::vector<int8_t> wt(K * N);                       // legacy layout needs W[K][N]
        for (size_t n = 0; n < N; ++n) for (size_t k = 0; k < K; ++k) wt[k * N + n] = w[n * K + k];
        std::vector<std::unique_ptr<uint8_t[]>> wi(copies), wl(copies);
        for (size_t c = 0; c < copies; ++c) {
            wi[c].reset(new uint8_t[bytes]); wl[c].reset(new uint8_t[bytes]);
            i2k::pack(w.data(), wi[c].get(), N, K);
            ternix::pack_ternary_weights(wl[c].get(), wt.data(), N * K);
            // touch with a perturbation-free rewrite so each copy is its own physical memory
            std::memcpy(wi[c].get(), wi[0].get(), bytes);
        }
        std::vector<float> x(K);
        for (auto& v : x) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 250.f;
        std::vector<int8_t> q(K);
        float deq; int32_t sq;
        i2k::quantize_act(x.data(), q.data(), K, &deq, &sq, Isa::Scalar);
        std::vector<int32_t> y(N), y_ref(N);
        i2k::gemv(wi[0].get(), q.data(), sq, y_ref.data(), 0, N, K, Isa::Scalar);

        struct Row { std::string name; double us; };
        std::vector<Row> rows;
        auto record = [&](const std::string& name, double us) { rows.push_back({name, us}); };

        record("legacy_avx2_1row (K x N/4)", median_us([&](int i) {
            mul_mat_ternary_avx2(q.data(), wl[i % copies].get(), y.data(), 1, K, N);
        }, reps, static_cast<int>(copies)));
        if (!(y == y_ref)) {  // legacy kernel adds no correction term: same product on ternary weights
            std::printf("  legacy kernel mismatch (expected: identical integers)\n");
        }
        for (Isa isa : {Isa::Scalar, Isa::Avx2, Isa::AvxVnni}) {
            if (isa == Isa::Avx2 && !cpu_info().avx2) continue;
            if (isa == Isa::AvxVnni && !cpu_info().avx_vnni) continue;
            record(std::string("i2k_1thread_") + isa_name(isa), median_us([&](int i) {
                i2k::gemv(wi[i % copies].get(), q.data(), sq, y.data(), 0, N, K, isa);
            }, reps, static_cast<int>(copies)));
            if (!(y == y_ref)) { std::printf("  MISMATCH %s\n", isa_name(isa)); return 1; }
        }
        const Isa best = best_isa();
        for (size_t chunk : {size_t(16), size_t(32), size_t(64)}) {
            const size_t nch = (N + chunk - 1) / chunk;
            record("i2k_pool_" + std::string(isa_name(best)) + "_chunk" + std::to_string(chunk), median_us([&](int i) {
                const uint8_t* W = wi[i % copies].get();
                pool.parallel_for(nch, [&](size_t c, int) {
                    const size_t n0 = c * chunk, n1 = std::min(N, n0 + chunk);
                    i2k::gemv(W, q.data(), sq, y.data() + n0, n0, n1, K, best);
                });
            }, reps, static_cast<int>(copies)));
        }
        // diagnostics: pure read bandwidth over the same rotating buffers, and the cache-resident (compute-bound) rate
        {
            auto read_sum = [&](const uint8_t* p, size_t n) {
                __m256i acc = _mm256_setzero_si256();
                for (size_t i = 0; i + 32 <= n; i += 32) acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i*)(p + i)));
                alignas(32) uint8_t t[32];
                _mm256_store_si256((__m256i*)t, acc);
                return (int)t[0] + t[7] + t[19];
            };
            volatile int sink = 0;
            record("read_only_1thread", median_us([&](int i) { sink = sink + read_sum(wi[i % copies].get(), bytes); }, reps, (int)copies));
            const size_t nt = (size_t)pool.size();
            std::vector<int> part(nt * 16);
            record("read_only_pool", median_us([&](int i) {
                const uint8_t* W = wi[i % copies].get();
                pool.parallel_for(nt * 4, [&](size_t c, int tid) { part[tid * 16] += read_sum(W + c * (bytes / (nt * 4)), bytes / (nt * 4)); });
            }, reps, (int)copies));
            record("i2k_1thread_cache_resident", median_us([&](int) {
                i2k::gemv(wi[0].get(), q.data(), sq, y.data(), 0, N, K, best_isa());
            }, reps, 8));
        }
        std::printf("\n%s  (weights %.2f MB, %zu rotating copies = %.0f MB)\n", s.name, bytes / 1e6, copies, copies * bytes / 1e6);
        for (const Row& r : rows)
            std::printf("  %-38s %9.1f us   %7.2f GB/s   %7.2f GMAC/s\n", r.name.c_str(), r.us, bytes / (r.us * 1e3),
                        double(N) * K / (r.us * 1e3));
        if (jf)
            for (const Row& r : rows) {
                std::fprintf(jf, "%s  {\"shape\": \"%s\", \"N\": %zu, \"K\": %zu, \"kernel\": \"%s\", \"median_us\": %.2f, \"gbps\": %.3f}",
                             first ? "" : ",\n", s.name, N, K, r.name.c_str(), r.us, bytes / (r.us * 1e3));
                first = false;
            }
    }
    if (jf) { std::fprintf(jf, "\n]}\n"); std::fclose(jf); }
    return 0;
}
