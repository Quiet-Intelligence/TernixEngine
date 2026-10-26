// Sequential read bandwidth versus thread count: the roofline for memory-bound decode on this machine.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>
#include "ternix/threadpool.hpp"

using namespace ternix;
using clk = std::chrono::steady_clock;

static int read_block(const uint8_t* p, size_t n) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        a0 = _mm256_add_epi8(a0, _mm256_loadu_si256((const __m256i*)(p + i)));
        a1 = _mm256_add_epi8(a1, _mm256_loadu_si256((const __m256i*)(p + i + 32)));
        a2 = _mm256_add_epi8(a2, _mm256_loadu_si256((const __m256i*)(p + i + 64)));
        a3 = _mm256_add_epi8(a3, _mm256_loadu_si256((const __m256i*)(p + i + 96)));
    }
    a0 = _mm256_add_epi8(_mm256_add_epi8(a0, a1), _mm256_add_epi8(a2, a3));
    alignas(32) uint8_t t[32];
    _mm256_store_si256((__m256i*)t, a0);
    return t[0] + t[9] + t[31];
}

int main() {
    const size_t bytes = 1ull << 30;
    std::vector<uint8_t> buf(bytes, 1);
    std::printf("sequential read bandwidth, %zu MiB buffer, 4 MiB chunks\n", bytes >> 20);
    for (int threads : {1, 2, 4, 8, 12, 16, 20, 24}) {
        ThreadPool pool(threads, true);
        const size_t chunk = 4u << 20, nch = bytes / chunk;
        std::vector<int> part(pool.size() * 16);
        double best = 0;
        for (int rep = 0; rep < 5; ++rep) {
            auto a = clk::now();
            pool.parallel_for(nch, [&](size_t c, int tid) { part[tid * 16] += read_block(buf.data() + c * chunk, chunk); });
            double s = std::chrono::duration<double>(clk::now() - a).count();
            best = std::max(best, bytes / s / 1e9);
        }
        std::printf("  threads %2d: %6.2f GB/s\n", pool.size(), best);
    }
    return 0;
}
