// EXP-007 prototype: zero-bitmap + compacted-sign ternary layout (BITCOS-style, 2 - z bits per weight) decoded with BMI2 pdep and
// AVX2, against the I2K128 kernel, on the four decode shapes of BitNet 2B4T with weights rotating beyond the LLC.
// Row layout: K/8 bytes of presence bitmap, then the sign bits of the nonzero weights packed back to back (1 = negative).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <random>
#include <vector>
#include "ternix/i2k.hpp"
#include "ternix/threadpool.hpp"

using namespace ternix;
using clk = std::chrono::steady_clock;

struct BitmapMatrix {
    size_t N = 0, K = 0;
    std::vector<uint8_t> data;       // rows back to back
    std::vector<size_t> row_off;     // byte offset of each row; row n spans [row_off[n], row_off[n + 1])
    double zero_frac = 0, bits_per_weight = 0;
};

static BitmapMatrix pack_bitmap(const std::vector<int8_t>& w, size_t N, size_t K) {
    BitmapMatrix m;
    m.N = N; m.K = K;
    m.row_off.push_back(0);
    size_t zeros = 0;
    for (size_t n = 0; n < N; ++n) {
        std::vector<uint8_t> bm(K / 8, 0), sg;
        size_t nz = 0;
        std::vector<uint8_t> bits;
        for (size_t k = 0; k < K; ++k) {
            const int8_t v = w[n * K + k];
            if (v == 0) { ++zeros; continue; }
            bm[k / 8] |= static_cast<uint8_t>(1u << (k % 8));
            bits.push_back(v < 0 ? 1 : 0);
            ++nz;
        }
        sg.assign((nz + 7) / 8 + 16, 0);                       // + slack so unaligned 128-bit reads stay inside the row
        for (size_t i = 0; i < nz; ++i) sg[i / 8] |= static_cast<uint8_t>(bits[i] << (i % 8));
        m.data.insert(m.data.end(), bm.begin(), bm.end());
        m.data.insert(m.data.end(), sg.begin(), sg.end());
        m.row_off.push_back(m.data.size());
    }
    m.zero_frac = static_cast<double>(zeros) / (static_cast<double>(N) * K);
    m.bits_per_weight = 8.0 * static_cast<double>(m.data.size()) / (static_cast<double>(N) * K);
    return m;
}

// 32-bit mask -> 32 byte lanes of 0 / 1
static inline __m256i expand_mask(uint32_t m, const __m256i shuf, const __m256i sel) {
    const __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(m)), shuf);
    return _mm256_min_epu8(_mm256_and_si256(b, sel), _mm256_set1_epi8(1));
}

// y = sum_k q[k] w[k] for one row; weights as codes c = 1 + pos - neg in {0,1,2}, so vpdpbusd applies and sum_q is subtracted
static int32_t dot_row(const uint8_t* row, size_t K, const int8_t* q, int32_t sum_q) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(128), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(128), 1, 2, 4, 8, 16, 32, 64,
                                         static_cast<char>(128), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(128));
    const __m256i one = _mm256_set1_epi8(1);
    const uint8_t* bm = row;
    const uint8_t* sg = row + K / 8;
    __m256i acc = _mm256_setzero_si256();
    size_t bitpos = 0;
    for (size_t b = 0; b < K / 64; ++b) {
        uint64_t m;
        std::memcpy(&m, bm + 8 * b, 8);
        const unsigned pc = static_cast<unsigned>(_mm_popcnt_u64(m));
        // compacted signs of this block start at bit `bitpos`: take 64 bits from an unaligned 128-bit window
        __m128i win = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sg + (bitpos >> 3)));
        const unsigned sh = static_cast<unsigned>(bitpos & 7);
        uint64_t lo = static_cast<uint64_t>(_mm_cvtsi128_si64(win)), hi = static_cast<uint64_t>(_mm_extract_epi64(win, 1));
        uint64_t s = sh ? (lo >> sh) | (hi << (64 - sh)) : lo;
        if (pc < 64) s &= (1ull << pc) - 1;
        const uint64_t neg = _pdep_u64(s, m);                  // sign bit placed at each nonzero position
        const uint64_t pos = m & ~neg;
        bitpos += pc;
        for (int h = 0; h < 2; ++h) {
            const uint32_t pm = static_cast<uint32_t>(pos >> (32 * h)), nm = static_cast<uint32_t>(neg >> (32 * h));
            __m256i c = _mm256_add_epi8(one, expand_mask(pm, shuf, sel));
            c = _mm256_sub_epi8(c, expand_mask(nm, shuf, sel));
            acc = _mm256_dpbusd_avx_epi32(acc, c, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 64 * b + 32 * h)));
        }
    }
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    return _mm_cvtsi128_si32(s4) - sum_q;
}

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
    const double zero_target = argc > 1 ? std::atof(argv[1]) : 0.42;
    ThreadPool pool(0);
    struct Shape { const char* name; size_t N, K; };
    const Shape shapes[] = {{"qo 2560x2560", 2560, 2560}, {"gateup 6912x2560", 6912, 2560}, {"down 2560x6912", 2560, 6912}};
    std::printf("zero density %.2f | pool threads %d\n", zero_target, pool.size());
    for (const Shape& sh : shapes) {
        const size_t N = sh.N, K = sh.K;
        std::mt19937 rng(3);
        std::vector<int8_t> w(N * K);
        for (auto& x : w) x = (static_cast<double>(rng() % 10000) / 10000.0 < zero_target) ? 0 : ((rng() & 1) ? 1 : -1);
        BitmapMatrix bmx = pack_bitmap(w, N, K);
        const size_t i2k_bytes = N * K / 4;
        const size_t copies = std::max<size_t>(3, (320u << 20) / std::max(i2k_bytes, bmx.data.size()));
        std::vector<std::unique_ptr<uint8_t[]>> wi(copies), wb(copies);
        std::vector<uint8_t> packed(i2k_bytes);
        i2k::pack(w.data(), packed.data(), N, K);
        for (size_t c = 0; c < copies; ++c) {
            wi[c].reset(new uint8_t[i2k_bytes]); std::memcpy(wi[c].get(), packed.data(), i2k_bytes);
            wb[c].reset(new uint8_t[bmx.data.size()]); std::memcpy(wb[c].get(), bmx.data.data(), bmx.data.size());
        }
        std::vector<int8_t> q(K);
        for (auto& v : q) v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
        int32_t sq = 0;
        for (int8_t v : q) sq += v;
        std::vector<int32_t> y(N), y_ref(N);
        i2k::gemv(wi[0].get(), q.data(), sq, y_ref.data(), 0, N, K, Isa::Scalar);
        for (size_t n = 0; n < N; ++n) y[n] = dot_row(bmx.data.data() + bmx.row_off[n], K, q.data(), sq);
        if (y != y_ref) { std::printf("MISMATCH %s\n", sh.name); return 1; }
        const Isa isa = best_isa();
        constexpr size_t chunk = 32;
        const size_t nch = (N + chunk - 1) / chunk;
        auto run_i2k = [&](int i) {
            const uint8_t* W = wi[static_cast<size_t>(i) % copies].get();
            pool.parallel_for(nch, [&](size_t c, int) { const size_t n0 = c * chunk, n1 = std::min(N, n0 + chunk); i2k::gemv(W, q.data(), sq, y.data() + n0, n0, n1, K, isa); });
        };
        auto run_bm = [&](int i) {
            const uint8_t* W = wb[static_cast<size_t>(i) % copies].get();
            pool.parallel_for(nch, [&](size_t c, int) {
                for (size_t n = c * chunk; n < std::min(N, c * chunk + chunk); ++n) y[n] = dot_row(W + bmx.row_off[n], K, q.data(), sq);
            });
        };
        const double t_i2k = median_us(run_i2k, 15, static_cast<int>(copies)), t_bm = median_us(run_bm, 15, static_cast<int>(copies));
        // single thread, cache resident: pure instruction cost
        const size_t rows = 256;
        const double c_i2k = median_us([&](int) { i2k::gemv(wi[0].get(), q.data(), sq, y.data(), 0, rows, K, isa); }, 15, 20);
        const double c_bm = median_us([&](int) { for (size_t n = 0; n < rows; ++n) y[n] = dot_row(bmx.data.data() + bmx.row_off[n], K, q.data(), sq); }, 15, 20);
        std::printf("%-17s bitmap %.3f bpw (zeros %.3f) | pool DRAM-rotating: i2k %.1f us (%.1f GB/s) bitmap %.1f us (%.1f GB/s) ratio %.2f | 1-thread cache-resident GMAC/s: i2k %.1f bitmap %.1f\n",
                    sh.name, bmx.bits_per_weight, bmx.zero_frac, t_i2k, i2k_bytes / (t_i2k * 1e3), t_bm, bmx.data.size() / (t_bm * 1e3), t_bm / t_i2k,
                    double(rows) * K / (c_i2k * 1e3), double(rows) * K / (c_bm * 1e3));
    }
    return 0;
}
