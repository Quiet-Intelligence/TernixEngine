#include "ternix/threadpool.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#include <immintrin.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace ternix {

std::vector<int> ThreadPool::performance_cores() {
    std::vector<int> cores;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return cores;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
        return cores;
    struct Core { BYTE eff; std::vector<int> lps; };
    std::vector<Core> all;
    BYTE max_eff = 0;
    for (size_t off = 0; off < len;) {
        auto* p = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) {
            Core c{p->Processor.EfficiencyClass, {}};
            for (WORD g = 0; g < p->Processor.GroupCount; ++g) {
                KAFFINITY m = p->Processor.GroupMask[g].Mask;
                for (int b = 0; b < 64; ++b)
                    if (m >> b & 1) c.lps.push_back(b);  // group 0 only is used for pinning below
            }
            max_eff = std::max(max_eff, c.eff);
            all.push_back(std::move(c));
        }
        off += p->Size;
    }
    for (auto& c : all)
        if (c.eff == max_eff)
            for (int lp : c.lps) cores.push_back(lp);
    std::sort(cores.begin(), cores.end());
#endif
    return cores;
}

static void pin_thread(int logical_cpu) {
#ifdef _WIN32
    if (logical_cpu >= 0 && logical_cpu < 64) SetThreadAffinityMask(GetCurrentThread(), 1ull << logical_cpu);
#else
    (void)logical_cpu;
#endif
}

ThreadPool::ThreadPool(int threads, bool pin) {
    // Experiment switches (EXP-005): TERNIX_PIN = p (performance cores, default) | all (every logical processor in order) | none;
    // TERNIX_SCHED = dynamic (default) | static (equal contiguous chunk ranges per thread).
    const char* pin_env = std::getenv("TERNIX_PIN");
    const std::string pin_mode = pin_env ? pin_env : "p";
    const char* sched_env = std::getenv("TERNIX_SCHED");
    static_sched_ = sched_env && std::string(sched_env) == "static";
    std::vector<int> pcores;
    size_t n_perf = 0;
    if (pin && pin_mode == "p") {
        pcores = performance_cores();
        n_perf = pcores.size();
        // spill beyond the performance cores onto the remaining logical processors instead of wrapping: two spinning workers pinned
        // to the same logical processor starve each other (EXP-005: 1.5 tok/s instead of 33)
        for (unsigned i = 0; i < std::thread::hardware_concurrency() && i < 64 && !pcores.empty(); ++i)
            if (std::find(pcores.begin(), pcores.begin() + static_cast<long>(n_perf), static_cast<int>(i)) == pcores.begin() + static_cast<long>(n_perf)) pcores.push_back(static_cast<int>(i));
    }
    else if (pin && pin_mode == "all")
        for (unsigned i = 0; i < std::thread::hardware_concurrency() && i < 64; ++i) pcores.push_back(static_cast<int>(i));
    if (threads <= 0) threads = pcores.empty() ? static_cast<int>(std::thread::hardware_concurrency()) : static_cast<int>(n_perf ? n_perf : pcores.size());
    nthreads_ = std::max(1, threads);
    if (!pcores.empty()) {
        pin_thread(pcores[0]);
    }
    for (int t = 1; t < nthreads_; ++t) {
        const int cpu = (!pcores.empty()) ? pcores[static_cast<size_t>(t) % pcores.size()] : -1;
        workers_.emplace_back([this, t, cpu] {
            if (cpu >= 0) pin_thread(cpu);
            worker(t);
        });
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    generation_.fetch_add(1);
    generation_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::drain(int tid) {
    if (static_sched_) {
        const size_t T = static_cast<size_t>(nthreads_), t = static_cast<size_t>(tid);
        for (size_t c = n_chunks_ * t / T; c < n_chunks_ * (t + 1) / T; ++c) fn_(ctx_, c, tid);
        return;
    }
    for (;;) {
        const size_t c = next_.fetch_add(1, std::memory_order_relaxed);
        if (c >= n_chunks_) break;
        fn_(ctx_, c, tid);
    }
}

void ThreadPool::worker(int tid) {
    unsigned seen = 0;
    for (;;) {
        unsigned g = generation_.load(std::memory_order_acquire);
        for (int spin = 0; g == seen && spin < 20000; ++spin) {
            _mm_pause();
            g = generation_.load(std::memory_order_acquire);
        }
        if (g == seen) {
            generation_.wait(seen, std::memory_order_acquire);
            g = generation_.load(std::memory_order_acquire);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = g;
        drain(tid);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::run(Fn fn, void* ctx, size_t n_chunks) {
    fn_ = fn;
    ctx_ = ctx;
    n_chunks_ = n_chunks;
    next_.store(0, std::memory_order_relaxed);
    pending_.store(nthreads_ - 1, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    drain(0);
    while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
}

} // namespace ternix
