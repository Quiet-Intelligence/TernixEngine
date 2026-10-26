#pragma once
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace ternix {

// Persistent worker pool with dynamic chunk scheduling. Chunks are claimed from one atomic counter, so threads that run
// on slower cores (E-cores on hybrid Intel parts) simply claim fewer chunks instead of stalling a static partition.
// Workers spin briefly after a job and then block on the generation counter (std::atomic::wait).
class ThreadPool {
public:
    // threads: total including the caller; <= 0 selects the number of performance cores (or all cores if unknown).
    // pin: bind workers (and the caller) to performance-core logical processors where the OS allows it.
    explicit ThreadPool(int threads = 0, bool pin = true);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return nthreads_; }

    // Runs fn(chunk, thread_id) for chunk in [0, n_chunks). Returns when all chunks are done. Not re-entrant.
    template <class F>
    void parallel_for(size_t n_chunks, F&& fn) {
        if (n_chunks == 0) return;
        if (nthreads_ == 1 || n_chunks == 1) {
            for (size_t i = 0; i < n_chunks; ++i) fn(i, 0);
            return;
        }
        struct Thunk {
            static void call(void* ctx, size_t c, int t) { (*static_cast<std::remove_reference_t<F>*>(ctx))(c, t); }
        };
        run(&Thunk::call, &fn, n_chunks);
    }

    // Logical processors that are performance cores (empty if the OS does not report core classes).
    static std::vector<int> performance_cores();

private:
    using Fn = void (*)(void*, size_t, int);
    void run(Fn fn, void* ctx, size_t n_chunks);
    void worker(int tid);
    void drain(int tid);

    int nthreads_;
    std::vector<std::thread> workers_;
    std::atomic<unsigned> generation_{0};
    std::atomic<size_t> next_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    size_t n_chunks_ = 0;
};

} // namespace ternix
