#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace ternix {

// Fixed-size, 64-byte-aligned, zero-initialised buffer (no resize; inner loops index raw pointers).
template <class T>
class AlignedBuf {
public:
    AlignedBuf() = default;
    explicit AlignedBuf(size_t n) { reset(n); }
    ~AlignedBuf() { release(); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    AlignedBuf& operator=(AlignedBuf&& o) noexcept {
        if (this != &o) {
            release();
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
    void reset(size_t n) {
        release();
        if (n == 0) return;
        const size_t bytes = (n * sizeof(T) + 63) / 64 * 64;
#ifdef _MSC_VER
        p_ = static_cast<T*>(_aligned_malloc(bytes, 64));
#else
        p_ = static_cast<T*>(std::aligned_alloc(64, bytes));
#endif
        if (!p_) throw std::bad_alloc();
        std::memset(static_cast<void*>(p_), 0, bytes);
        n_ = n;
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return p_[i]; }
    const T& operator[](size_t i) const { return p_[i]; }

private:
    void release() {
        if (!p_) return;
#ifdef _MSC_VER
        _aligned_free(p_);
#else
        std::free(p_);
#endif
        p_ = nullptr;
        n_ = 0;
    }
    T* p_ = nullptr;
    size_t n_ = 0;
};

} // namespace ternix
