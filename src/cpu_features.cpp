#include "ternix/cpu_features.hpp"
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid(int r[4], int leaf, int sub) { __cpuidex(r, leaf, sub); }
static unsigned long long xgetbv0() { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid(int r[4], int leaf, int sub) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
static unsigned long long xgetbv0() {
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

namespace ternix {
namespace {

CpuInfo detect() {
    CpuInfo c;
    int r[4] = {};
    cpuid(r, 0, 0);
    const int max_leaf = r[0];
    char brand[49] = {};
    cpuid(r, static_cast<int>(0x80000000), 0);
    if (static_cast<unsigned>(r[0]) >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            cpuid(r, static_cast<int>(0x80000002 + i), 0);
            std::memcpy(brand + 16 * i, r, 16);
        }
    }
    c.brand = brand;
    cpuid(r, 1, 0);
    const bool osxsave = (r[2] >> 27) & 1;
    c.fma = (r[2] >> 12) & 1;
    c.f16c = (r[2] >> 29) & 1;
    const bool ymm_ok = osxsave && ((xgetbv0() & 0x6) == 0x6);
    if (max_leaf >= 7) {
        cpuid(r, 7, 0);
        c.avx2 = ymm_ok && ((r[1] >> 5) & 1);
        const bool zmm_ok = osxsave && ((xgetbv0() & 0xE6) == 0xE6);
        c.avx512_vnni = zmm_ok && ((r[1] >> 16) & 1) && ((r[2] >> 11) & 1);
        cpuid(r, 7, 1);
        c.avx_vnni = ymm_ok && ((r[0] >> 4) & 1);
    }
    c.fma = c.fma && ymm_ok;
    c.f16c = c.f16c && ymm_ok;
    return c;
}

} // namespace

const CpuInfo& cpu_info() {
    static const CpuInfo info = detect();
    return info;
}

Isa best_isa() {
    const CpuInfo& c = cpu_info();
    Isa best = c.avx_vnni ? Isa::AvxVnni : (c.avx2 ? Isa::Avx2 : Isa::Scalar);
    if (const char* e = std::getenv("TERNIX_ISA")) {
        Isa cap = best;
        if (!std::strcmp(e, "scalar")) cap = Isa::Scalar;
        else if (!std::strcmp(e, "avx2")) cap = Isa::Avx2;
        else if (!std::strcmp(e, "avx_vnni")) cap = Isa::AvxVnni;
        if (static_cast<int>(cap) < static_cast<int>(best)) best = cap;
    }
    return best;
}

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2: return "avx2";
        case Isa::AvxVnni: return "avx_vnni";
    }
    return "?";
}

} // namespace ternix
