#pragma once
#include <string>

namespace ternix {

enum class Isa { Scalar = 0, Avx2 = 1, AvxVnni = 2 };

struct CpuInfo {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool avx_vnni = false;
    bool avx512_vnni = false;
    std::string brand;
};

// Detected once (cpuid + XGETBV, so an OS that masks YMM state is respected).
const CpuInfo& cpu_info();

// Best kernel ISA for this machine; TERNIX_ISA=scalar|avx2|avx_vnni caps it (never raises it).
Isa best_isa();
const char* isa_name(Isa isa);

} // namespace ternix
