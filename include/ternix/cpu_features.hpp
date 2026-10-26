#pragma once
#include <string>

namespace ternix {

enum class Isa { Scalar = 0, Avx2 = 1, AvxVnni = 2, Avx512Vnni = 3 };

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

// Best kernel ISA for this machine; TERNIX_ISA=scalar|avx2|avx_vnni caps it (never raises it). Avx512Vnni is never chosen automatically:
// it is selected only by TERNIX_ISA=avx512_vnni on a CPU that reports it (the kernel has not been run on real hardware by the author).
Isa best_isa();
const char* isa_name(Isa isa);

} // namespace ternix
