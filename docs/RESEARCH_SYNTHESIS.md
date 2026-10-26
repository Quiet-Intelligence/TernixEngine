> **Superseded in part (2026-10).** This note records the original synthesis of the five foundation papers. The 2024-2026 literature survey,
> the gap analysis and the measured results are in `references/INDEX.md`, `references/ANALYSIS.md` and `docs/research/EXPERIMENTS.md`. Two of the
> directives below were tested and revised: T-MAC-style offline interleaving is implemented as the I2K128 layout, and the 32x32 `__shfl_down_sync` warp
> reduction was replaced by warp-per-row dp4a and `mma.sync` kernels (docs/ARCHITECTURE.md).

# Research Synthesis: 1.58-bit SIMD-Native Inference Engine

## 1. Introduction
The pursuit of high-performance Large Language Model (LLM) inference on edge and consumer hardware necessitates radical shifts from traditional 16-bit floating-point arithmetic. The introduction of 1.58-bit ternary networks (BitNet) represents a fundamental paradigm shift. In a 1.58-bit network, every parameter is restricted to the ternary set {-1, 0, 1}, theoretically eliminating the need for computationally expensive floating-point multiplications and replacing them with highly efficient integer additions. 

## 2. Core Mathematical Bottlenecks
Despite the theoretical advantages, implementing a highly optimized engine tailored for 1.58-bit LLMs introduces several critical hardware-level bottlenecks:

1. **The De-Quantization Wall:** A naive execution paradigm involves unpacking highly compressed (e.g., 2-bit packed) weights into broader integer formats (like INT8) prior to entering the Generalized Matrix Multiplication (GEMM) loops. This intermediate expansion aggressively saturates the L1 cache bandwidth, stalling the arithmetic logic units. 
2. **Sparse Branch Misprediction:** Ternary networks frequently exhibit structural sparsity (i.e., weights that are 0). Employing conditional branching (`if (weight == 0)`) to skip these computations completely disrupts the CPU's branch predictor pipeline, inducing severe branch miss penalties. 
3. **CUDA Thread Divergence:** In GPU execution, naive extraction of sub-byte elements can lead to unaligned global memory fetches and warp divergence.

## 3. Analysis of Hardware Execution Solutions
Recent literature highlights several systematic approaches to mitigating the aforementioned bottlenecks:

### MARLIN (Mixed-Precision Auto-Regressive LINear)
MARLIN introduces GPU optimizations tailored to mixed-precision, batched inference:
- **Asynchronous Memory Transfers:** Uses `cp.async` instructions with `evict_first` cache-hints. This allows loading weight tiles directly from Global Memory into Shared Memory, entirely bypassing the L1 cache and minimizing cache pollution. 
- **Bank Conflict Mitigation:** Implements an XOR-based memory swizzling strategy for `__shared__` memory (storing element `ij` at `i(i ⊕ j)`), enabling conflict-free Tensor Core loads via `ldmatrix.sync`.
- **Striped Partitioning:** Distributes `__shared__` memory tiles across Streaming Multiprocessors (SMs) using a column-wise striping methodology. This promotes consistent SM occupancy and minimizes the need for slow global reduction synchronizations.

### QuIP# (Quantization with Hadamard Incoherence and Lattice Codebooks)
QuIP# establishes a hardware-centric methodology for rapid lookups and codebook utilization:
- **Lattice Codebooks (E8P):** Employs highly symmetric 8-dimensional unit ball packing (E8 lattice). The resulting E8P codebook is strictly confined to 1KiB, guaranteeing absolute residency within the L1 cache, thereby ensuring rapid activation or weight scaling without incurring DRAM retrieval penalties.

### T-MAC (Table Lookup for Low-Bit LLM Deployment)
T-MAC formulates CPU execution by completely discarding multiplications in favor of Lookup Table (LUT) techniques:
- **Offline Weight Interleaving:** In standard hardware architectures (which are little-endian), unpacking sub-byte weights sequentially can be problematic. T-MAC resolves this by interleaving the packed weights offline, meaning that simple bit-shifts sequentially decode weights in correct order.
- **Bit-Serial Execution & LUT:** Precomputes lookup tables (LUTs) based on input activations and accumulates partial dot-products utilizing bit-wise shifts and additions (`avg/rhadd`), ensuring strict utilization of SIMD instruction pipelines over individual element processing.

## 4. Proposed Architecture for TernixEngine
Synthesizing these insights alongside the project constraints yields the following verified architectural directives for TernixEngine:

### C++ AVX2 SIMD Architecture
- **In-Register Weight Unpacking:** Weight matrices will be statically packed to 2-bits per weight. During the CPU kernel loop, AVX2 instructions (`_mm256_srlv_epi32` and `_mm256_and_si256`) will be used to extract the 2-bit weights directly inside the 256-bit SIMD registers.
- **Offline Memory Layout:** The packed weights must be structurally interleaved in an offline preprocessing step to counteract the inherent little-endian backwards unpacking limitation of the `_mm256_srlv_epi32` instruction.
- **Branchless Masking and Accumulation:** The CPU kernel will avoid explicit decoding to INT8. Instead, the extracted 2-bit weight will determine a sign-mask or zero-mask. The `_mm256_sign_epi32` (or `_mm256_sign_epi8`) instruction will act upon the INT8 activations. A bitwise AND will zero out the activation if the ternary weight is 0. Accumulation proceeds using parallel integer adds (`_mm256_add_epi32`).

### CUDA Warp-Tiling Architecture
- **Global-to-Shared Streaming:** The CUDA kernel will leverage `cp.async` to asynchronously load 32x32 packed weight tiles and INT8 activation tiles directly into `__shared__` memory, maximizing global memory bandwidth.
- **XOR Swizzling:** To preclude shared memory bank conflicts across warps during simultaneous read/writes, `__shared__` memory array indices will apply an XOR layout transformation `i(i ⊕ j)`.
- **In-Warp Accumulation:** Extracted weights within the `__shared__` memory tile will be iteratively accumulated into local thread registers, finalized by a logarithmic parallel reduction using `__shfl_down_sync`. 
- **Striped Partitioning:** The workload division across the GPU grid will employ striped assignment, allowing multiple SMs to operate on the same tile columns iteratively to maintain maximal throughput during decode phases.
