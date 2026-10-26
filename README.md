# TernixEngine
**A SIMD-native kernel library for 1.58-bit (ternary) LLM inference: AVX2 on CPU, `cp.async`-pipelined CUDA on GPU**
*Distributed under the Elastic License 2.0 (see [License](#15-license)).*

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![CUDA](https://img.shields.io/badge/CUDA-sm__80%2B-green) ![AVX2](https://img.shields.io/badge/SIMD-AVX2-orange) ![Python](https://img.shields.io/badge/Python-PyBind11-yellow)

## Abstract
TernixEngine is a C++20 / CUDA library for the operation that dominates inference of 1.58-bit ("ternary") large language models: the matrix product of int8 activations with weights restricted to {-1, 0, +1} and stored at two bits each. Because every weight is -1, 0 or +1, a multiply degenerates into an add, a subtract or nothing, and four weights fit in one byte, an 8x reduction in weight traffic relative to FP16. The engineering question this project studies is how to keep that advantage on real hardware, where the usual implementation (unpack the weights to int8 or FP16, then run a conventional GEMM) throws it away.

The kernels in this repository answer that question in three ways:

* **No materialised unpacking.** Weights are decoded inside a 256-bit AVX2 register (CPU) or inside the thread that consumes them (GPU). No dequantised weight tensor ever exists in memory.
* **No weight-dependent control flow.** The ternary value is turned into a sign control and applied with `_mm256_sign_epi32` (CPU) or with arithmetic selects (GPU), so inner loops contain no `if (w == 0)` branches and do not diverge.
* **Bit-exact verification.** Every kernel is checked against an independent integer GEMM on the unpacked weights, from C++ (GoogleTest), from Python (NumPy) and GPU-versus-CPU, over ragged shapes and worst-case values.

> [!NOTE]
> **Status: active research and development.** This is a *kernel library*, not yet an end-to-end LLM runtime. The weight quantizer (`scripts/quantize.py`), the model loader and the transformer block are stubs; there is no tokenizer path and no tokens-per-second figure. [Section 10](#10-implementation-status) states precisely what works and what does not. Performance numbers are measured against in-repo baselines only; the library has not yet been compared head-to-head with other ternary engines such as bitnet.cpp or T-MAC ([Section 11](#11-limitations-and-research-roadmap)).

## Table of Contents

1. [Problem Statement: The De-Quantization Wall and Warp Divergence](#1-problem-statement-the-de-quantization-wall-and-warp-divergence)
2. [Ternary Inference in Plain Terms](#2-ternary-inference-in-plain-terms)
3. [Kernel Design and Methodology](#3-kernel-design-and-methodology)
4. [Software Architecture and Data Flow](#4-software-architecture-and-data-flow)
5. [Repository Layout](#5-repository-layout)
6. [Technology Stack](#6-technology-stack)
7. [Build, Run and Python Usage](#7-build-run-and-python-usage)
8. [Testing, Continuous Integration and Build Automation](#8-testing-continuous-integration-and-build-automation)
9. [Performance Evaluation](#9-performance-evaluation)
10. [Implementation Status](#10-implementation-status)
11. [Limitations and Research Roadmap](#11-limitations-and-research-roadmap)
12. [Troubleshooting](#12-troubleshooting)
13. [Support and Maintenance](#13-support-and-maintenance)
14. [Contributing](#14-contributing)
15. [License](#15-license)
16. [Citation](#16-citation)

## 1. Problem Statement: The De-Quantization Wall and Warp Divergence
Engines that execute sub-byte quantized LLMs commonly run into three hardware-level problems, and ternary weights make each of them easy to trigger.

1. **The de-quantization wall.** The straightforward implementation expands 2-bit weights into int8 or FP16 arrays before the GEMM loop. The expanded array is 4x to 8x larger than the packed one, so the loop re-reads far more data than the compression saved, saturates L1 bandwidth and leaves the arithmetic units idle. Compression only pays off if the weights stay packed until the moment they are consumed.
2. **Sparse-branch misprediction.** Ternary matrices contain many zeros, and the natural way to exploit that is `if (w == 0) continue;`. The pattern of zeros is data-dependent and effectively random, so the branch predictor fails, the pipeline flushes, and the cost of the mispredictions can exceed the arithmetic that was skipped. Zero must be handled arithmetically (multiply by a sign mask), not by control flow.
3. **GPU transaction and divergence problems.** Reading sub-byte values directly from global memory produces unaligned, narrow loads that break coalescing; lanes that branch on their own weight diverge within a warp. Both lower SM utilisation. The fix is to move whole tiles with wide, aligned, asynchronous copies and decode only after the data is on-chip.

## 2. Ternary Inference in Plain Terms
*For readers who are not kernel engineers.* A neural network is mostly a very large number of multiplications between numbers it has learned (weights) and numbers flowing through it (activations). "Ternary" models are trained so that every weight is only -1, 0 or +1. The multiplication then reduces to "add this number, subtract it, or ignore it", which is far cheaper than a floating-point multiply and needs about an eighth of the memory for the weights. For language-model inference the memory traffic is usually the limiting factor, so this matters: a model that is small and cheap to run can fit on a laptop CPU or a modest GPU.

The catch is that standard hardware and standard software are built for multiplication of wider numbers, and the obvious way to run a ternary model (convert it back to wider numbers) erases the savings. TernixEngine is a set of low-level routines that keep the data in its small packed form until the instant it is used, and that avoid the branching logic that modern processors handle badly. It is currently a verified, benchmarked set of building blocks; turning them into a complete model runner is the next stage of the project.

## 3. Kernel Design and Methodology
The design draws on four lines of work on low-bit LLM inference (BitNet / BitNet b1.58, T-MAC, QuIP#, MARLIN; the papers are kept under `references/`, and `docs/RESEARCH_SYNTHESIS.md` records what was taken from each). What follows describes what is *implemented*; ideas from those papers that are not yet implemented are listed in [Section 11](#11-limitations-and-research-roadmap).

### 3.1 Weight format
All kernels compute `C[M,N] (int32) = A[M,K] (int8) x W[K,N]`, where `W` is ternary. Weights are packed four per byte, little-endian along N, into a row-major `K x (N/4)` byte array. Each weight occupies a two-bit field:

| bits | `00` | `01` | `11` |
|---|---|---|---|
| value | 0 | +1 | -1 |

`10` is never produced by the packer (`pack_ternary_weights`) and is outside the supported domain. The encoding is chosen so that the value is recovered by *sign-extending the two-bit field* (`11` sign-extends to -1, `01` stays +1), which is a shift-left/arithmetic-shift-right pair rather than a table lookup or a compare. The same layout feeds every CPU and GPU kernel, which is what makes cross-checking them against each other meaningful. `bitpack.cpp` provides pack/unpack and the round trip is tested.

### 3.2 CPU kernels (AVX2)
* **Reference and baselines.** `mul_mat_ternary_naive` is the textbook i-j-k loop with a branching decode; it walks down a column of `W` and is deliberately cache-hostile. `mul_mat_ternary_scalar` is the fair scalar baseline: i-k-j order and a branch-free decode, `w = (b & 1) - (b & 2)`. Speed-ups are quoted against the scalar kernel as well as the naive one.
* **In-register decode (`mul_mat_ternary_avx2`).** For each activation scalar `a[i,k]`, the packed weight byte(s) covering eight output columns are broadcast into a 256-bit register. `_mm256_srlv_epi32` with the per-lane shift vector (0, 2, ..., 14) moves a different weight into the low bits of each 32-bit lane, `_mm256_and_si256` masks the field, and `slli`/`srai` by 30 sign-extends it into a lane value of -1, 0 or +1. `_mm256_sign_epi32(a_broadcast, w)` then yields `-a`, `0` or `+a` in one instruction, with no branch and no multiply, and `_mm256_add_epi32` accumulates. The loop nest is M, K, N with N tiled by eight, so the eight int32 accumulators stay in one register for the whole K loop.
* **Register blocking (`mul_mat_ternary_avx2_blocked`).** Decoding the weights is as expensive as using them, so the blocked variant processes four activation rows per pass: each decoded weight vector is applied to four accumulators before the next weight is loaded, amortising the decode cost by 4x. Rows that do not fill a block fall through to the one-row kernel.
* **Auxiliary operators.** `rms_norm_avx2` is a real AVX2+FMA implementation (tested against the scalar version for sizes that are not multiples of eight); `swiglu` is scalar.
* **Safety properties.** Loads use `memcpy` (no unaligned-access UB); int32 accumulation cannot overflow for int8 activations with K up to a stated bound (a K = 4096 worst-case test covers it).

### 3.3 GPU kernels (CUDA)
* **Baseline (`mul_mat_ternary_cuda_naive`).** One thread per output element, straight from global memory. It exists to give the optimised kernel something honest to be measured against.
* **Tiled kernel (`mul_mat_ternary_cuda_tiled`).** The tile is 64x64x64 with 256 threads (a 16x16 arrangement), and each thread owns a 4x4 micro-tile of outputs, so one packed weight byte covers exactly the four columns a thread computes. Tiles are moved global-to-shared with `cp.async.cg` in 16-byte copies through a **two-stage pipeline**: while the threads do arithmetic on tile *t*, the copies for tile *t+1* are already in flight, and `cp.async.wait_group 1` blocks only on the older group. The int8 activation tile uses a padded row pitch (80 bytes instead of 64) so that the threads of a warp reading different rows do not hit the same shared-memory bank. The kernel requires M, N and K to be multiples of 64 and validates the shapes on the host.
* **Hardware requirement.** `cp.async` needs compute capability 8.0 or newer; the build defaults to the native architecture and falls back to sm_80.

### 3.4 Correctness model
The unit of truth is a deliberately simple integer GEMM on the *unpacked* int8 weights, written independently of every kernel. All kernels, CPU and GPU, must match it exactly (integer arithmetic, so equality, not a tolerance), including ragged shapes, rows that do not fill a block, and int8-extreme inputs that would expose accumulator overflow.

## 4. Software Architecture and Data Flow
<div align="center">
  <img src="docs/assets/ternix-architecture.png" alt="TernixEngine architecture: Python and C++ entry points, shared packed-weight layout, CPU AVX2 and CUDA kernels, test and benchmark harnesses" width="100%" />
</div>

The Python module `ternix_engine` (PyBind11) packs NumPy weights and dispatches to any CPU or CUDA kernel by name; the C++ core exposes the same kernels to the GoogleTest suites and to the Google Benchmark and CUDA-event benchmark harnesses. All paths use the single weight layout described in [Section 3.1](#31-weight-format), so a result produced by one kernel can be reproduced by any other. `Model` and `TransformerLayer` are placeholders for the future end-to-end path and are not exercised by any test or benchmark.

## 5. Repository Layout
```text
TernixEngine/
├── include/ternix/            # Public headers
│   ├── mul_mat_ternary.h      #   CPU GEMM kernels
│   ├── cuda_kernels.cuh       #   GPU GEMM kernels and host wrappers
│   ├── bitpack.hpp            #   2-bit pack / unpack
│   ├── simd_math.hpp          #   RMSNorm (AVX2+FMA), SwiGLU
│   ├── tensor.hpp             #   32-byte-aligned, move-only tensor storage
│   └── model.hpp, transformer.hpp   # placeholders for the end-to-end path
├── src/
│   ├── mul_mat_ternary.cpp    # naive / scalar / AVX2 / AVX2-blocked kernels
│   ├── cuda_kernels.cu        # naive and cp.async-tiled kernels + host wrappers
│   ├── bitpack.cpp, simd_math.cpp, tensor.cpp
│   └── model.cpp, transformer.cpp, main.cpp   # stubs / CLI entry point
├── bindings/python/           # PyBind11 module: pack_ternary, unpack_ternary,
│                              #   ternary_matmul(impl=...), has_cuda, Model
├── tests/                     # GoogleTest (CPU, CUDA) and pytest (binding)
├── benchmarks/                # Google Benchmark (CPU), CUDA-event harness (GPU), charts
├── scripts/                   # plot_results.py (charts + README tables), quantize.py (stub)
├── docs/                      # RESEARCH_SYNTHESIS.md, architecture diagram (PNG + .excalidraw)
├── references/                # the foundational papers (BitNet, T-MAC, QuIP#, MARLIN)
├── CMakeLists.txt, run_all.bat, run_all.sh
└── .github/workflows/ci.yml
```

## 6. Technology Stack
* **C++20** (MSVC, GCC, Clang), AVX2 and FMA intrinsics (`<immintrin.h>`).
* **CUDA** (sm_80 and newer, inline PTX for `cp.async`); the CUDA code is compiled as C++17 so older CMake versions in container images configure correctly.
* **PyBind11** for the Python module; **NumPy** for the Python reference.
* **CMake** (>= 3.20) with Ninja; dependencies (Google Benchmark, GoogleTest, PyBind11) fetched with `FetchContent`.
* **GoogleTest, Google Benchmark, pytest, matplotlib** for verification, measurement and reporting.

## 7. Build, Run and Python Usage

### System requirements
**Hardware:** an x86-64 CPU with AVX2 and FMA (Intel Haswell / AMD Zen or newer). Optional: an NVIDIA GPU with compute capability 8.0 or newer (Ampere onward) to run the CUDA kernels and tests. 16 GB of RAM is comfortable.

**Software:** a C++20 compiler (MSVC from Visual Studio 2022/2026 Build Tools, GCC 11+, or Clang), CMake 3.20+, Ninja (recommended), Python 3.9+ with NumPy and matplotlib, Git (CMake downloads the dependencies), and optionally the CUDA toolkit with `nvcc` on `PATH`. Linux, WSL2 and Windows are supported.

### Automated pipeline
`run_all` configures, builds, runs the C++ and Python tests, runs the benchmarks (median of five repetitions, pinned to one core via `TERNIX_BENCH_CPU`), and regenerates the charts and the tables in [Section 9](#9-performance-evaluation).

```bash
# Linux / WSL
./run_all.sh
```
```bat
:: Windows: open "x64 Native Tools Command Prompt for VS" (not PowerShell)
run_all.bat
```

### Using the kernels from Python
```python
import numpy as np, ternix_engine as tx          # build/ must be on PYTHONPATH
a = np.random.randint(-128, 128, (64, 256), dtype=np.int8)
w = np.random.randint(-1, 2, (256, 128), dtype=np.int8)
c = tx.ternary_matmul(a, tx.pack_ternary(w), impl="avx2_blocked")
# impl: naive | scalar | avx2 | avx2_blocked | cuda_naive | cuda_tiled
assert (c == a.astype(np.int32) @ w.astype(np.int32)).all()
```

## 8. Testing, Continuous Integration and Build Automation
The test strategy is differential: every optimised kernel is compared with an independent, obviously correct implementation, and the comparison is exact.

* **C++ (`ternix_tests`):** every CPU kernel against an independent integer GEMM on unpacked weights over ragged shapes (M not a multiple of the row block, K not a multiple of eight), plus an int8-extreme case at K = 4096 that would expose accumulator overflow; pack/unpack round trip; RMSNorm AVX2 versus scalar across sizes; SwiGLU known values.
* **CUDA (`ternix_cuda_tests`):** both GPU kernels against the CPU reference, bit-exact, including a 64x4096x64 case that exercises many pipeline iterations; the host wrapper's rejection of unsupported shapes.
* **Python (`tests/test_python_binding.py`):** all CPU and CUDA implementations against NumPy `int32` matmul on random ternary data (parametrised over implementations), and input validation (wrong dimensionality or shape raises `ValueError`).
* **CI (`.github/workflows/ci.yml`):** a Linux (GCC) job and a Windows (MSVC) job each build, run all CPU tests and the binding tests, and execute a benchmark smoke run; a CUDA container job compiles the GPU code for sm_80, 86, 89 and 90. Hosted runners have no GPU, so GPU *execution* is verified locally, and the README says so rather than implying otherwise.

Run the suites with `ctest --output-on-failure` from the build directory, or let `run_all` do it.

## 9. Performance Evaluation
*Testbed: Intel Core i7-14650HX (single thread, pinned to a P-core), NVIDIA GeForce RTX 5060 Laptop GPU (sm_120), MSVC 19.5, CUDA 13.3. The tables below are regenerated by `run_all` from `benchmarks/*.json`; none of the numbers is hand-entered.*

### Method
Inputs are seeded random int8 activations and random ternary weights, identical for every kernel. CPU timings are the median of five repetitions of a single-threaded run pinned to a P-core (unpinned runs on a hybrid CPU can be several times slower if the scheduler places the thread on an E-core). GPU timings use CUDA events around the kernel only, with data resident on the device, median of 31 launches, and every result is checked bit-for-bit against the CPU kernel before the time is reported.

### CPU
<!-- BENCH_CPU_START -->
CPU: 24 logical cores, one thread, release build. Seeded random int8 activations and random ternary weights (identical for every kernel).

| Kernel | 128³ ms | 256³ ms | 512³ ms | G ops/s @ 512³ | vs naive @ 512³ |
|---|---:|---:|---:|---:|---:|
| Naive (i-j-k, branching decode) | 5.31 | 53.12 | 445.31 | 0.60 | 1.0× |
| Scalar (i-k-j, branch-free decode) | 1.47 | 11.23 | 86.81 | 3.09 | 5.1× |
| AVX2, one row at a time | 0.22 | 1.80 | 14.38 | 18.67 | 31.0× |
| AVX2, 4-row register blocking | 0.13 | 1.07 | 8.54 | 31.43 | 52.1× |
<!-- BENCH_CPU_END -->

<div align="center">
  <img src="benchmarks/benchmark_results.png" alt="CPU benchmark results for the naive, scalar, AVX2 and blocked AVX2 kernels" width="85%" />
</div>

### GPU
<!-- BENCH_GPU_START -->
GPU: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120). CUDA-event kernel time, data resident on device, median of 31 launches; every result is checked bit-for-bit against the CPU kernel.

| Size | naive ms | tiled (cp.async) ms | speed-up | tiled effective TOPS |
|---:|---:|---:|---:|---:|
| 512³ | 0.261 | 0.066 | 4.0× | 4.08 |
| 1024³ | 1.866 | 0.341 | 5.5× | 6.29 |
| 2048³ | 14.853 | 2.657 | 5.6× | 6.47 |
| 4096³ | 118.893 | 21.280 | 5.6× | 6.46 |
<!-- BENCH_GPU_END -->

<div align="center">
  <img src="benchmarks/benchmark_cuda.png" alt="GPU benchmark results for the naive and tiled CUDA kernels" width="85%" />
</div>

### Interpretation
* **What the CPU numbers show.** Relative to the fair scalar baseline, the AVX2 kernel is about 6x faster and the register-blocked kernel about 10x faster at 512³. The jump from one-row to four-row blocking is a 1.7x gain from a single idea, decoding once and using the result four times, which confirms that decode cost, not accumulation, is the bottleneck on this path. The 52x figure against the naive loop mixes the SIMD gain with the cache behaviour of its loop order; the scalar row is the honest comparison.
* **What the GPU numbers show.** The tiled kernel is 4.0x to 5.6x faster than the naive GPU kernel and its throughput plateaus at about 6.5 effective TOPS from 1024³ upward, which is the signature of a compute-bound regime on CUDA cores. The speed-up grows with size because the one-time pipeline fill and launch costs amortise.
* **What they do not show.** "Effective TOPS" counts two operations per ternary multiply-accumulate and is not a tensor-core peak; the kernel uses CUDA cores. It has **not** been compared with cuBLAS/cuBLASLt int8, nor with other ternary runtimes. Do not read these figures as evidence of being faster than existing libraries; they establish a verified, reproducible baseline.

## 10. Implementation Status
TernixEngine is a verified kernel library with scaffolding for a future runtime.

### Working and verified
* **Weight packing:** 2-bit pack/unpack with a tested round trip.
* **CPU GEMM:** naive, scalar, AVX2 and 4-row-blocked AVX2 kernels, bit-exact against the reference.
* **GPU GEMM:** naive and `cp.async`-pipelined tiled kernels, bit-exact against the CPU.
* **AVX2+FMA RMSNorm**, and a scalar SwiGLU.
* **Python module** exposing packing, every kernel by name, and a CUDA-availability query.
* **Benchmark and reporting pipeline**, CI, and a one-command build and verify script.

### Not yet implemented (active research and development)
* `scripts/quantize.py` is a stub: it does not load a HuggingFace checkpoint, apply BitNet's absmean scaling, or pack a real model.
* `Model::load` and `TransformerLayer::forward` are stubs: there is no attention or feed-forward forward pass, no KV cache and no tokenizer path, hence no tokens-per-second numbers and no perplexity check.
* `interleave_weights_offline` is an identity copy. The layout is plain row-major packing along N, not the offline interleaving or lookup-table scheme of T-MAC.
* The GPU kernel does not use tensor cores and has not been compared with vendor int8 GEMM.

## 11. Limitations and Research Roadmap
1. **End-to-end path (highest priority).** A quantizer that converts a published ternary checkpoint (BitNet b1.58 2B4T and the open ternary Bonsai models are natural targets), a model loader, a BitNet block (attention plus SwiGLU feed-forward), a KV cache, and a perplexity check against the reference implementation. Only then is a tokens-per-second figure meaningful.
2. **Comparison with prior art.** Head-to-head measurements on one machine against bitnet.cpp (I2_S and the TL1/TL2 lookup-table kernels), T-MAC and llama.cpp's ternary types on CPU, and against BitBLAS and cuBLAS on GPU, with the same model, prompts, threads and pinning. Until this exists, the claims in this repository are limited to correctness and to speed-ups over its own baselines.
3. **CPU.** An AVX-512 VNNI variant (`_mm512_dpbusd_epi32`); multi-threading over M (OpenMP is already linked); lookup-table kernels in the style of T-MAC and QuIP# codebooks; offline weight interleaving.
4. **GPU.** An int8 tensor-core (`mma.sync`) path with `ldmatrix`-friendly layouts; tiles for ragged shapes; a skinny-M (decode, M = 1) kernel, since autoregressive decoding is memory-bound and quite unlike the large-GEMM case benchmarked here; CUDA-graph integration.
5. **Exploiting structure in real models.** Published analysis of pretrained 1.58-bit models reports a large fraction of zero weights (on the order of 40%); whether skip-zero or structured-sparsity kernels can turn that into speed on these layouts is an open question this library is positioned to test.
6. **Reporting.** Energy per token, roofline analysis and Nsight Compute profiles for the GPU kernels.

## 12. Troubleshooting
* **`nvcc` is not recognised.** Add the CUDA bin directory (for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin`) to `PATH` and restart the terminal or IDE so it inherits the change.
* **Nsight Visual Studio Edition warning during CUDA install.** When a new CUDA toolkit is installed alongside a newer Visual Studio than the installer knows, it may warn that the Nsight plug-in could not be installed for an older version. This is harmless; it does not affect `nvcc` or building and running CUDA code.
* **CMake cannot find the compiler or CUDA.** On Windows use the *x64 Native Tools Command Prompt*, not PowerShell. A stale `build/CMakeCache.txt` can also cause this; `run_all` deletes it before configuring.
* **`nvcc` reports errors about multiple input files.** Optimisation flags must be wrapped in generator expressions (`$<$<COMPILE_LANGUAGE:CXX>:/O2>`), otherwise CMake passes host-compiler flags such as `/arch:AVX2` to `nvcc`, which parses them as file names.
* **`cp.async` fails to compile or run.** It requires compute capability 8.0 or newer; the build falls back to sm_80 when the native architecture cannot be detected. Mismatched driver and toolkit versions can also produce PTX errors; use a driver at least as new as the toolkit.
* **CUDA container build fails to configure.** Older CMake versions in container images lack the CUDA 20 dialect; the project sets the CUDA standard to 17.
* **Benchmarks vary by several times between runs.** On hybrid Intel CPUs an unpinned single-thread run may land on an E-core. Set `TERNIX_BENCH_CPU` to the index of a P-core (the scripts default to a P-core).
* **Intermittent MSVC `LNK1104` link errors.** Usually a transient file lock from antivirus or an indexer; re-run the build.
* **C2719 (MSVC).** Pass `__m256i` arguments by reference or pointer; MSVC rejects aligned types passed by value in some configurations.
* **Pipeline scripts.** The pipeline uses `.bat` (Windows) and `.sh` (Linux/WSL) files; the MSVC Native Tools prompt is built on `cmd.exe`, so PowerShell scripts are not used.

## 13. Support and Maintenance
Bug reports, feature requests and performance discussions belong in the issue tracker. For performance deviations please include your CPU/GPU model, compiler and CUDA versions, whether the benchmark was pinned, and the contents of `build/CMakeCache.txt`.

## 14. Contributing
Contributions are welcome under the Elastic License 2.0; see `CONTRIBUTING.md`. A new kernel must come with a differential test against the reference GEMM (ragged shapes included) in `tests/`, and the benchmark and README tables must be regenerated with `run_all` rather than edited by hand. Claims of speed-up over other libraries should include the comparison method (same model, hardware, threading and pinning).
## 15. License
This project is distributed under the **Elastic License 2.0**.

* You may view, use and modify the code, and share it with attribution.
* You may not provide the software to third parties as a hosted or managed service.
* You may not circumvent license-key functionality.

See `LICENSE` for the full text. The software is provided as-is, without warranty.

## 16. Citation
If you use TernixEngine in academic or production work, please cite it as:

```bibtex
@software{ternix_engine_2026,
  author = {Tripathi, Pundarikaksh N.},
  title  = {TernixEngine: SIMD-Native Ternary GEMM Kernels for CPU and GPU},
  year   = {2026},
  url    = {https://github.com/Quiet-Intelligence/TernixEngine}
}
```

