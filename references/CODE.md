# Competitor implementations read at source level

Method: the repositories were cloned and the relevant files read (not run). Commits: `microsoft/BitNet` 0b341e5 (2026-07-27), `ggml-org/llama.cpp` d854d98 (build 11889, the
same commit as the release binaries used in `bench/run_competitors.ps1`). Statements below are what the code says; speed claims are not made here. T-MAC is read at README level only
(see below and INDEX S2); entries marked "bound" are my arithmetic on the code, not observations.

## bitnet.cpp, I2_S kernel (`src/ggml-bitnet-mad.cpp`)

| Fact (file evidence) | Consequence for the comparison |
|---|---|
| `QK_I2_S` is 128 on AVX/AVX2/SSSE3 and 64 on NEON (lines 11-15); weights are 2-bit codes, four 32-byte lanes per 128-column block selected by shifts 6/4/2/0 and mask 0x03 | the same block idea as our I2K128 layout (`docs/FORMAT.md`): not a layout advantage for either side |
| x86 inner loop uses `_mm256_maddubs_epi16` and sums into **int16** lanes over up to 32 iterations x 4 products, then `_mm256_madd_epi16` (lines ~210-250) | bound: each maddubs lane is at most 2x2x128 = 512 in magnitude, 128 additions could reach 65,536 > 32,767 if every product had the same sign and maximal size; real activations mix signs, so I do not claim an observed error. Our kernels accumulate in int32 (`vpdpbusd`) |
| `grep -ci 'vnni\|dpbusd'` over the file returns 0; AVX-512 appears only in preprocessor guards | no AVX-VNNI path: on a Raptor Lake part the competitor leaves `vpdpbusd` unused |
| `s[row] = (float)sumi;` the kernel returns an integer dot; per-tensor and activation scales are applied by the caller | scale handling is outside the kernel; a fair end-to-end number must come from the whole engine, not the kernel |
| GPU: `bitlinear_int8xint2` (`gpu/bitnet_kernels/bitnet_kernels.cu`) launches `ladder_int8xint2_kernel<1, N, K, ...>` for ten hard-coded (N, K) pairs (2560/6912/3840/13824/3200/20480/..., all with M = 1) | GEMV only, shape-specialised to listed models; no prefill GEMM, no tensor cores; our CUDA path has a tensor-core prefill kernel and a generic shape path |
| LUT kernels (TL1/TL2) come from `preset_kernels/` for three models (bitnet_b1_58-large, -3B, Llama3-8B-1.58-100B-tokens) plus generated headers (`setup_env.py` lists Falcon3 1B-10B and 2B4T as supported) | a new matrix shape needs generated kernel headers, not a runtime switch |

## llama.cpp TQ2_0 / TQ1_0 (`ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml-common.h`, `ggml-cuda/`)

| Fact | Consequence |
|---|---|
| `block_tq2_0` = 64 bytes of 2-bit codes + one fp16 scale per 256 weights (66 bytes, 2.0625 bits/weight) (`ggml-common.h` 285-288) | the per-tensor ternary scale is stored per 256 weights in fp16 (extra 0.06 bpw over I2K128's 2.0) |
| `ggml_vec_dot_tq2_0_q8_K` asserts `nrc == 1` (one weight row per call); AVX2 only (`maddubs_epi16`, int16 sums per 256-block, comment: "256*127 still fits"), then subtracts the Q8_K block sums `bsums` | the same unsigned-code-minus-activation-sum correction as ours; no AVX-VNNI path in this function and no multi-row register tiling; Q8_K activations carry one fp32 scale per 256 values, not per token |
| `grep` for `TQ2_0`/`TQ1_0` under `ggml/src/ggml-cuda` finds nothing; `ggml-vulkan` has `mul_mat_vec_tq2_0.comp` and dequant shaders | llama.cpp on CUDA has no ternary-type kernel at this commit: a `-ngl 99` TQ2_0 run is not a GPU measurement of ternary inference. The script therefore adds Q2_K and Q4_0 as the closest CUDA-supported formats (not lossless, labelled with their bits per weight) and records the TQ2_0 CUDA attempt without treating it as a result |
| the BitNet converter (`conversion/bitnet.py`) loads the vocabulary with `_set_vocab_sentencepiece()`, which requires `tokenizer.model` | the 2B4T checkpoint ships only a BPE `tokenizer.json`; the script patches this one call (diff saved) because the vocabulary does not influence `llama-bench` |
| `BitnetModel.weight_quant` re-ternarises the weights with the per-tensor absmean rule before writing | the f16 GGUF holds the same ternary values and scale as our `.ternix` conversion; TQ2_0 stores them without further loss apart from the fp16 block scale |

## T-MAC (`microsoft/T-MAC`), README level only (fetched 2026-10-26; source not read, verification level S)

| Fact from the README | Consequence |
|---|---|
| ternary support is stated as "W1(.58)A8 from BitNet"; the usage example and the one reported BitNet benchmark use `bitnet_b1_58-3B`; no 2B4T model is mentioned | the model T-MAC demonstrates is the older Llama-style 3B checkpoint (no ReLU^2 / SubLN), which this engine's architecture 1 does not load, so there is no shared model for a head-to-head run without porting one side |
| end-to-end inference goes through a llama.cpp integration; benchmarks were profiled on llama.cpp b2794 (May 2024), rebased to a newer llama.cpp on 2024-10-10 | the llama.cpp version differs from the b11889 used here; any T-MAC number quoted from the README is tied to a 2024 baseline |
| build needs Python 3.8 (for TVM), cmake >= 3.22, TVM built from source; on Windows Visual Studio 2019/2022, Clang 17.0.6 and conda | a heavy third-party toolchain on this machine; the auto-mode classifier has denied building third-party source, so this has to be a user-run step |
| "cannot guarantee significant speedup ... on all x86 platforms"; recommended evaluation devices are Surface Book 3 or ARM | the authors do not claim a gain on x86 desktop CPUs; the i7-14650HX is such a part |
| no GPU path described (CUDA GPUs appear only as comparison baselines on Jetson); NPU extension noted 2025-05 | GPU comparison is not applicable |
| reported BitNet 3B: Surface Laptop 7 20 tok/s on 1 core, 48 tok/s on 4 cores (4-5x over llama.cpp); prefill results are marked "TODO" | only decode is reported, on ARM, against a 2024 llama.cpp |

Consequence for the plan: a same-model T-MAC row would need either bitnet_b1_58-3B in this engine (not supported) or a T-MAC conversion of 2B4T (not documented); until one of those exists T-MAC stays a literature comparison
(reported numbers, different model, different hardware) and must not appear in a ranking table.

## Open items (not read)

vlut.cpp, Litespark-Inference and the PrismML llama.cpp fork have not been cloned; INDEX entries S4, S6 and S18 stay at verification level A/S; T-MAC is covered at README level above.
No competitor has been executed on this machine yet: `docs/research/data/competitors/` does not exist until `bench/run_competitors.ps1` is run.
