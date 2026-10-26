# Competitor implementations read at source level

Method: the repositories were cloned and the relevant files read (not run). Commits: `microsoft/BitNet` 0b341e5 (2026-07-27), `ggml-org/llama.cpp` d854d98 (build 11889, the
same commit as the release binaries used in `bench/run_competitors.ps1`). Statements below are what the code says; speed claims are not made here. T-MAC was **not** cloned
or read (see INDEX S2, level S); entries marked "bound" are my arithmetic on the code, not observations.

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

## Open items (not read)

T-MAC (`microsoft/T-MAC`), vlut.cpp, Litespark-Inference and the PrismML llama.cpp fork have not been cloned; INDEX entries S2, S4, S6 and S18 stay at verification level A/S.
No competitor has been executed on this machine yet: `docs/research/data/competitors/` does not exist until `bench/run_competitors.ps1` is run.
