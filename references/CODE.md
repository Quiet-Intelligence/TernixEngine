# Competitor implementations read at source level

Method: the repositories were cloned and the relevant files read (not run). Commits: `microsoft/BitNet` 0b341e5 (2026-07-27), `ggml-org/llama.cpp` d854d98 (build 11889, the
same commit as the release binaries used in `bench/run_competitors.ps1`), `microsoft/T-MAC` 7042f8f (2025-06-03), `OpenBitSys/vlut.cpp` 7a813d9 (2026-06-24),
`Mindbeam-AI/Litespark-Inference` 671dafa (2026-06-15), `PrismML-Eng/llama.cpp` 6447906 (2026-11-06); the last four were read on 2026-11-15. Statements below are what the
code says; speed claims are not made here. Entries marked "bound" are my arithmetic on the code, not observations. None of this code was built or executed.

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

## T-MAC (`python/t_mac/weights.py`, `ops/qgemm.py`, `intrins/tbl.cc`, `model_utils.py`, `include/t-mac/tmac_gemm_wrapper.h`, `tools/run_pipeline.py`)

| Fact (file evidence) | Consequence for the comparison |
|---|---|
| `preprocess_weights` splits every b-bit weight into b one-bit planes (`(w >> ib) & 1`) and packs g = 4 one-bit weights per table index; ternary weights are passed as bits = 2 | a ternary matrix costs two 16-entry lookups per 4 weights (one per plane); the third state is not exploited (2.0 bits stored and processed for 1.58 bits of information) |
| the table holds sums of 4 **float** activations, re-quantised to int8 per activation group (`LUT_Scales = max abs(sum of g activations) / 127`, `QGeMMLUTBitsPreprocessorCodegen`); the tensorised kernel is verified with `nmse <= 5e-4` and the table with `atol = 1` (`_verify`) | the arithmetic is not the W1.58A8 integer arithmetic of the BitNet reference: the rounding happens on 4-activation sums, not on activations. Results are approximate by construction; this engine's integer path is bit-exact against a scalar reference |
| x86 path (`__AVX2__` branch of `tbl_g4_int8_float_update_impl`): 128-bit table duplicated into a 256-bit register, `_mm256_shuffle_epi8`, each int8 result widened to int16 (`SignedWideningAdder`), then int16 to int32 to fp32 and one `fmadd` per activation group | no VNNI, no AVX-512; one widen and add per 16-entry lookup. TVM targets are `-mcpu=core-avx2` (`platform.py`) |
| `fast_aggregation` uses `_mm256_avg_epu8` (unsigned rounding average) on the same int8 table values for which the NEON path uses the signed `vrhaddq_s8`; the docstring says "additional speedup but with nonnegligible error"; "TODO: implement fast aggregation for unified scale" | the BitNet (unified scale, `m_groups = 1`) path does not use fast aggregation; the x86 fast path differs from the ARM one as written (not executed here, so no error figure) |
| `tbl_g4_int8_int16_update_impl`: the AVX2 branch is `// TODO: implement this` | the int16 aggregation variant exists on ARM only |
| `_PRESET_KERNELS` lists kernel shapes per model with N = 1 for every entry; BitNet presets are `hf-bitnet-3b` (3200/8640), `ms-bitnet-3b` (3200/10240/800) and `hf-bitnet-large` (1536/4096). No 2560/6912 entry | 2B4T has no preset; a run needs TVM code generation for new shapes. All presets are GEMV (one token): the T-MAC kernel has no multi-token (prefill) variant, which the vlut.cpp README also states |
| `tmac_gemm_wrapper.h` and `qgemm.py`: "the parallelism of preprocessor is disabled due to large thread communication overhead ... TODO: Find out the reason" | table construction is single-threaded |
| `deploy/tuned/` contains tuned configurations for aarch64 only | the x86 kernels ship untuned defaults, consistent with the README statement "cannot guarantee significant speedup on all x86 platforms" |
| README: reported BitNet 3B decode 20 tok/s (1 core) and 48 tok/s (4 cores) on Surface Laptop 7; prefill marked TODO; profiled against llama.cpp b2794 (2024-05) | reported numbers are ARM, another model, a 2024 baseline |

Consequence for the plan: a same-model T-MAC row needs TVM code generation for the 2B4T shapes plus the T-MAC llama.cpp fork and a 2B4T conversion that the repository does not document.
Until that is run by the repository owner, T-MAC stays a literature comparison and must not appear in a ranking table.

## vlut.cpp / Vec-LUT (`ggml/src/ggml-quants-vlut.c`, `ggml/src/ggml-cpu/ggml-cpu-quants-vlut.c`, `cmake/vlut.cmake`)

| Fact | Consequence |
|---|---|
| `quantize_i2_v`: 4 trits per byte in base 3 (81 values, 2.0 bpw), stored column-major across output rows (`i2_weight[i * nrows + j]`); I1_V types use 5 trits per byte (1.6 bpw); variants `_2/_4/_8/_16` interleave rows | weights are indexed by input-column group first, so one table serves all output rows |
| `ggml_gemm_i2v_i8v_lut`: per 4 input columns `gemm_make_table_i2v` builds 81 int16 vectors of `TABLE_ENTRY_SIZE` (default 32) tokens by add/sub/negate chains; every output row then adds one table vector into an int16 accumulator; accumulators are flushed to int32 every 512 columns | the lookup cost is shared across the tokens of a tile, so the gain exists only for parallel input. At one token (decode) a table of 81 entries is built for 4 columns; the README names T-MAC as the method for non-parallel use |
| int16 accumulation over 128 lookups per 512-column group, each entry a sum of up to 4 int8 values | bound: 128 x 4 x 128 = 65,536 > 32,767 if all activations were maximal and sign-aligned; not an observed error (same caveat as the bitnet.cpp kernel). This engine accumulates in int32 |
| `quantize_row_i8_v` computes the scale from `MAX(max, x[i])` without `fabs`, then clamps to [-128, 127] | as written, a row whose largest magnitude is negative is scaled by its largest positive value and its negative tail is clipped. Not executed here; stated as a reading of the code |
| the GEMM function ignores `ith`/`nth` (`UNUSED`) and allocates `sum_i16`, `sum_i32` and the table with `malloc` on every call | threading is not inside the kernel (the call site was not traced); three allocations per matrix product |
| SIMD: `VLUT_AVX512`, `VLUT_SVE`, `VLUT_ACCELERATE` are opt-in CMake options (all OFF by default); otherwise a plain loop left to the auto-vectoriser | no hand-written AVX2 path; the default x86 build depends on auto-vectorisation of int16 adds |
| no reference to a GPU backend for the I2_V / I1_V types | CPU only |

Relation to this engine: Vec-LUT and the I2K128 GEMM solve the same problem (multi-token ternary GEMM on CPU) by different means: table sharing across tokens versus `vpdpbusd` with register tiling.
A head-to-head run on the same x86 CPU does not exist; it needs a vlut.cpp build (user-run). It is the most relevant CPU prefill comparator.

## Litespark-Inference (`litespark_inference/kernels/x86_64/matmul_free_avx_vnni.cpp`, README)

| Fact | Consequence |
|---|---|
| `pack_weights_avx_int8`: ternary weights stored one int8 per weight, rows padded to 32 | 8 bits per weight in memory, 4x the bytes read per token of a 2-bit layout; on a single-channel DDR5 machine such a kernel reaches the memory ceiling sooner |
| activations converted to uint8 by adding 128, then `_mm256_dpbusd_epi32(acc, x_u8, w_s8)` with 4 (or 8) weight rows per activation load | the same VNNI instruction as this engine with the operand roles swapped (unsigned activations, signed weights); here the unsigned operand is the 2-bit code and the correction term is the activation sum, which needs no per-row weight sums |
| OpenMP `parallel for` over rows; kernels are a `torch/extension.h` module; README: "Torchless generation is greedy-only today" | no core pinning or P/E-core handling in the kernel file; sampling is not available without torch |
| README reports pp128 and tg128 against thread count and energy per token against PyTorch | the baseline of the reported ratios is PyTorch, not bitnet.cpp or llama.cpp |

## PrismML llama.cpp fork (`ggml/src/ggml-common.h`, `ggml-cpu/arch/x86/quants.c`, `ggml-cuda/mmvq-pq2_0.cuh`, `ggml-cuda/mmq-pq2-j128.cuh`)

This is the vendor runtime for the Bonsai models that architecture 2 of this engine loads, and the strongest existing implementation found.

| Fact | Consequence |
|---|---|
| `block_pq2_0`: 32 bytes of 2-bit codes + one fp16 scale per 128 weights (2.125 bpw); `block_ptq1_0`: base-3 packing, 26 bytes + fp16 scale per 128 weights (1.75 bpw), described as lossless for group-128 ternary checkpoints | the same group-128 scale structure as our `.gscale` tensors; PTQ1_0 is denser than our 2-bit layout (1.75 against 2.0 bpw plus scales) |
| x86 `ggml_vec_dot_pq2_0_q8_0`: codes unpacked with `srlv` and mask, `sum((c-1) q) = sum(c q) - sum(q)`, `vpdpbusd` when `GGML_DPBUSD_256` is defined (maddubs otherwise), one FMA per 32-value sub-block for the activation scale and one per 128 for the weight scale; asserts `nrc == 1` | the same code-minus-sum identity and the same VNNI instruction as our group-scaled kernel. Differences: activations are Q8_0 (one scale per 32 values, ours per 128-block), one weight row per call (ours tiles several rows per activation load) |
| CUDA decode `mul_mat_vec_pq2_0_mc`: raw 2-bit codes paired with permuted activation bytes, integer bias subtraction, up to 4 rows per warp reusing the activation slice, optional fused gate/GLU epilogue, optional split-K | the same design points as our `ternary_gemv_gs_kernel`, plus fusion of the gate projection, which this engine does not have |
| CUDA prefill `mmq-pq2-j128.cuh`: `mma.sync.aligned.m16n8k32` int8 tensor-core kernel, bit-exact with the generic mmq path, compiled for sm_90 and newer (`__CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER`), adapted from Layr-Labs/cudafast-bonsai2-9b-engine | an open ternary engine does use int8 tensor cores for prefill at this commit. The statement in `ANALYSIS.md` G4 that none does is corrected there. Our tensor-core kernel is not novel as a technique; what remains specific to this engine is the per-128-block activation quantisation and the combination with the certified int8 head |
| `ggml-cuda/fwht.cu`: Hadamard transform on GPU for the Bonsai 2 rotated basis | Bonsai 2 is supported by the fork and not by this engine |

Consequence: the correct GPU and CPU comparator for architecture 2 is this fork with the `*-PQ2_0.gguf` file of Ternary-Bonsai-1.7B, not upstream llama.cpp. That run (prebuilt binaries from the Bonsai-demo repository, or a build)
is in the same class as the denied actions (executing downloaded binaries, building third-party source) and is left to the repository owner; no number for it exists in this repository.

## Open items

No competitor has been executed on this machine: `docs/research/data/competitors/` does not exist until `bench/run_competitors.ps1` is run by the repository owner.
Not traced: the vlut.cpp call site (threading), the Litespark AVX-512 and NEON kernels, the PrismML PTQ1_0 CUDA kernel and Metal/Vulkan backends.
