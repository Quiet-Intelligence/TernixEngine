# Architecture

```
scripts/quantize.py  --bf16 safetensors-->  .ternix (docs/FORMAT.md)  --mmap-->  ModelFile (src/format.cpp)
                                                                               |
              Tokenizer (src/tokenizer.cpp, Llama-3 byte-level BPE)            +--> Model      (CPU, src/model.cpp)
              Sampler   (src/sampler.cpp)                                      +--> CudaModel  (GPU, src/cuda_engine.cu)
              CLI       (src/main.cpp)  Python binding + server (bindings/, scripts/serve.py)
```

## Forward pass (identical arithmetic on both backends)

Per layer: `x += o_proj(attn_sub_norm(attention(rope(q), rope(k), v)))`, then
`x += down_proj(ffn_sub_norm(relu2(gate) * up))`. Each projection is a BitLinear: RMS-normed input -> per-token absmax int8
quantisation -> ternary GEMM with int32 accumulation -> rescale by `deq_step * weight_scale`. Residual stream, norms, RoPE and
softmax are float32; K/V are fp16 (CPU optionally int8); the head is the tied bf16 embedding.

Architecture 2 (Qwen3-style, e.g. Ternary-Bonsai-1.7B) replaces the sub-norm/ReLU^2 block by: per-head RMSNorm of q and k (QK-norm) before RoPE,
SiLU-GLU, YaRN rotary tables from a precomputed `rope.inv_freq` times the attention factor, per-128-group weight scales with per-128-block activation
quantisation (`i2k::gemm_rows_gs` on the CPU; `ternary_gemv_gs_kernel` and `ternary_mma_gs_kernel` on the GPU, which fold the int32 accumulators into float after
every 128-column block), and a tied fp16 head (the certified int8 head works on fp16 rows as well). Both backends run it.

## CPU backend

* `i2k::gemm_rows` (src/i2k.cpp): I2K128 weights, `vpdpbusd` (AVX-VNNI) or `vpmaddubsw`+`vpmaddwd` (AVX2) or scalar, selected at run
  time from cpuid (`TERNIX_ISA` can lower it). Codes {0,1,2} are used as the unsigned operand; the sum of the activations is
  subtracted once per row. Four output rows share each activation load; up to four activation rows share each decoded weight.
* `i2k::gemm_rows_avx512` (src/i2k_avx512.cpp, own translation unit with `/arch:AVX512`): two 128-column blocks per 64-byte weight load, `vpdpbusd` on zmm, 256-bit tail for an odd
  block count. Opt-in (`TERNIX_ISA=avx512_vnni`) and unexecuted on hardware; `tests/i2k_avx512_emulated.cpp` compiles the same source against scalar-emulated intrinsics.
* `ThreadPool` (src/threadpool.cpp): persistent workers pinned to performance cores, dynamic chunk claiming from one atomic
  counter, spin-then-wait. One `parallel_for` per fused group of projections (q/k/v, gate/up).
* `OutputHead` (src/head.cpp): optional int8 / int4 copy of the head with per-row residual norms; greedy decoding is certified
  exact for int8 (Cauchy-Schwarz bound) and 6.5-sigma statistical for int4. See the header for the proof sketch.
* Exact activation-sparse down projection (`i2k::gemv_sparse`): K-major copy of the down matrices, only nonzero input columns are
  read (80% of that input is exactly zero in BitNet 2B4T). Bit-identical to the dense kernel; batch 1 only.
* int8 KV cache (`Model::set_kv_int8`).

## CUDA backend

* `ternary_gemv_kernel<NT>`: dp4a (`dp4a.u32.s32`, unsigned codes x signed activations), up to 8 tokens per pass, warp per two rows.
* `ternary_mma_kernel<MT>`: int8 tensor cores (`mma.sync.m16n8k32`) for chunks above 8 tokens, on a byte-permuted copy of the
  weights so each lane loads its two B words with one 8-byte load; bit-identical to the dp4a kernel (tested).
* Decode (n = 1): the whole step including the argmax is captured in a CUDA graph and replayed; position and token come from
  device memory. Attention uses split-KV (256 positions per block) plus a combine kernel; prefill uses one block per (token, head).
* Certified int8 head (`head_int8`): score, bound, candidate list, exact bf16 rescoring, all on the device.

## Tests

`ctest` (GoogleTest): integer kernels against an independent reference on ragged shapes and extremes, sparse kernel against the
dense kernel, CUDA kernels against the CPU kernels. `pytest tests/`: binding, tokenizer parity with Hugging Face `tokenizers`,
engine end-to-end (needs `models/bitnet-2b4t.ternix`), exactness of the int8 head and of the sparse path.
`scripts/verify_logits.py`: logits against an independent float32 PyTorch implementation (and Hugging Face) with the noise
floor of EXP-001 as the yardstick.
