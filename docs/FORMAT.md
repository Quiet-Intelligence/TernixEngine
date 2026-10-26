# The .ternix container

Little-endian, memory-mappable, one file per model. Writer: `scripts/ternix_format.py`; reader: `src/format.cpp`.

```
offset 0    char[8]  magic "TERNIX01"
offset 8    u64      header_size            bytes up to and including the tensor table
offset 16   u32[40]  config slots           see below
            f32[16]  config floats          [0] rms_eps  [1] rope_theta
            tensor table, repeated config[15] times:
              u16 name_len, char name[name_len]
              u32 dtype, u32 ndim, u64 dims[ndim]
              f32 scale          per-tensor weight scale (mean |W|) for ternary tensors, else 1
              u64 offset, u64 nbytes
(64-byte aligned) tensor data, in table order
(64-byte aligned) tokenizer section (offset/size in config slots 16/17)
```

u32 config slots: 0 arch (1 = BitNet block, 2 = Qwen3-style group-scaled block), 1 vocab, 2 hidden, 3 layers, 4 heads, 5 kv_heads, 6 head_dim, 7 ffn, 8 max_pos,
9 tie_embeddings, 10 activation (0 SiLU-GLU, 1 ReLU^2-GLU), 11 sub-norms, 12 bos, 13 eos, 14 eot, 15 n_tensors,
16 tokenizer offset, 17 tokenizer bytes. Architecture 2 adds slots 18 QK-norm, 19 group size (128), 20 activation block (128; 0 = per token),
21 head dtype (0 bf16, 1 f16), 22 add-BOS, 23 digit-group length, 24 ignore-merges. They are read only when arch == 2; arch 1 files keep their defaults.
Architecture 2 also stores `rope.inv_freq` (f32, head_dim/2, precomputed by the converter, YaRN included; the attention factor is float slot 2),
`<weight>.gscale` (f16 `[N][K/128]` per-group scales that replace the per-tensor scale), `q_norm`/`k_norm` weights, and an NFC composition table
in the tokenizer blob.

dtypes: 0 f32, 1 bf16, 2 f16, 3 I2K128 (ternary weights).

## I2K128 ternary matrices

Logical weight `W[n][k]` (n = output row, k = input column, `K % 128 == 0`), values in {-1, 0, +1}. Row `n` occupies `K/4`
bytes. A row is a sequence of 32-byte blocks, one per 128 columns. Within block `b`, byte `l` (0..31) holds the codes of the
four columns `128b + 32j + l`, `j = 0..3`, in bit pair `j` (bits `2j+1:2j`). Code = `w + 1`, so -1 -> 0, 0 -> 1, +1 -> 2
(3 is unused).

Why this layout: one 32-byte load masked with `0x03` (and shifted by 2, 4, 6) gives four 32-lane weight vectors whose
activation partners are four *contiguous* 32-byte loads, so the dot product needs no byte shuffles on either operand. With
codes in {0,1,2} the product is an unsigned x signed byte dot (`vpdpbusd` / `vpmaddubsw`), and
`sum_k q_k w_k = sum_k q_k c_k - sum_k q_k` where the second term is one scalar per activation row.

Group-scaled variant (arch 2): the same 2-bit codes, but the scale is per (row, 128-column block) and the activations are quantised per
128-column block (`deq_step` and the activation sum are per block): `y = sum_b s[n][b] * deq[b] * (sum_{k in b} q_k c_k - sum_{k in b} q_k)`,
accumulated in float lanes.

`y = (sum_k q_k w_k) * deq_step * scale`, with `q = clamp(rint(x * 127/amax), -128, 127)` per token (`amax` floored at 1e-5)
and `deq_step = amax/127`: this is exactly BitLinear's fake-quantised matmul evaluated in integer arithmetic.

## Tokenizer section

```
u32 n_vocab; n_vocab x (u32 len, bytes)          token id order
u32 n_special; n_special x u32 id
u32 n_merges; n_merges x (u32 left_id, u32 right_id)   rank = index
3 x (u32 n_ranges; n_ranges x (u32 lo, u32 hi))        Unicode classes: letters (L*), numbers (N*), white space
```
The Unicode tables come from Python's `unicodedata` at conversion time so the C++ pre-tokenizer needs no Unicode library.
