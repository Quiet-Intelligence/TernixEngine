"""Convert a BitNet b1.58 (bf16 master weights) Hugging Face checkpoint into a .ternix file.

Semantics follow the model's own forward pass (BitLinear, absmean weight quantisation, per-tensor scale):
    scale = mean(|W|);  W_t = clip(round(W / max(scale, 1e-5)), -1, 1)   (round half to even, float32)
    y     = (x_int8 @ W_t^T) * (amax(x) / 127) * scale
Ternary matrices are written as I2K128 (see ternix_format.py). Embeddings stay bf16 (tied lm_head), norms become f32.

    uv run scripts/quantize.py --input models/bitnet-b1.58-2B-4T-bf16 --output models/bitnet-2b4t.ternix
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ternix_format as tf  # noqa: E402

TERNARY_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
               (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def ternarize(w: np.ndarray) -> tuple[np.ndarray, float]:
    w = w.astype(np.float32)
    scale = float(np.abs(w).mean(dtype=np.float64))
    inv = np.float32(1.0 / max(scale, 1e-5))
    t = np.clip(np.rint(w * inv), -1, 1).astype(np.int8)
    return t, scale


def byte_level_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode: byte-level token characters -> raw byte value."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def category_ranges(prefix: str) -> list[tuple[int, int]]:
    out, start = [], None
    for cp in range(0x110000):
        hit = unicodedata.category(chr(cp)).startswith(prefix)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def build_tokenizer_blob(tok_json: dict, vocab_size: int) -> bytes:
    dec = byte_level_decoder()
    vocab = tok_json["model"]["vocab"]
    tokens: list = [None] * vocab_size
    for s, i in vocab.items():
        tokens[i] = bytes(dec[c] for c in s)
    special_ids = []
    for t in tok_json["added_tokens"]:
        tokens[t["id"]] = t["content"].encode("utf-8")
        if t.get("special", True):
            special_ids.append(t["id"])
    assert all(t is not None for t in tokens), "vocabulary has holes"
    merges = []
    for m in tok_json["model"]["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        merges.append((vocab[a], vocab[b]))

    blob = bytearray()
    blob += struct.pack("<I", vocab_size)
    for t in tokens:
        blob += struct.pack("<I", len(t)) + t
    blob += struct.pack("<I", len(special_ids)) + struct.pack(f"<{len(special_ids)}I", *special_ids)
    blob += struct.pack("<I", len(merges))
    blob += np.asarray(merges, dtype="<u4").tobytes()
    for ranges in (category_ranges("L"), category_ranges("N"), WHITE_SPACE):
        blob += struct.pack("<I", len(ranges)) + np.asarray(ranges, dtype="<u4").tobytes()
    if (tok_json.get("normalizer") or {}).get("type") == "NFC":   # Qwen: canonical composition of (base, combining) pairs
        comp = []
        for cp in range(0x110000):
            d = unicodedata.decomposition(chr(cp))
            parts = d.split()
            if d and not d.startswith("<") and len(parts) == 2:
                a, b = int(parts[0], 16), int(parts[1], 16)
                if unicodedata.normalize("NFC", chr(a) + chr(b)) == chr(cp):
                    comp.append((a, b, cp))
        blob += struct.pack("<I", len(comp)) + np.asarray(comp, dtype="<u4").tobytes()
    return bytes(blob)


def ternarize_groups(w: np.ndarray, group: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """w [N, K] with values in {-s_g, 0, +s_g} per group of `group` columns (an "unpacked" group-scaled ternary checkpoint).
    Returns int8 ternary [N, K] and fp16 scales [N, K/group]. The scale is the mean magnitude of the nonzero entries (the L2-optimal
    value; a few groups in published checkpoints differ by up to 0.8% between elements because of fp16 rounding upstream).
    Raises if a group is not ternary within 2% of its scale."""
    n, k = w.shape
    g = w.astype(np.float32).reshape(n, k // group, group)
    mx = np.abs(g).max(-1, keepdims=True)
    nz = np.abs(g) > 0.5 * mx
    cnt = np.maximum(nz.sum(-1, keepdims=True), 1)
    s = (np.abs(g) * nz).sum(-1, keepdims=True) / cnt
    t = (np.sign(g) * nz).astype(np.int8)
    err = float((np.abs(t * s - g).max(-1) / np.maximum(mx[..., 0], 1e-30)).max())
    if err > 2e-2:
        raise SystemExit(f"weights are not ternary per group of {group} (max relative error {err:.2e})")
    return t.reshape(n, k), s.reshape(n, k // group).astype(np.float16)


def convert_qwen3(src: Path, cfg: dict, out: Path, stats_path: str | None) -> None:
    """Qwen3-style dense decoder with group-128 scaled ternary linears (Ternary-Bonsai "unpacked" checkpoints)."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    st = tf.SafeTensors(src / "model.safetensors")
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    hf_cfg = AutoConfig.from_pretrained(str(src))
    rope_type = (cfg.get("rope_scaling") or {}).get("rope_type", "default")
    inv_freq, attn_factor = ROPE_INIT_FUNCTIONS[rope_type](hf_cfg, "cpu")
    tok_json = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    eos = cfg["eos_token_id"] if isinstance(cfg["eos_token_id"], int) else cfg["eos_token_id"][0]
    u32 = {
        tf.C_ARCH: 2, tf.C_VOCAB: cfg["vocab_size"], tf.C_HIDDEN: cfg["hidden_size"], tf.C_LAYERS: cfg["num_hidden_layers"],
        tf.C_HEADS: cfg["num_attention_heads"], tf.C_KV_HEADS: cfg["num_key_value_heads"], tf.C_HEAD_DIM: hd, tf.C_FFN: cfg["intermediate_size"],
        tf.C_MAX_POS: cfg["max_position_embeddings"], tf.C_TIE: int(cfg.get("tie_word_embeddings", True)), tf.C_ACT: 0, tf.C_SUBLN: 0,
        tf.C_BOS: 0, tf.C_EOS: eos, tf.C_EOT: eos, tf.C_QKNORM: 1, tf.C_GROUP: 128, tf.C_ACTBLOCK: 1, tf.C_HEAD_DTYPE: 2, tf.C_ADD_BOS: 0,
        tf.C_DIGITS: 1, tf.C_IGNORE_MERGES: int(bool(tok_json["model"].get("ignore_merges", False))),
    }
    f32 = {tf.F_EPS: cfg["rms_norm_eps"], tf.F_ROPE_THETA: cfg["rope_theta"], tf.F_ROPE_ATTN: float(attn_factor)}
    tensors, stats = [], {}
    tensors.append(("rope.inv_freq", tf.DT_F32, (len(inv_freq),), 1.0, np.asarray(inv_freq, dtype=np.float32)))
    for name in sorted(st.names(), key=lambda n: (("layers" in n), n)):
        arr, dt = st.get(name)
        assert dt == "F16", f"{name}: expected F16, got {dt}"
        if name == "model.embed_tokens.weight":
            tensors.append((name, tf.DT_F16, arr.shape, 1.0, np.ascontiguousarray(arr)))
        elif name.rsplit(".", 2)[-2] in TERNARY_SUFFIXES:
            t, gs = ternarize_groups(arr)
            tensors.append((name, tf.DT_I2K128, t.shape, 1.0, tf.pack_i2k128(t)))
            tensors.append((name + ".gscale", tf.DT_F16, gs.shape, 1.0, np.ascontiguousarray(gs)))
            stats[name] = {"shape": list(t.shape), "frac_zero": float((t == 0).mean())}
        else:   # RMSNorm weights, including per-head q_norm / k_norm
            tensors.append((name, tf.DT_F32, arr.shape, 1.0, arr.astype(np.float32)))
    tok = build_tokenizer_blob(tok_json, cfg["vocab_size"])
    tf.write_ternix(out, u32, f32, tensors, tok)
    zeros = np.mean([v["frac_zero"] for v in stats.values()])
    print(f"wrote {out}: {len(tensors)} tensors, {out.stat().st_size / 2**20:.1f} MiB, mean zero fraction {zeros:.3f}, rope {rope_type} attn_factor {float(attn_factor):.4f}")
    if stats_path:
        Path(stats_path).write_text(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory with model.safetensors, config.json, tokenizer.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats", help="write per-tensor ternary statistics (JSON)")
    args = ap.parse_args()
    src = Path(args.input)
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") == "qwen3":
        convert_qwen3(src, cfg, Path(args.output), args.stats)
        return
    st = tf.SafeTensors(src / "model.safetensors")

    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    u32 = {
        tf.C_ARCH: 1, tf.C_VOCAB: cfg["vocab_size"], tf.C_HIDDEN: cfg["hidden_size"], tf.C_LAYERS: cfg["num_hidden_layers"],
        tf.C_HEADS: cfg["num_attention_heads"], tf.C_KV_HEADS: cfg["num_key_value_heads"], tf.C_HEAD_DIM: head_dim,
        tf.C_FFN: cfg["intermediate_size"], tf.C_MAX_POS: cfg["max_position_embeddings"],
        tf.C_TIE: int(cfg.get("tie_word_embeddings", True)), tf.C_ACT: 1 if cfg["hidden_act"] == "relu2" else 0,
        tf.C_SUBLN: 1, tf.C_BOS: cfg["bos_token_id"], tf.C_EOS: cfg["eos_token_id"], tf.C_EOT: 128009,
    }
    f32 = {tf.F_EPS: cfg["rms_norm_eps"], tf.F_ROPE_THETA: cfg["rope_theta"]}

    tensors, stats = [], {}
    for name in sorted(st.names(), key=lambda n: (("layers" in n), n)):
        arr, dt = st.get(name)
        assert dt == "BF16", f"{name}: expected BF16, got {dt}"
        if name == "model.embed_tokens.weight":
            tensors.append((name, tf.DT_BF16, arr.shape, 1.0, np.ascontiguousarray(arr)))
        elif name.rsplit(".", 2)[-2] in TERNARY_SUFFIXES:
            w = tf.bf16_to_f32(arr)
            t, scale = ternarize(w)
            packed = tf.pack_i2k128(t)
            assert (tf.unpack_i2k128(packed, t.shape[1]) == t).all()
            tensors.append((name, tf.DT_I2K128, t.shape, scale, packed))
            stats[name] = {"shape": list(t.shape), "scale": scale, "frac_zero": float((t == 0).mean()),
                           "frac_pos": float((t == 1).mean()), "frac_neg": float((t == -1).mean())}
        else:  # RMSNorm weights
            tensors.append((name, tf.DT_F32, arr.shape, 1.0, tf.bf16_to_f32(arr).copy()))
    tok = build_tokenizer_blob(json.loads((src / "tokenizer.json").read_text(encoding="utf-8")), cfg["vocab_size"])
    tf.write_ternix(Path(args.output), u32, f32, tensors, tok)
    zeros = np.mean([s["frac_zero"] for s in stats.values()])
    print(f"wrote {args.output}: {len(tensors)} tensors, {Path(args.output).stat().st_size / 2**20:.1f} MiB, "
          f"mean zero fraction {zeros:.3f}")
    if args.stats:
        Path(args.stats).write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
