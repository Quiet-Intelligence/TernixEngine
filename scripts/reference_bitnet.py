"""Independent float32 PyTorch reference for BitNet b1.58, used to verify the C++ engine.

Written from the BitLinear definition (absmean ternary weights, per-token absmax int8 activations, sub-norms, ReLU^2
gated FFN, RoPE rotate-half, GQA, tied head), NOT by importing the Hugging Face modelling code, so that an error shared
by both implementations is unlikely. `hf_logits` additionally runs the Hugging Face implementation for a three-way check.

    reference_logits(model_dir, token_ids) -> float32 [n_tokens, vocab]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ternix_format as tf  # noqa: E402


def act_quant(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    s = 127.0 / amax
    return (x * s).round().clamp(-128, 127) / s


def w_quant(w: torch.Tensor) -> torch.Tensor:
    s = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * s).round().clamp(-1, 1) / s


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, cos, sin):  # x [T, H, D]; rotate-half
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@torch.no_grad()
def reference_logits(model_dir: str | Path, ids: list[int], all_positions: bool = True) -> np.ndarray:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    st = tf.SafeTensors(model_dir / "model.safetensors")

    def W(name):
        a, dt = st.get(name)
        assert dt == "BF16"
        return torch.from_numpy(tf.bf16_to_f32(a).copy())

    H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    T = len(ids)
    emb = W("model.embed_tokens.weight")
    x = emb[torch.tensor(ids)]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

    def lin(x, name):
        return act_quant(x) @ w_quant(W(name)).T

    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rmsnorm(x, W(p + "input_layernorm.weight"), eps)
        q = lin(h, p + "self_attn.q_proj.weight").view(T, nh, hd)
        k = lin(h, p + "self_attn.k_proj.weight").view(T, nkv, hd)
        v = lin(h, p + "self_attn.v_proj.weight").view(T, nkv, hd)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        k = k.repeat_interleave(nh // nkv, dim=1)
        v = v.repeat_interleave(nh // nkv, dim=1)
        att = torch.einsum("thd,shd->hts", q, k) / hd ** 0.5 + mask
        o = torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, H)
        o = rmsnorm(o, W(p + "self_attn.attn_sub_norm.weight"), eps)
        x = x + lin(o, p + "self_attn.o_proj.weight")
        h = rmsnorm(x, W(p + "post_attention_layernorm.weight"), eps)
        g = lin(h, p + "mlp.gate_proj.weight")
        u = lin(h, p + "mlp.up_proj.weight")
        m = rmsnorm(torch.relu(g).pow(2) * u, W(p + "mlp.ffn_sub_norm.weight"), eps)
        x = x + lin(m, p + "mlp.down_proj.weight")
    x = rmsnorm(x, W("model.norm.weight"), eps)
    if not all_positions:
        x = x[-1:]
    return (x @ emb.T).numpy()


@torch.no_grad()
def hf_logits(model_dir: str | Path, ids: list[int], dtype=torch.float32) -> np.ndarray:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    m.eval()
    return m(torch.tensor([ids])).logits[0].float().numpy()


if __name__ == "__main__":
    mdir = sys.argv[1]
    ids = [int(t) for t in sys.argv[2:]]
    lg = reference_logits(mdir, ids)
    print(lg.shape, lg[-1].argmax())
