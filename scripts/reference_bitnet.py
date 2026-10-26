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


class RefModel:
    """Float32 reference with ternary weights kept as int8 on the device (2 GB for 2B4T), so a forward pass takes seconds."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)
        self.cfg = cfg = json.loads((model_dir / "config.json").read_text())
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        st = tf.SafeTensors(model_dir / "model.safetensors")
        self.layers = []

        def f32(name):
            a, dt = st.get(name)
            assert dt == "BF16"
            return torch.from_numpy(tf.bf16_to_f32(a).copy()).to(self.dev)

        def lin(name):
            w = f32(name)
            s = 1.0 / w.abs().mean().clamp(min=1e-5)
            return (w * s).round().clamp(-1, 1).to(torch.int8), float(1.0 / s)   # ternary, scale = 1/s as in w_quant

        self.emb = f32("model.embed_tokens.weight")
        self.final_norm = f32("model.norm.weight")
        for l in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{l}."
            self.layers.append({
                **{k: lin(p + n) for k, n in [("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
                                               ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                                               ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
                                               ("down", "mlp.down_proj.weight")]},
                **{k: f32(p + n) for k, n in [("in_norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight"),
                                               ("attn_sub", "self_attn.attn_sub_norm.weight"), ("ffn_sub", "mlp.ffn_sub_norm.weight")]},
            })

    @torch.no_grad()
    def logits(self, ids: list[int], all_positions: bool = True, noise: float = 0.0, seed: int = 0) -> np.ndarray:
        cfg, dev = self.cfg, self.dev
        H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = H // nh
        eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        T = len(ids)
        x = self.emb[torch.tensor(ids, device=dev)]
        if noise:
            g = torch.Generator(device=dev).manual_seed(seed)
            x = x * (1.0 + noise * (torch.rand(x.shape, generator=g, device=dev) * 2 - 1))
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=dev) / hd))
        ang = torch.arange(T, dtype=torch.float32, device=dev)[:, None] * inv[None]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)

        def lin(x, wl):
            w, sc = wl
            return act_quant(x) @ (w.float() * sc).T

        for L in self.layers:
            h = rmsnorm(x, L["in_norm"], eps)
            q = lin(h, L["q"]).view(T, nh, hd)
            k = lin(h, L["k"]).view(T, nkv, hd)
            v = lin(h, L["v"]).view(T, nkv, hd)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            k = k.repeat_interleave(nh // nkv, dim=1)
            v = v.repeat_interleave(nh // nkv, dim=1)
            att = torch.einsum("thd,shd->hts", q, k) / hd ** 0.5 + mask
            o = torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, H)
            x = x + lin(rmsnorm(o, L["attn_sub"], eps), L["o"])
            h = rmsnorm(x, L["post_norm"], eps)
            m = rmsnorm(torch.relu(lin(h, L["gate"])).pow(2) * lin(h, L["up"]), L["ffn_sub"], eps)
            x = x + lin(m, L["down"])
        x = rmsnorm(x, self.final_norm, eps)
        if not all_positions:
            x = x[-1:]
        return (x @ self.emb.T).cpu().numpy()


_CACHE: dict = {}


def reference_logits(model_dir: str | Path, ids: list[int], all_positions: bool = True, **kw) -> np.ndarray:
    key = str(model_dir)
    if key not in _CACHE:
        _CACHE[key] = RefModel(model_dir)
    return _CACHE[key].logits(ids, all_positions, **kw)


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
