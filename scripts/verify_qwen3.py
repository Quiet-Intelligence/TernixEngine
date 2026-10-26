"""Check the Qwen3-style (architecture 2, group-scaled ternary) path against Hugging Face transformers.

The reference is the "unpacked" checkpoint (fp16 values {-s,0,+s} per group of 128) run by transformers in float32 with float activations, i.e.
weight-only quantisation. The engine quantises activations to int8 per 128-column block, so the difference measured here is the price of
W1.58A8 on a model that was not trained for A8 (plus float summation noise, see docs/research/EXPERIMENTS.md EXP-001).

    python scripts/verify_qwen3.py --hf models/Ternary-Bonsai-1.7B-unpacked --ternix models/bonsai-1p7b.ternix [--ppl-tokens 4096]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build-dev"))
import ternix_engine as tx  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "<|im_start|>user\nName three prime numbers.<|im_end|>\n<|im_start|>assistant\n",
    "In 1969, Neil Armstrong became the first person to walk on the Moon. " * 4,
]


def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--ternix", required=True)
    ap.add_argument("--text", default="data/wikitext2_test.txt")
    ap.add_argument("--ppl-tokens", type=int, default=4096)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", default="docs/research/data/verify_qwen3.json")
    a = ap.parse_args()
    eng = tx.Engine(a.ternix, gpu=False, ctx=1024)
    res = {"prompts": []}
    m32 = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32).eval()
    for text in PROMPTS:
        ids = eng.encode(text, True, True)
        with torch.no_grad():
            ref = m32(torch.tensor([ids])).logits[0, -1].numpy()
        got = eng.logits(ids)
        lr, lg = log_softmax(ref), log_softmax(got)
        kl = float((np.exp(lr) * (lr - lg)).sum())
        rec = {"n": len(ids), "kl": kl, "max_abs": float(np.abs(ref - got).max()), "top1_equal": bool(ref.argmax() == got.argmax()),
               "cos": float(ref @ got / np.linalg.norm(ref) / np.linalg.norm(got))}
        res["prompts"].append(rec)
        print(rec, flush=True)
    del m32
    # perplexity: identical windows for both (window W tokens, stride W-1, no BOS, every token except the first scored once)
    text = Path(a.text).read_text(encoding="utf-8")
    ids = eng.encode(text[:200000], False, False)[: a.ppl_tokens]
    m16 = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float16).cuda().eval()
    W = a.window
    nll_ref = nll_eng = 0.0
    count = 0
    for s in range(0, len(ids) - 1, W - 1):
        win = ids[s:s + W]
        if len(win) < 2:
            break
        with torch.no_grad():
            lg = m16(torch.tensor([win]).cuda()).logits[0, :-1].float()
        nll_ref += float(torch.nn.functional.cross_entropy(lg, torch.tensor(win[1:]).cuda(), reduction="sum"))
        nll, _ = eng.nll(win[:-1], win[1:])
        nll_eng += float(np.sum(nll))
        count += len(win) - 1
    res["ppl"] = {"tokens_scored": count, "window": W, "hf_fp16_weight_only": float(np.exp(nll_ref / count)), "engine_w158_a8_block128": float(np.exp(nll_eng / count))}
    res["ppl"]["relative_increase"] = res["ppl"]["engine_w158_a8_block128"] / res["ppl"]["hf_fp16_weight_only"] - 1
    print(res["ppl"])
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
