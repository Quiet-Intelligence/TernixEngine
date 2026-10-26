"""Reference timing of the official Hugging Face / PyTorch path for BitNet b1.58 2B4T (the path the model card warns is not efficient).

    python bench/hf_bench.py --model models/bitnet-b1.58-2B-4T-bf16 --device cuda --dtype bfloat16 --pp 128 --tg 64 --reps 5
Prints one JSON line: {"pp_tps": median, "tg_tps": median, ...}. Greedy decode with the KV cache, fixed token ids, warm-up run excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    ids = torch.tensor([[1000 + (i * 7919) % 50000 for i in range(a.pp)]], device=a.device)
    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    pp, tg = [], []
    with torch.no_grad():
        for r in range(-1, a.reps):
            sync(); t0 = time.perf_counter()
            out = m(ids, use_cache=True)
            sync(); t1 = time.perf_counter()
            past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(a.tg):
                out = m(tok, past_key_values=past, use_cache=True)
                past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            sync(); t2 = time.perf_counter()
            if r >= 0:
                pp.append(a.pp / (t1 - t0)); tg.append(a.tg / (t2 - t1))
    print(json.dumps({"pp_tps": statistics.median(pp), "tg_tps": statistics.median(tg), "pp_all": pp, "tg_all": tg, "device": a.device, "dtype": a.dtype}))


if __name__ == "__main__":
    main()
