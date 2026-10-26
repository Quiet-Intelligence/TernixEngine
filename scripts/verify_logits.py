"""Three-way logit check: C++ engine vs independent float32 reference (and optionally Hugging Face).

    uv run scripts/verify_logits.py --model models/bitnet-b1.58-2B-4T-bf16 --ternix models/bitnet-2b4t.ternix [--hf]

Reports per prompt: max |dlogit|, cosine similarity, KL(ref || engine), top-1 agreement and top-5 overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reference_bitnet as rb  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "User: How many legs does a spider have?<|eot_id|>Assistant: ",
    "Die schnelle braune Fuchs springt über den faulen Hund. 今天天气很好，",
    "In 1969, Neil Armstrong became the first person to walk on the Moon. " * 6,
]


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = np.exp(p_logits - p_logits.max()); p /= p.sum()
    lq = q_logits - q_logits.max(); lq = lq - np.log(np.exp(lq).sum())
    lp = np.log(np.maximum(p, 1e-30))
    return float((p * (lp - lq)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ternix", required=True)
    ap.add_argument("--cli", default="build/ternix_cli.exe")
    ap.add_argument("--hf", action="store_true", help="also compare the Hugging Face implementation (float32)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--isa", default="")
    ap.add_argument("--gpu", action="store_true", help="run the CUDA engine")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    worst = 0
    for text in PROMPTS:
        ids = tk.encode(text).ids
        cache = Path(args.model).parent / "ref_cache" / (hashlib.sha1(("ref" + str(ids)).encode()).hexdigest()[:16] + ".npy")
        if cache.exists():
            ref = np.load(cache)
        else:
            ref = rb.reference_logits(args.model, ids, all_positions=False)[-1]
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, ref)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "l.f32"
            cmd = [str(Path(args.cli).resolve()), "logits", args.ternix, "--ids", " ".join(map(str, ids)), "--out", str(out), "--threads", str(args.threads)]
            if args.isa:
                cmd += ["--isa", args.isa]
            if args.gpu:
                cmd += ["--gpu"]
            subprocess.run(cmd, check=True, capture_output=True)
            eng = np.fromfile(out, dtype=np.float32)
        cos = float(ref @ eng / (np.linalg.norm(ref) * np.linalg.norm(eng)))
        t5r, t5e = set(np.argsort(-ref)[:5]), set(np.argsort(-eng)[:5])
        line = (f"{len(ids):4d} tok | max|d| {np.abs(ref - eng).max():8.4f} | cos {cos:.7f} | KL {kl(ref, eng):.2e} | "
                f"top1 {'ok' if ref.argmax() == eng.argmax() else 'DIFF'} | top5 overlap {len(t5r & t5e)}/5")
        if args.hf:
            hf = rb.hf_logits(args.model, ids)[-1]
            line += f" || HF vs ref: max|d| {np.abs(ref - hf).max():.4f} KL {kl(ref, hf):.2e} top1 {'ok' if ref.argmax() == hf.argmax() else 'DIFF'}"
        print(line, flush=True)
        worst = max(worst, kl(ref, eng))
    print(f"worst KL(ref||engine) = {worst:.3e}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
