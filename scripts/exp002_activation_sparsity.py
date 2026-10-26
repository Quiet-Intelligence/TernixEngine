"""EXP-002: exact zeros in the int8 activations of BitNet b1.58 2B4T (ReLU^2 FFN) on real text."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reference_bitnet as rb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "bitnet-b1.58-2B-4T-bf16"

def main():
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    text = (ROOT / "data" / "wikitext2_test.txt").read_text(encoding="utf-8")
    ids = tk.encode(text[:200000]).ids[:2048]
    ref = rb.RefModel(MODEL)
    out = {}
    for w in range(0, 4):   # four windows of 512 tokens
        stats: dict = {}
        ref.logits(ids[w * 512:(w + 1) * 512], False, collect=stats)
        for tag, rows in stats.items():
            out.setdefault(tag, []).append({k: [r[k] for r in rows] for k in rows[0]})
    summary = {}
    for tag, wins in out.items():
        summary[tag] = {k: float(np.mean([np.mean(w[k]) for w in wins])) for k in wins[0]}
        summary[tag]["per_layer_zero"] = [float(np.mean([w["zero"][l] for w in wins])) for l in range(len(wins[0]["zero"]))]
    (ROOT / "docs/research/data/exp002_activation_sparsity.json").write_text(json.dumps(summary, indent=1))
    for tag, v in summary.items():
        print(tag, {k: round(x, 4) for k, x in v.items() if k != "per_layer_zero"})
    print("down_in per-layer zero:", [round(x, 2) for x in summary["down_in"]["per_layer_zero"]])

if __name__ == "__main__":
    main()
