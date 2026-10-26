"""EXP-001: numerical noise floor of W1.58A8 logits (see docs/research/EXPERIMENTS.md).

Perturbs the embedded input of the float32 reference by relative noise and measures the logit KL against the unperturbed run;
also reports engine-vs-reference and CPU-vs-GPU for the same prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reference_bitnet as rb  # noqa: E402
from verify_logits import PROMPTS, kl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "bitnet-b1.58-2B-4T-bf16"


def stats(a, b):
    return {"kl": kl(a, b), "max_abs": float(np.abs(a - b).max()), "top1_equal": bool(a.argmax() == b.argmax())}


def main():
    from tokenizers import Tokenizer
    import ternix_engine as tx
    tk = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    ref = rb.RefModel(MODEL)
    cpu = tx.Engine(str(ROOT / "models" / "bitnet-2b4t.ternix"), gpu=False, ctx=512)
    gpu = tx.Engine(str(ROOT / "models" / "bitnet-2b4t.ternix"), gpu=True, ctx=512)
    out = {"prompts": []}
    for text in PROMPTS:
        ids = tk.encode(text).ids
        base = ref.logits(ids, False)[-1]
        rec = {"n_tokens": len(ids), "noise": {}, "engine": {}}
        for amp in (2.0 ** -24, 2.0 ** -16):
            rec["noise"][f"{amp:.3g}"] = [stats(base, ref.logits(ids, False, noise=amp, seed=s)[-1]) for s in range(3)]
        e_cpu, e_gpu = cpu.logits(ids), gpu.logits(ids)
        rec["engine"] = {"cpu_vs_ref": stats(base, e_cpu), "gpu_vs_ref": stats(base, e_gpu), "cpu_vs_gpu": stats(e_cpu, e_gpu)}
        out["prompts"].append(rec)
        print(json.dumps({"n": len(ids), **{k: [round(x["kl"], 6) for x in v] for k, v in rec["noise"].items()},
                          **{k: round(v["kl"], 6) for k, v in rec["engine"].items()}}), flush=True)
    Path(ROOT / "docs/research/data/exp001_noise_floor.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
