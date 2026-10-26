"""lm-evaluation-harness adapter for the TernixEngine Python binding.

    python bench/lm_eval_ternix.py --model models/bitnet-2b4t.ternix --gpu --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
        --limit 500 --out docs/research/data/lm_eval_ternix.json

Scoring is teacher-forced on the engine (device-side NLL and argmax), so the same weights can be scored with the bf16 head, the
certified int8 head, or the int8 KV cache and the task accuracies compared on identical requests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build-dev"))
import ternix_engine as tx  # noqa: E402

BOS = 128000


class TernixLM(LM):
    def __init__(self, engine: "tx.Engine", max_len: int = 2048):
        super().__init__()
        self.e = engine
        self.max_len = min(max_len, engine.max_ctx)

    def _enc(self, text: str) -> list[int]:
        return self.e.encode(text, False, False)

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False):
        out = []
        for req in requests:
            ctx, cont = req.args
            whole = self._enc(ctx + cont)
            ctx_enc = self._enc(ctx)
            cont_enc = whole[len(ctx_enc):] or self._enc(cont)
            ids = ([BOS] + ctx_enc + cont_enc)[-(self.max_len + 1):]
            n_cont = len(cont_enc)
            tokens, targets = ids[:-1], list(ids[1:])
            for i in range(len(targets) - n_cont):
                targets[i] = -1
            nll, am = self.e.nll(tokens, targets)
            ll = -float(np.sum(nll[len(targets) - n_cont:]))
            greedy = bool(np.all(np.asarray(am[len(targets) - n_cont:]) == np.asarray(ids[1:][len(targets) - n_cont:])))
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False):
        out = []
        for req in requests:
            ids = [BOS] + self._enc(req.args[0])
            total = 0.0
            for s in range(0, len(ids) - 1, self.max_len):
                chunk = ids[s:s + self.max_len + 1]
                nll, _ = self.e.nll(chunk[:-1], chunk[1:])
                total += float(np.sum(nll))
            out.append((-total,))
        return out

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        res = []
        for req in requests:
            ctx, kw = req.args
            until = kw.get("until", [])
            text = self.e.generate(ctx, int(kw.get("max_gen_toks", 128)), 0.0)
            for u in until:
                text = text.split(u)[0]
            res.append(text)
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--head", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--kv-int8", action="store_true")
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,piqa,winogrande")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = tx.Engine(a.model, gpu=a.gpu, ctx=2048, head=a.head, kv_int8=a.kv_int8)
    lm = TernixLM(eng)
    res = evaluator.simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit, num_fewshot=0, bootstrap_iters=2000)
    keep = {t: {k: v for k, v in m.items() if not k.startswith("alias")} for t, m in res["results"].items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": keep, "n_samples": res.get("n-samples")}, indent=1, default=str))
    for t, m in keep.items():
        print(t, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
