"""Architecture 2 (Qwen3-style, group-scaled ternary) checks on models/bonsai-1p7b.ternix (skipped when absent).
The numeric reference comparison against transformers lives in scripts/verify_qwen3.py (EXP-009)."""
import os
from pathlib import Path

import numpy as np
import pytest

ternix = pytest.importorskip("ternix_engine")
MODEL = Path(os.environ.get("TERNIX_QWEN3_MODEL", Path(__file__).resolve().parent.parent / "models" / "bonsai-1p7b.ternix"))
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not present")


@pytest.fixture(scope="module")
def eng():
    return ternix.Engine(str(MODEL), gpu=False, ctx=256)


def test_no_bos_added(eng):
    ids = eng.encode("Hello world", True, True)
    assert ids == eng.encode("Hello world", False, False)


def test_factual_prompt(eng):
    assert "Paris" in eng.generate("The capital of France is", n=6)


def test_prefix_consistent_and_deterministic(eng):
    ids = eng.encode("The quick brown fox jumps over the lazy dog.", True, True)
    a = eng.logits(ids)
    np.testing.assert_array_equal(a, eng.logits(ids))
    assert a.shape[-1] == 151669


def test_nll_matches_logits(eng):
    ids = eng.encode("In 1969, Neil Armstrong walked on the Moon.", True, True)
    lg = eng.logits(ids[:-1]).astype(np.float64)
    lg -= lg.max()
    expect = -(lg[ids[-1]] - np.log(np.exp(lg).sum()))
    nll, _ = eng.nll(ids[:-1], ids[1:])
    assert abs(float(np.asarray(nll)[-1]) - expect) < 2e-3


@pytest.fixture(scope="module")
def gpu():
    if not getattr(ternix, "has_cuda", False):
        pytest.skip("built without CUDA")
    return ternix.Engine(str(MODEL), gpu=True, ctx=256)


def test_gpu_matches_cpu(eng, gpu):
    for text in ["The capital of France is", "In 1969, Neil Armstrong became the first person to walk on the Moon. " * 5]:
        ids = eng.encode(text, True, True)
        a, b = eng.logits(ids), gpu.logits(ids)
        assert a.argmax() == b.argmax()
        la, lb = a - a.max(), b - b.max()
        la -= np.log(np.exp(la).sum())
        lb -= np.log(np.exp(lb).sum())
        assert float((np.exp(la) * (la - lb)).sum()) < 1e-3


def test_gpu_int8_head_is_exact():
    if not getattr(ternix, "has_cuda", False):
        pytest.skip("built without CUDA")
    a = ternix.Engine(str(MODEL), gpu=True, ctx=256, head="bf16")
    b = ternix.Engine(str(MODEL), gpu=True, ctx=256, head="int8")
    for p in ["Explain how a transformer language model works.", "The history of the Roman Empire began"]:
        assert a.generate(p, n=64) == b.generate(p, n=64)
