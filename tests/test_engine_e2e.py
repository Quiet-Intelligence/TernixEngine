"""End-to-end engine checks on a converted model (skipped when models/bitnet-2b4t.ternix is absent).
CPU vs GPU agreement, determinism, KV-cache consistency, teacher-forced NLL vs logits, and a factual sanity prompt."""
import os
from pathlib import Path

import numpy as np
import pytest

ternix = pytest.importorskip("ternix_engine")
MODEL = Path(os.environ.get("TERNIX_MODEL", Path(__file__).resolve().parent.parent / "models" / "bitnet-2b4t.ternix"))
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not present")


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


@pytest.fixture(scope="module")
def cpu():
    return ternix.Engine(str(MODEL), gpu=False, ctx=512)


@pytest.fixture(scope="module")
def gpu():
    if not getattr(ternix, "has_cuda", False):
        pytest.skip("built without CUDA")
    return ternix.Engine(str(MODEL), gpu=True, ctx=512)


def test_factual_prompt(cpu):
    assert "Paris" in cpu.generate("The capital of France is", n=6)


def test_deterministic_and_prefix_consistent(cpu):
    ids = cpu.encode("The quick brown fox jumps over the lazy dog.", True, True)
    a = cpu.logits(ids)
    b = cpu.logits(ids)
    np.testing.assert_array_equal(a, b)
    # logits of the last position of a prefix must equal the matching row of the all-positions run
    full = cpu.logits(ids, all=True)
    np.testing.assert_allclose(full[-1], a, rtol=0, atol=2e-3)
    np.testing.assert_allclose(cpu.logits(ids[:5]), full[4], rtol=0, atol=2e-3)


def test_nll_matches_logits(cpu):
    ids = cpu.encode("Ternary weights take the values minus one, zero and plus one.", True, True)
    lg = log_softmax(cpu.logits(ids, all=True))
    nll, am = cpu.nll(ids[:-1], ids[1:])
    ref = -lg[np.arange(len(ids) - 1), ids[1:]]
    np.testing.assert_allclose(nll, ref, atol=2e-3)
    np.testing.assert_array_equal(am, lg[:-1].argmax(-1))


def test_gpu_agrees_with_cpu(cpu, gpu):
    ids = cpu.encode("In 1969, Neil Armstrong became the first person to walk on the Moon.", True, True)
    a, b = cpu.logits(ids), gpu.logits(ids)
    pa, pb = np.exp(log_softmax(a)), np.exp(log_softmax(b))
    kl = float((pa * (np.log(np.maximum(pa, 1e-30)) - log_softmax(b))).sum())
    assert kl < 2.5e-2, kl   # W1.58A8 noise floor, see docs/research/EXPERIMENTS.md EXP-001
    assert a.argmax() == b.argmax()
    assert set(np.argsort(-a)[:3]) == set(np.argsort(-b)[:3])


PROMPTS = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "User: Name three prime numbers.<|eot_id|>Assistant: "]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_int8_head_greedy_is_identical(backend):
    """The certified int8 head must not change a single greedy token (the proof is in include/ternix/head.hpp)."""
    if backend == "gpu" and not getattr(ternix, "has_cuda", False):
        pytest.skip("built without CUDA")
    ref = ternix.Engine(str(MODEL), gpu=backend == "gpu", ctx=512, head="bf16")
    fast = ternix.Engine(str(MODEL), gpu=backend == "gpu", ctx=512, head="int8")
    for p in PROMPTS:
        assert ref.generate(p, n=40) == fast.generate(p, n=40), p


def test_sparse_down_is_bit_identical():
    dense = ternix.Engine(str(MODEL), gpu=False, ctx=512)
    sparse = ternix.Engine(str(MODEL), gpu=False, ctx=512, sparse_down=True)
    ids = dense.encode(PROMPTS[0], True, True)
    a = dense.prefill(ids)
    b = sparse.prefill(ids)
    for _ in range(8):   # decode steps exercise the sparse path (batch 1)
        t = int(a.argmax())
        a, b = dense.step(t), sparse.step(t)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kv_int8", [False, True])
def test_decode_step_matches_prefill(kv_int8):
    """Incremental decode (split-chunk grouped attention) must reproduce the prefill logits of the same prefix."""
    e = ternix.Engine(str(MODEL), gpu=False, ctx=1024, kv_int8=kv_int8)
    text = "In 1969, Neil Armstrong became the first person to walk on the Moon. " * 8   # > 256 tokens: several attention chunks
    ids = e.encode(text, True, True)[:300]
    full = e.logits(ids)
    inc = e.prefill(ids[:-1])
    inc = e.step(ids[-1])
    p, q = np.exp(log_softmax(full)), log_softmax(inc)
    kl = float((p * (np.log(np.maximum(p, 1e-30)) - q)).sum())
    assert kl < (2e-2 if kv_int8 else 5e-3), kl
    assert full.argmax() == inc.argmax()
