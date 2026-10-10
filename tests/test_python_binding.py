"""Python-binding tests. Run with the build directory on PYTHONPATH (see CI / run_all)."""
import numpy as np
import pytest

ternix = pytest.importorskip("ternix_engine")

CPU_IMPLS = ["naive", "scalar", "avx2", "avx2_blocked"]


def reference(a, w):
    return a.astype(np.int32) @ w.astype(np.int32)


def make(m, k, n, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    w = rng.integers(-1, 2, size=(k, n), dtype=np.int8)
    return a, w


def test_pack_unpack_roundtrip():
    _, w = make(1, 32, 24, 0)
    packed = ternix.pack_ternary(w)
    assert packed.shape == (32, 6) and packed.dtype == np.uint8
    np.testing.assert_array_equal(ternix.unpack_ternary(packed), w)


@pytest.mark.parametrize("impl", CPU_IMPLS)
@pytest.mark.parametrize("shape", [(1, 8, 8), (5, 33, 24), (64, 256, 128)])
def test_cpu_kernels_match_numpy(impl, shape):
    a, w = make(*shape, seed=sum(shape))
    got = ternix.ternary_matmul(a, ternix.pack_ternary(w), impl)
    np.testing.assert_array_equal(got, reference(a, w))


@pytest.mark.skipif(not getattr(ternix, "has_cuda", False), reason="built without CUDA")
@pytest.mark.parametrize("impl", ["cuda_naive", "cuda_tiled"])
def test_cuda_kernels_match_numpy(impl):
    a, w = make(128, 256, 192, seed=3)
    got = ternix.ternary_matmul(a, ternix.pack_ternary(w), impl)
    np.testing.assert_array_equal(got, reference(a, w))


def test_input_validation():
    a, w = make(4, 8, 8, 1)
    with pytest.raises(ValueError):
        ternix.pack_ternary((w * 2).astype(np.int8))          # values outside {-1,0,1}
    with pytest.raises(ValueError):
        ternix.ternary_matmul(a, ternix.pack_ternary(w), "nope")
