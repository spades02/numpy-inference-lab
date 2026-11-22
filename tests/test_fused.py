import numpy as np
import pytest

from softmax import softmax
from fused import softmax_inplace, softmax_blocked

VERSIONS = [softmax_inplace, softmax_blocked]


@pytest.mark.parametrize("fn", VERSIONS)
def test_matches_reference_float64(fn):
    x = np.random.default_rng(0).normal(size=(20, 1000))
    np.testing.assert_allclose(fn(x), softmax(x), rtol=1e-12)


@pytest.mark.parametrize("fn", VERSIONS)
def test_matches_reference_float32(fn):
    x = np.random.default_rng(1).normal(size=(20, 1000)).astype(np.float32)
    y = fn(x)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, softmax(x.astype(np.float64)), rtol=1e-5)


@pytest.mark.parametrize("fn", VERSIONS)
def test_extreme_inputs_are_finite(fn):
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    y = fn(x)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y.sum(axis=-1), [1.0, 1.0])


def test_writes_into_given_buffer():
    x = np.random.default_rng(2).normal(size=(4, 10))
    out = np.empty_like(x)
    y = softmax_inplace(x, out=out)
    assert y is out


def test_blocked_handles_rows_not_divisible_by_block():
    x = np.random.default_rng(3).normal(size=(13, 50))
    np.testing.assert_allclose(softmax_blocked(x, rows=4), softmax(x), rtol=1e-12)