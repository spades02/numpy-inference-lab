import numpy as np
import pytest

from softmax import softmax, softmax_naive


def test_known_values():
    np.testing.assert_allclose(
        softmax([1.0, 2.0, 3.0]),
        [0.09003057, 0.24472847, 0.66524096],
        rtol=1e-6,
    )


def test_sums_to_one():
    x = np.random.default_rng(0).normal(size=1000)
    assert np.isclose(softmax(x).sum(), 1.0)


def test_shift_invariance():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(softmax(x), softmax(x + 500.0))


@pytest.mark.parametrize("x", [
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -1001.0, -1002.0],
])
def test_extreme_inputs_are_finite(x):
    p = softmax(x)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, softmax([0.0, 1.0, 2.0]) if x[0] > 0 else softmax([0.0, -1.0, -2.0]))


def test_naive_breaks_on_large_inputs():
    with np.errstate(over="ignore", invalid="ignore"):
        p = softmax_naive([1000.0, 1001.0, 1002.0])
    assert np.any(np.isnan(p))


def test_batch_rows_independent():
    X = np.array([[1.0, 2.0, 3.0],
                  [1000.0, 1000.0, 1000.0]])
    P = softmax(X)
    assert P.shape == X.shape
    np.testing.assert_allclose(P.sum(axis=-1), [1.0, 1.0])
    np.testing.assert_allclose(P[1], [1/3, 1/3, 1/3])
    np.testing.assert_allclose(P[0], softmax(X[0]))