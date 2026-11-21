import numpy as np

from block import layernorm, gelu, ffn, block, init_params

D = 8


def test_layernorm_rows_have_mean_0_var_1():
    x = np.random.default_rng(0).normal(5.0, 3.0, size=(4, D))
    y = layernorm(x, np.ones(D), np.zeros(D))
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, rtol=1e-4)


def test_layernorm_ignores_a_shift_of_the_input():
    x = np.random.default_rng(1).normal(size=(3, D))
    g, b = np.ones(D), np.zeros(D)
    np.testing.assert_allclose(layernorm(x, g, b), layernorm(x + 100.0, g, b), atol=1e-9)


def test_layernorm_applies_gamma_and_beta():
    x = np.random.default_rng(2).normal(size=(2, D))
    gamma = np.full(D, 2.0)
    beta = np.full(D, 0.5)
    plain = layernorm(x, np.ones(D), np.zeros(D))
    np.testing.assert_allclose(layernorm(x, gamma, beta), plain * 2.0 + 0.5)


def test_layernorm_constant_row_is_finite():
    x = np.full((1, D), 3.0)                     # variance is exactly 0
    y = layernorm(x, np.ones(D), np.zeros(D))
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, 0.0)


def test_gelu_known_values():
    np.testing.assert_allclose(gelu(np.array([0.0])), [0.0])
    np.testing.assert_allclose(gelu(np.array([1.0])), [0.841192], rtol=1e-5)
    np.testing.assert_allclose(gelu(np.array([10.0])), [10.0], rtol=1e-6)
    np.testing.assert_allclose(gelu(np.array([-10.0])), [0.0], atol=1e-6)


def test_ffn_shapes():
    rng = np.random.default_rng(3)
    p = init_params(D, 4 * D, rng)
    x = rng.normal(size=(5, D))
    h = ffn(x, p["W1"], p["b1"], p["W2"], p["b2"])
    assert h.shape == (5, D)


def test_block_with_zero_output_weights_is_identity():
    rng = np.random.default_rng(4)
    p = init_params(D, 4 * D, rng)
    p["W2"] = np.zeros_like(p["W2"])             # FFN now outputs exactly b2 = 0
    x = rng.normal(size=(3, D))
    np.testing.assert_allclose(block(x, p), x)