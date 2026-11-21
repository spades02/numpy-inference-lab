import numpy as np


def layernorm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    # TODO: mean and variance of each row (last axis, keepdims, as in softmax)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # TODO: return (x - mean) / sqrt(var + eps) * gamma + beta
    return (x - mean) / np.sqrt(variance + eps) * gamma + beta


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pass


def ffn(x, W1, b1, W2, b2):
    # TODO: one line
    return gelu(x @ W1 + b1) @ W2 + b2


def block(x, p):
    return x + ffn(layernorm(x, p["gamma"], p["beta"]), p["W1"], p["b1"], p["W2"], p["b2"])


def init_params(d, hidden, rng):
    """Random weights at GPT-2's starting scale (std 0.02). Not the learning part."""
    return {
        "gamma": np.ones(d), "beta": np.zeros(d),
        "W1": rng.normal(0, 0.02, size=(d, hidden)), "b1": np.zeros(hidden),
        "W2": rng.normal(0, 0.02, size=(hidden, d)), "b2": np.zeros(d),
    }