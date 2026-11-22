import numpy as np


def softmax_inplace(x, out=None):
    """Stable softmax with no temporary arrays: every step writes into `out`."""
    x = np.asarray(x)
    if out is None:
        out = np.empty_like(x)
    m = np.max(x, axis=-1, keepdims=True)
    np.subtract(x, m, out = out)
    np.exp(out, out=out)
    s = np.sum(out, axis=-1, keepdims=True)
    np.divide(out, s, out = out)
    return out


def softmax_blocked(x, out=None, rows=8):
    """Same, but a few rows at a time, so each block stays in cache across all steps."""
    x = np.asarray(x)
    if out is None:
        out = np.empty_like(x)
    for r in range(0, x.shape[0], rows):
        softmax_inplace(x[r:r+rows], out=out[r:r+rows])
    return out