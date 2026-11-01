import numpy as np

def softmax_naive(x):
    a = np.asarray(x, dtype=np.float64)
    e = np.exp(a)
    total = np.sum(e, axis=-1, keepdims=True)
    y = e/total
    return y

def softmax(x):
    a = np.asarray(x, dtype=np.float64)
    c = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - c)
    total = np.sum(e, axis=-1, keepdims=True)
    y = e/total
    return y
