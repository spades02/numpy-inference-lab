import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softmax import softmax
from fused import softmax_inplace, softmax_blocked
from harness import benchmark


def gbps(nbytes, ms):
    return nbytes / 1e9 / (ms / 1000)


def main():
    rng = np.random.default_rng(42)
    x64 = rng.normal(size=(64, 50257))
    x32 = x64.astype(np.float32)
    o64 = np.empty_like(x64)
    o32 = np.empty_like(x32)

    rows = [
        ("v0 stable (Day 3)",       lambda: softmax(x64),                 x64.nbytes),
        ("v1 in-place f64",         lambda: softmax_inplace(x64, o64),    x64.nbytes),
        ("v2 in-place f32",         lambda: softmax_inplace(x32, o32),    x32.nbytes),
        ("v3 blocked f32 (8 rows)", lambda: softmax_blocked(x32, o32, 8), x32.nbytes),
    ]

    results = [(name, benchmark(fn)[1], nb) for name, fn, nb in rows]
    base = results[0][1]

    print(f"{'version':<26}{'ms':<10}{'GB/s':<8}{'speedup':<8}")
    for name, ms, nb in results:
        print(f"{name:<26}{ms:<10.3f}{gbps(2 * nb, ms):<8.2f}{base / ms:<.2f}x")


if __name__ == "__main__":
    main()