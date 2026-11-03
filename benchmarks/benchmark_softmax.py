import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from softmax import softmax, softmax_naive
from harness import benchmark

VOCAB = 50257


def gb_per_s(batch, ms):
    s = ms / 1000
    bytes_moved =  2 * batch * VOCAB * 8
    gb = bytes_moved / 1024 ** 3
    return gb/s


def main():
    rng = np.random.default_rng(42)

    print(f"{'Batch':<8}{'Naive (ms)':<14}{'Stable (ms)':<14}"
          f"{'Naive GB/s':<13}{'Stable GB/s':<13}")

    for batch in [1, 8, 64]:
        logits = rng.normal(size = (batch, VOCAB))
        _, naive_min = benchmark(lambda: softmax_naive(logits))
        _, stable_min = benchmark(lambda:softmax(logits))
        naive_gbps = gb_per_s(batch, naive_min)
        stable_gbps = gb_per_s(batch, stable_min)
        print(
            f"{batch:<8}{naive_min:<14.4f}{stable_min:<14.4f}{naive_gbps:<13.2f}{stable_gbps:<13.2f}"
        )
        


if __name__ == "__main__":
    main()