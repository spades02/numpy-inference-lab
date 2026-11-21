import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from block import layernorm, gelu, init_params
from harness import benchmark

D, HIDDEN = 768, 3072


def main():
    rng = np.random.default_rng(42)
    p = init_params(D, HIDDEN, rng)

    print(f"{'tokens':<8}{'LN ms':<10}{'matmul1 ms':<12}{'GELU ms':<10}"
          f"{'matmul2 ms':<12}{'total ms':<10}{'matmul %':<10}{'GFLOP/s':<8}")

    for T in [1, 8, 64, 512]:
        x = rng.normal(size=(T, D))

        # Build each stage's INPUT outside the timer, so each timing covers one stage only
        xn = layernorm(x, p["gamma"], p["beta"])
        h = xn @ p["W1"] + p["b1"]
        g = gelu(h)

        # TODO 1: time the 4 stages (always: _, name_ms = benchmark(lambda: ...))
        #   LN:      layernorm(x, p["gamma"], p["beta"])
        _, LN = benchmark(lambda: layernorm(x, p["gamma"], p["beta"]))
        #   matmul1: xn @ p["W1"] + p["b1"]
        _, matmul1 = benchmark(lambda: xn @ p["W1"] + p["b1"])
        #   GELU:    gelu(h)
        _, GELU = benchmark(lambda: gelu(h))
        #   matmul2: g @ p["W2"] + p["b2"]
        _, matmul2 = benchmark(lambda: g @ p["W2"] + p["b2"])

        # TODO 2: total = sum of the 4;  matmul % = 100 * (matmul1 + matmul2) / total
        total = LN + matmul1 + matmul2 + GELU
        # total = np.sum(LN, matmul1, GELU, matmul2)
        matmul_perc = 100 * (matmul1 + matmul2) / total
        # TODO 3: GFLOP/s of the two matmuls:
        #   flops = 2 * T * D * HIDDEN * 2      (2 per multiply-add, 2 matmuls)
        flops = 2 * T * D *  HIDDEN * 2
        #   GFLOP/s = flops / 1e9 / (seconds spent in matmul1 + matmul2)
        GFLOPS = flops / 1e9 / ((matmul1 + matmul2) / 1000)
        # TODO 4: print one row (.3f for ms, .0f for %, .1f for GFLOP/s)
        print(f"{T:<8}{LN:<10.3f}{matmul1:<12.3f}{GELU:<8.3f}"
              f"{matmul2:<12.3f}{total:<10.3f}{matmul_perc:<10.0f}{GFLOPS:<8.1f}")


if __name__ == "__main__":
    main()