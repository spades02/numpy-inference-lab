from harness import benchmark
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from matrix import matmul, matmul_numpy

def main():
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    print(
        f"{'Size':<10}"
        f"{'Python (ms)':<18}"
        f"{'NumPy (ms)':<18}"
        f"{'Speedup':<12}"
    )
    for n in sizes:
        rng = np.random.default_rng(42)

        A = rng.random((n, n))
        B = rng.random((n, n))

        numpy_med, numpy_min = benchmark(
            lambda: matmul_numpy(A, B)
        )
        if n <= 128:
            A_list = A.tolist()
            B_list = B.tolist()
        
            py_med, py_min = benchmark(
                lambda: matmul(A_list, B_list)
            )
            py_text = f"{py_min:.4f}"
            speedup_text = f"{py_min / numpy_min:.2f}x"
        else:
            py_text = "—"
            speedup_text = "—"

        print(f"{n:<10}{py_text:<18}{numpy_min:<18.4f}{speedup_text:<12}")


if __name__ == "__main__":
    main()
