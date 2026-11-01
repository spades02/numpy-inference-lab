# numpy-inference-lab

Phase 1 of my path to AI inference engineering: numerical building blocks of
neural networks, written by hand, tested, and benchmarked.

## What's here

- `operations.py`: vector add, multiply, dot product, magnitude
- `matrix.py`: matrix multiplication in pure Python and NumPy
- `benchmarks/benchmark_matrix.py`: 3 warm-ups, 20 timed runs, median and min

## Run it

    uv sync
    uv run python -m pytest -v
    uv run python benchmarks/benchmark_matrix.py

## Results (Windows laptop, float64, min of 20 runs)

| n    | Python (ms) | NumPy (ms) | NumPy, 1 thread (ms) |
| ---- | ----------- | ---------- | -------------------- |
| 64   | 22.28       | 0.022      | 0.029                |
| 128  | 156.09      | 0.207      | 0.194                |
| 256  | —           | 0.582      | 1.010                |
| 1024 | —           | 21.320     | 61.750               |

## Findings

- Pure Python costs ~75 ns per multiply-add, constant across sizes, so time
  grows with n³ as expected.
- NumPy's throughput rises from ~1.7 GFLOP/s at n=16 to ~100 GFLOP/s at n=1024:
  small problems can't keep the hardware busy.
- OpenBLAS multithreading starts paying off between n=128 and n=256 on this CPU
  (2.9× faster than 1 thread at n=1024).
- First runs varied up to 4× between identical runs; reporting median and min
  over 20 runs fixed it.
