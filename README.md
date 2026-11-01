# numpy-inference-lab

Phase 1 of my path to AI inference engineering: the numerical building blocks
of neural networks, written by hand, tested, and benchmarked.

## What's here

- `operations.py`: vector add, elementwise multiply, dot product, magnitude
- `matrix.py`: matrix multiplication in pure Python (with input validation) and NumPy
- `benchmarks/benchmark_matrix.py`: 3 warm-up runs, then 20 timed runs per size
- `tests/`: correctness tests, including rectangular shapes, empty input, and
  incompatible dimensions

## Run it

```
uv sync
uv run python -m pytest -v
uv run python benchmarks/benchmark_matrix.py
```

## Environment

|        |                                                               |
| ------ | ------------------------------------------------------------- |
| CPU    | Intel Core i5-10310U @ 1.70 GHz, 4 cores / 8 threads (laptop) |
| OS     | Windows                                                       |
| Python | 3.12                                                          |
| NumPy  | 2.5.4                                                         |
| BLAS   | scipy-openblas 0.3.34 (Haswell kernels, AVX2)                 |
| Data   | float64, random square matrices, fixed seed                   |

## Results

Minimum of 20 timed runs. One full run per configuration. Python is skipped
above n = 128, where a single call would take seconds to minutes.

| n    | Python (ms) | NumPy (ms) | Speedup | NumPy GFLOP/s | NumPy, 1 thread (ms) |
| ---- | ----------- | ---------- | ------- | ------------- | -------------------- |
| 16   | 0.782       | 0.0061     | 128×    | 1.3           | 0.0056               |
| 32   | 2.830       | 0.0136     | 208×    | 4.8           | 0.0146               |
| 64   | 22.283      | 0.0221     | 1008×   | 23.7          | 0.0285               |
| 128  | 156.087     | 0.2068     | 755×    | 20.3          | 0.1942               |
| 256  | —           | 0.5818     | —       | 57.7          | 1.0096               |
| 512  | —           | 3.0819     | —       | 87.1          | 8.7927               |
| 1024 | —           | 21.3198    | —       | 100.7         | 61.7497              |

GFLOP/s = 2n³ floating-point operations ÷ time. The 1-thread column was
measured with `OPENBLAS_NUM_THREADS=1`.

## Findings

1. **Python's cost per multiply-add settles at about 75–85 ns once n ≥ 32**,
   so its time grows with n³, as the triple loop predicts. At n = 16, fixed
   per-call overhead makes it look more expensive (~190 ns).

2. **NumPy's throughput rises about 75× with problem size**, from 1.3 GFLOP/s
   at n = 16 to 100.7 GFLOP/s at n = 1024. Small matrices can't keep the
   hardware busy: fixed call overhead dominates until there is enough work
   to amortize it.

3. **OpenBLAS multithreading starts paying off between n = 128 and n = 256.**
   At n = 1024, all threads are 2.9× faster than one thread on this
   4-core CPU. At n ≤ 128 the two settings are within noise of each other.
   The 1-thread run looks 6% faster at n = 128, but that is a single run.

4. **Throughput stalls between n = 64 and n = 128** (23.7 → 20.3 GFLOP/s),
   and it stalls with one thread too, so threading alone doesn't explain it.
   Not yet investigated.

5. **Laptop benchmarks are noisy.** Early runs with 5 repeats differed by up to
   4× between identical runs. Taking the minimum of 20 runs after warm-up made
   the results reproducible enough to compare.
