# numpy-inference-lab

Phase 1 of my path to AI inference engineering: the numerical building blocks
of neural networks, written by hand, tested, and benchmarked.

## What's here

- `operations.py`: vector add, elementwise multiply, dot product, magnitude
- `matrix.py`: matrix multiplication in pure Python (with input validation) and NumPy
- `softmax.py`: naive and numerically stable softmax, batched over the last axis
- `benchmarks/harness.py`: shared timer, 3 warm-up runs, then 20 timed runs
- `benchmarks/benchmark_matrix.py`: Python vs NumPy matmul, GFLOP/s
- `benchmarks/benchmark_softmax.py`: naive vs stable softmax at GPT-2 vocab size, GB/s
- `benchmarks/bandwidth_ceiling.py`: memory copy ceiling and exp throughput
- `tests/`: correctness tests, including overflow, underflow, batches,
  rectangular shapes, empty input, and incompatible dimensions

## Run it

```
uv sync
uv run python -m pytest -v
uv run python benchmarks/benchmark_matrix.py
uv run python benchmarks/benchmark_softmax.py
uv run python benchmarks/bandwidth_ceiling.py
```

## Environment

|        |                                                                                 |
| ------ | ------------------------------------------------------------------------------- |
| CPU    | Intel Core i5-10310U @ 1.70 GHz, 4 cores / 8 threads (laptop), AVX2, no AVX-512 |
| OS     | Windows                                                                         |
| Python | 3.12                                                                            |
| NumPy  | 2.5.4                                                                           |
| BLAS   | scipy-openblas 0.3.34 (Haswell kernels)                                         |
| Data   | float64 unless noted, random inputs, fixed seed                                 |

All timings are the minimum of 20 timed runs after 3 warm-up runs.

---

## 1. Matrix multiplication

Python is skipped above n = 128, where a single call would take seconds to minutes.

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

**Findings**

1. **Python's cost per multiply-add settles at about 75–85 ns once n ≥ 32**,
   so its time grows with n³, as the triple loop predicts. At n = 16, fixed
   per-call overhead makes it look more expensive (~190 ns).
2. **NumPy's throughput rises about 75× with problem size**, from 1.3 GFLOP/s
   at n = 16 to 100.7 GFLOP/s at n = 1024. Small matrices can't keep the
   hardware busy: fixed call overhead dominates until there is enough work
   to amortize it.
3. **OpenBLAS multithreading starts paying off between n = 128 and n = 256.**
   At n = 1024, all threads are 2.9× faster than one thread on this 4-core
   CPU. At n ≤ 128 the two settings are within noise of each other.
4. **Throughput stalls between n = 64 and n = 128** (23.7 → 20.3 GFLOP/s),
   and it stalls with one thread too, so threading alone doesn't explain it.
   Not yet investigated.
5. **Laptop benchmarks are noisy.** Early runs with 5 repeats differed by up
   to 4× between identical runs. The minimum of 20 runs after warm-up made
   results reproducible enough to compare.

---

## 2. Softmax

Shape `(batch, 50257)`, the vocabulary size of GPT-2: one row of logits per
sequence being generated.

| Batch | Naive (ms) | Stable (ms) | Naive GB/s | Stable GB/s |
| ----- | ---------- | ----------- | ---------- | ----------- |
| 1     | 0.5170     | 0.6313      | 1.45       | 1.19        |
| 8     | 7.1037     | 9.3363      | 0.84       | 0.64        |
| 64    | 49.0254    | 58.3349     | 0.98       | 0.82        |

GB/s counts the minimum memory traffic: read the input once and write the
output once (2 × batch × 50,257 × 8 bytes).

**Hardware ceiling** (shape `(64, 50257)`, separate run):

| Operation                                             | GB/s | ns per number |
| ----------------------------------------------------- | ---- | ------------- |
| `np.copyto` (memory ceiling for one read + one write) | 16.1 | —             |
| `np.exp`, float64, one pass, no temporaries           | 2.6  | 6.10          |
| `np.exp`, float32, one pass, no temporaries           | ~5.8 | 1.37          |

**Findings**

1. **The naive version returns `nan` on large or very negative logits.**
   `exp` overflows to `inf` above x ≈ 709.78 (`inf / inf = nan`) and
   underflows to 0 for very negative inputs (`0 / 0 = nan`). Subtracting each
   row's max before `exp` fixes both without changing the result.
2. **Stability costs 20–30% more time.** Finding the max is an extra full pass
   over every row.
3. **Softmax should be memory-bound but isn't on this CPU.** At about 0.3
   operations per byte (vs ~85 for matmul at n = 1024) it should be limited by
   memory. Yet it reaches only 0.8–1.5 GB/s against a 16.1 GB/s ceiling.
4. **`exp` is the bottleneck.** A single pass of float64 `exp` reaches only
   2.6 GB/s (16% of the ceiling), about 6.1 ns per number on one core. It
   takes roughly half of softmax's time. The other half comes from the extra
   passes: the stable version does about 5 reads and 3 writes of the full array
   (plus 3 new temporary arrays) where the minimum is 1 read and 1 write.
5. **float32 `exp` is 4.5× faster per number than float64** (1.37 vs 6.10 ns).
   This is consistent with NumPy using a vectorized AVX2 path for float32 `exp`
   but needing AVX-512, which this CPU lacks, to vectorize float64 `exp`.
   The data type decides which code path runs.

**Next:** a fused version that makes fewer passes over memory, and float32
throughout.
