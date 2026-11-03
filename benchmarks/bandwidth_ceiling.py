import numpy as np
from harness import benchmark

a = np.random.default_rng(0).normal(size=(64, 50257))
b = np.empty_like(a)
gb = 2 * a.nbytes / 1e9            # one read + one write

_, copy_ms = benchmark(lambda: np.copyto(b, a))
_, exp_ms = benchmark(lambda: np.exp(a, out=b))

print(f"copy:     {gb / (copy_ms / 1000):6.1f} GB/s   (memory ceiling for this pattern)")
print(f"exp only: {gb / (exp_ms / 1000):6.1f} GB/s   (one pass, no temporaries)")


a32 = a.astype(np.float32)
b32 = np.empty_like(a32)
_, exp32_ms = benchmark(lambda: np.exp(a32, out=b32))

print(f"exp f64:  {exp_ms * 1e6 / a.size:5.2f} ns per number")
print(f"exp f32:  {exp32_ms * 1e6 / a.size:5.2f} ns per number")