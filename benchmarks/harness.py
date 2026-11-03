import time
import statistics
def benchmark(fn, repeats=20):

    # Warm-up
    for _ in range(3):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000, min(times) * 1000
