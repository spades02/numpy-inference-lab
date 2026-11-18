import numpy as np
from harness import benchmark

def gb_per_s(nbytes, ms):
    s = ms / 1000
    gb = nbytes / 1e9
    return gb/s
    pass


def main():
    rng = np.random.default_rng(42)

    print(f"{'n':<8}{'copy (ms)':<12}{'T.copy (ms)':<14}"
          f"{'copy GB/s':<12}{'T.copy GB/s':<14}{'slower by':<10}")

    for n in [1024, 4096]:
        # TODO 1: make the matrix a
        a = rng.random((n, n))
        # TODO 2: time a.copy() and a.T.copy()
        _, copy = benchmark(lambda: a.copy())
        _, T_copy = benchmark(lambda: a.T.copy())
        nbytes = 2 * a.nbytes
        copy_gbps = gb_per_s(nbytes, copy)
        T_copy_gbps = gb_per_s(nbytes, T_copy)
        slower_by = T_copy / copy 

        # TODO 4: print one row
        print(f"{n:<8}{copy:<12.4}{T_copy:<14.4}"
              f"{copy_gbps:<12.4}{T_copy_gbps:<14.4}{slower_by:<10.4}")
        pass

def exp_B():
    rng = np.random.default_rng(42)
    count = 2**19                             # always sum this many values
    x = rng.random(count * 64)                # ~268 MB; build once, outside the timer
    
    print(f"{'stride':<8}{'ms':<12}{'ns':<14}")
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        v = x[:count * stride:stride]         # exactly `count` values, `stride` apart
        # TODO: time v.sum(), then print stride, ms, and ns per value
        _, ms = benchmark(lambda: v.sum())
        ns = ms * 1e6 / count
        print(f"{stride:<8}{ms:<12.4}{ns:<14.4}")
if __name__ == "__main__":
    main()
    exp_B()