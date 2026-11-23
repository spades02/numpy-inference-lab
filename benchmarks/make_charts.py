"""Draw the README charts from the measured results (numbers copied from README.md)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, AMBER, RED, GREEN, GREY = "#3557d6", "#d9930b", "#c23a2b", "#1d8a5b", "#6b7785"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Matmul throughput -------------------------------------------------------
n = [16, 32, 64, 128, 256, 512, 1024]
ms_all = [0.0061, 0.0136, 0.0221, 0.2068, 0.5818, 3.0819, 21.3198]
ms_one = [0.0056, 0.0146, 0.0285, 0.1942, 1.0096, 8.7927, 61.7497]
gf = lambda size, ms: 2 * size**3 / (ms / 1000) / 1e9

fig, ax = plt.subplots(figsize=(7, 3.8))
x = range(len(n))
ax.bar(x, [gf(s, m) for s, m in zip(n, ms_all)], color=BLUE, width=0.6, label="all threads")
ax.plot(x, [gf(s, m) for s, m in zip(n, ms_one)], color=AMBER, marker="o", lw=2, label="1 thread")
for i, (s, m, m1) in enumerate(zip(n, ms_all, ms_one)):
    top = max(gf(s, m), gf(s, m1)) if i < 4 else gf(s, m)
    ax.text(i, top + 2, f"{gf(s, m):.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x), [str(v) for v in n])
ax.set_xlabel("matrix size n")
ax.set_ylabel("GFLOP/s")
ax.set_title("NumPy matmul: small matrices waste the hardware")
ax.legend(frameon=False, loc="upper left")
save(fig, "matmul_gflops.png")

# 2. Softmax versions --------------------------------------------------------
names = ["v0  stable, float64", "v1  in-place, float64", "v2  in-place, float32", "v3  blocked, float32"]
ms = [42.213, 29.627, 10.231, 7.915]
fig, ax = plt.subplots(figsize=(7, 3.2))
y = range(len(names))[::-1]
ax.barh(list(y), ms, color=[GREY, AMBER, BLUE, GREEN], height=0.6)
for yi, m in zip(y, ms):
    ax.text(m + 0.6, yi, f"{m:.1f} ms   {ms[0] / m:.2f}×", va="center", fontsize=9)
ax.set_yticks(list(y), names)
ax.set_xlim(0, 54)
ax.set_xlabel("time for a (64, 50257) softmax, ms (lower is better)")
ax.set_title("Softmax, one change at a time: 5.3× faster, same math")
ax.grid(axis="y", visible=False)
save(fig, "softmax_versions.png")

# 3. Stride sweep ------------------------------------------------------------
stride = [1, 2, 4, 8, 16, 32, 64]
ns = [0.32, 0.98, 2.11, 4.85, 7.78, 9.30, 9.37]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(stride, ns, color=BLUE, marker="o", lw=2)
ax.set_xscale("log", base=2)
ax.set_xticks(stride, [str(s) for s in stride])
ax.axvline(8, color=GREY, ls="--", lw=1)
ax.text(8.4, 1.0, "8 values = 64 B\n(one cache line)", fontsize=8, color=GREY)
ax.annotate("levels off past 16\n(128 B: Intel fetches line pairs)", xy=(16, 7.78), xytext=(2.2, 7.6),
            fontsize=8, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
ax.set_xlabel("stride (values apart)")
ax.set_ylabel("ns per value")
ax.set_title("Memory arrives in cache lines, not single numbers")
save(fig, "stride_sweep.png")

# 4. Transformer block regimes ----------------------------------------------
T = ["1", "8", "64", "512"]
ln = [0.046, 0.197, 0.807, 9.640]
mm = [1.216 + 1.531, 1.660 + 1.557, 6.388 + 3.966, 34.054 + 19.827]
ge = [0.281, 1.542, 12.745, 106.480]
gfl = [3.4, 23.5, 58.3, 89.7]
tot = [a + b + c for a, b, c in zip(ln, mm, ge)]
pct = lambda v: [100 * a / t for a, t in zip(v, tot)]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = range(len(T))
b1 = pct(mm); b2 = pct(ge); b3 = pct(ln)
ax.bar(x, b1, color=BLUE, width=0.6, label="two matmuls")
ax.bar(x, b2, bottom=b1, color=AMBER, width=0.6, label="GELU")
ax.bar(x, b3, bottom=[a + b for a, b in zip(b1, b2)], color=GREY, width=0.6, label="LayerNorm")
for i, g in enumerate(gfl):
    ax.text(i, b1[i] / 2, f"{g} GFLOP/s", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
ax.set_xticks(list(x), T)
ax.set_xlabel("tokens processed at once (T)")
ax.set_ylabel("% of block time")
ax.set_ylim(0, 100)
ax.set_title("GPT-2-sized block: memory-bound at T = 1, GELU-bound at T = 512")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
ax.grid(axis="x", visible=False)
save(fig, "block_regimes.png")