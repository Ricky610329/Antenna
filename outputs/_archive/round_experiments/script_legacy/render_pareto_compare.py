"""挑一個 config，rw=0 vs rw=2 detailed 對比。"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THETA_DEG = np.arange(-90, 90.1, 0.5)


root = Path("outputs/dataset_v1")
entries = []
with open(root / "entries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

# Pick a representative entry: 38 GHz × n=31 × broadside × w=20
target = next(
    e for e in entries
    if e["config"]["freq_ghz"] == 38.0
    and e["config"]["n"] == 31
    and e["config"]["target_theta_c"] == 0.0
    and e["config"]["target_width_deg"] == 20.0
)
cfg = target["config"]
main_lo, main_hi = target["main_idx_range"]

rw0 = next(p for p in target["pareto"] if p["ripple_weight"] == 0.0)
rw2 = next(p for p in target["pareto"] if p["ripple_weight"] == 2.0)

pat0 = np.load(root / rw0["pattern_file"])
resp0 = np.load(root / rw0["response_file"])
pat2 = np.load(root / rw2["pattern_file"])
resp2 = np.load(root / rw2["response_file"])

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Top row: rw=0
axes[0, 0].imshow(pat0, cmap="binary", vmin=0, vmax=1, aspect="equal")
axes[0, 0].set_title(f"rw=0 binary pattern 31×31\non-rate={pat0.mean()*100:.1f}%", fontsize=11)
axes[0, 0].set_xlabel("element x")
axes[0, 0].set_ylabel("element y")

axes[0, 1].plot(THETA_DEG, resp0, "r-", linewidth=1.2)
axes[0, 1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5,
                    color="green", alpha=0.15, label="main beam region")
axes[0, 1].axhline(0, color="black", linewidth=0.5)
axes[0, 1].axhline(-3, color="black", linewidth=0.5, linestyle="--", label="-3 dB ceiling")
axes[0, 1].set_ylim(-40, 5)
axes[0, 1].set_xlabel("θ (deg)")
axes[0, 1].set_ylabel("response (dB)")
m0 = rw0["metrics"]
axes[0, 1].set_title(
    f"rw=0 (max-max steering loss)\n"
    f"worst supp={m0['worst_supp']:+.2f} | headline={m0['headline_supp']:+.2f} | "
    f"ripple={m0['main_ripple']:.2f} | main < -3 dB: {m0['main_below_3dB']}/{m0['main_total']}",
    fontsize=10,
)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# Histogram of side response
side0 = np.delete(resp0, np.arange(main_lo, main_hi))
main0 = resp0[main_lo:main_hi]
axes[0, 2].hist(side0, bins=30, color="lightcoral", alpha=0.7, label="sidelobe", edgecolor="k")
axes[0, 2].hist(main0, bins=30, color="lightgreen", alpha=0.7, label="main beam", edgecolor="k")
axes[0, 2].axvline(-3, color="black", linewidth=1, linestyle="--", label="-3 dB cap")
axes[0, 2].set_xlabel("response (dB)")
axes[0, 2].set_title("rw=0 response distribution", fontsize=10)
axes[0, 2].legend(fontsize=9)

# Bottom row: rw=2
axes[1, 0].imshow(pat2, cmap="binary", vmin=0, vmax=1, aspect="equal")
axes[1, 0].set_title(f"rw=2 binary pattern 31×31\non-rate={pat2.mean()*100:.1f}%", fontsize=11)
axes[1, 0].set_xlabel("element x")
axes[1, 0].set_ylabel("element y")

axes[1, 1].plot(THETA_DEG, resp2, "b-", linewidth=1.2)
axes[1, 1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5,
                    color="green", alpha=0.15, label="main beam region")
axes[1, 1].axhline(0, color="black", linewidth=0.5)
axes[1, 1].axhline(-3, color="black", linewidth=0.5, linestyle="--")
axes[1, 1].set_ylim(-40, 5)
axes[1, 1].set_xlabel("θ (deg)")
axes[1, 1].set_ylabel("response (dB)")
m2 = rw2["metrics"]
axes[1, 1].set_title(
    f"rw=2 (worst-case + ripple penalty)\n"
    f"worst supp={m2['worst_supp']:+.2f} | headline={m2['headline_supp']:+.2f} | "
    f"ripple={m2['main_ripple']:.2f} | main < -3 dB: {m2['main_below_3dB']}/{m2['main_total']}",
    fontsize=10,
)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

side2 = np.delete(resp2, np.arange(main_lo, main_hi))
main2 = resp2[main_lo:main_hi]
axes[1, 2].hist(side2, bins=30, color="lightblue", alpha=0.7, label="sidelobe", edgecolor="k")
axes[1, 2].hist(main2, bins=30, color="lightgreen", alpha=0.7, label="main beam", edgecolor="k")
axes[1, 2].axvline(-3, color="black", linewidth=1, linestyle="--", label="-3 dB cap")
axes[1, 2].set_xlabel("response (dB)")
axes[1, 2].set_title("rw=2 response distribution", fontsize=10)
axes[1, 2].legend(fontsize=9)

# Compare hamming distance between pat0 and pat2
hamming = (pat0 != pat2).sum() / pat0.size
fig.suptitle(
    f"Pareto Comparison — 38 GHz × n=31 × broadside × main beam width 20°\n"
    f"Hamming distance(rw=0, rw=2) = {hamming:.2%}  "
    f"(同 config 下兩 mode pattern 差很多 → 設計選擇影響大)",
    fontsize=12,
)
fig.tight_layout()
out = "outputs/pareto_compare_38GHz_n31.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved: {out}")
print(f"Hamming distance: {hamming:.2%}")
