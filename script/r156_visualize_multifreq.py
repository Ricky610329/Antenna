"""Round 156 — Visualize R154 + R155 multi-freq findings.

Two-panel summary of the broadband optimization results:
  Panel 1 (R154): single-freq vs multi-freq @ ~10% BW (3 freqs around 38GHz)
  Panel 2 (R155): bandwidth limit (10% / 32% / 53% rel BW)

Pure visualization from hardcoded experiment data — no GPU re-run needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# R154 data: single-freq @ 38GHz vs multi-freq joint
freqs_r154 = [36, 38, 40]
single_means = [0.80, 1.66, 0.98]
single_mins = [-0.85, 0.69, 0.32]
single_flats = [4, 5, 5]   # out of 5
multi_means = [2.19, 2.12, 1.91]
multi_mins = [1.20, 0.94, 0.66]
multi_flats = [5, 5, 5]

# R155 data: BW sweep
bw_labels = ["~10% BW\n(36/38/40)", "~32% BW\n(32/38/44)", "~53% BW\n(28/38/48)"]
bw_means = [
    [2.44, 2.47, 2.27],   # 10% BW per-freq means
    [1.36, 2.01, 2.18],   # 32% BW
    [1.51, 1.19, 1.64],   # 53% BW
]
bw_flats = [
    [3, 3, 3],            # 10% BW per-freq flat (out of 3)
    [1, 1, 2],            # 32% BW
    [2, 1, 1],            # 53% BW
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ===== Panel 1: R154 single vs multi @ 10% BW =====
ax = axes[0]
x = np.arange(len(freqs_r154))
bar_w = 0.35

# Means as bars, mins as small lines
b1 = ax.bar(x - bar_w/2, single_means, bar_w, label="single-freq @38GHz",
            color="#c44e52", edgecolor="black", linewidth=1.2)
b2 = ax.bar(x + bar_w/2, multi_means, bar_w, label="multi-freq joint",
            color="#55a868", edgecolor="black", linewidth=1.2)

# Min worst as horizontal lines on top of each bar
for i in range(len(freqs_r154)):
    ax.plot([i - bar_w, i], [single_mins[i], single_mins[i]], "k_",
            markersize=14, markeredgewidth=2)
    ax.plot([i, i + bar_w], [multi_mins[i], multi_mins[i]], "k_",
            markersize=14, markeredgewidth=2)
    # Annotate bar height
    ax.text(i - bar_w/2, single_means[i] + 0.1, f"{single_means[i]:+.2f}",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(i + bar_w/2, multi_means[i] + 0.1, f"{multi_means[i]:+.2f}",
            ha="center", fontsize=9, fontweight="bold")
    # Flat-top label
    sf = "OK" if single_flats[i] == 5 else f"{single_flats[i]}/5"
    mf = "OK" if multi_flats[i] == 5 else f"{multi_flats[i]}/5"
    sf_color = "darkgreen" if single_flats[i] == 5 else "darkred"
    mf_color = "darkgreen" if multi_flats[i] == 5 else "darkred"
    ax.text(i - bar_w/2, -1.5, f"flat: {sf}", ha="center", fontsize=8,
            color=sf_color, fontweight="bold")
    ax.text(i + bar_w/2, -1.5, f"flat: {mf}", ha="center", fontsize=8,
            color=mf_color, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{f} GHz" for f in freqs_r154])
ax.set_ylabel("worst suppression (dB)", fontsize=11)
ax.set_title("(a) R154: Multi-freq joint beats single-freq UNIVERSALLY\n"
             "(black tick = min across 5 seeds)", fontweight="bold", fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-2.0, 4.0)

# Annotation: bandwidth gain
ax.annotate("", xy=(0 + bar_w/2, 2.19), xytext=(0 - bar_w/2, 0.80),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2))
ax.text(0.0, 1.4, "+1.39 dB\n@36GHz", color="darkgreen", fontsize=9,
        fontweight="bold", ha="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="darkgreen"))

# ===== Panel 2: R155 BW limit sweep =====
ax = axes[1]

# Stack bars: 3 per BW
freq_labels_per_bw = [
    ["36G", "38G", "40G"],
    ["32G", "38G", "44G"],
    ["28G", "38G", "48G"],
]
bw_colors = ["#55a868", "#dd8452", "#c44e52"]  # green/orange/red for difficulty
bar_w_b = 0.25

# Three groups (one per BW), each with 3 bars (one per freq within group)
for bw_idx, (means, flats, color, freq_lbls) in enumerate(zip(bw_means, bw_flats, bw_colors, freq_labels_per_bw)):
    for f_idx, (m, f, lbl) in enumerate(zip(means, flats, freq_lbls)):
        x_pos = bw_idx + (f_idx - 1) * bar_w_b
        ax.bar(x_pos, m, bar_w_b, color=color, edgecolor="black",
               alpha=0.6 if f_idx != 1 else 1.0)
        ax.text(x_pos, m + 0.05, f"{m:+.2f}", ha="center", fontsize=8,
                fontweight="bold")
        # Flat-top compliance below bar
        flat_color = "darkgreen" if f == 3 else "darkred"
        ax.text(x_pos, -0.4, f"{f}/3", ha="center", fontsize=8,
                color=flat_color, fontweight="bold")
        # freq label inside/below bar
        ax.text(x_pos, -0.9, lbl, ha="center", fontsize=8, color="gray")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(np.arange(len(bw_labels)))
ax.set_xticklabels(bw_labels)
ax.set_ylabel("mean worst suppression (dB)", fontsize=11)
ax.set_title("(b) R155: Bandwidth limit\n"
             "(faded = side freqs, solid = center freq, flat fraction below)",
             fontweight="bold", fontsize=11)
ax.set_ylim(-1.2, 3.5)
ax.grid(axis="y", alpha=0.3)

# Verdict boxes
ax.text(0, 3.0, "PASS", ha="center", fontsize=11, fontweight="bold",
        color="darkgreen",
        bbox=dict(boxstyle="round", facecolor="lightgreen", edgecolor="darkgreen"))
ax.text(1, 3.0, "FAIL\nflat-top", ha="center", fontsize=10, fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="darkred"))
ax.text(2, 3.0, "FAIL", ha="center", fontsize=11, fontweight="bold",
        color="darkred",
        bbox=dict(boxstyle="round", facecolor="lightcoral", edgecolor="darkred"))

fig.suptitle("Multi-frequency RIS optimization (n=51, inc=51deg, w=10) — patch BW analog",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out = "outputs/r156_multifreq_summary.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
