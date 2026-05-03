"""Consolidated visualization for R118-R127 phase report.

Generates 4 figures from hardcoded experiment results (no re-simulation):
  Figure 1: Recipe progression bar chart (R94 -> R119 -> R121)
  Figure 2: 4-axis universal validation overview (inc/freq/steering/aperture)
  Figure 3: +45 deg boundary probe (5 recipes vs physical limit)
  Figure 4: Aperture scaling at broadside vs +45 deg

Output: outputs/report_summary_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# Data (hardcoded from R118-R127 results)
# ---------------------------------------------------------------

# Recipe progression
recipes = ["R94 baseline\n(1-bit, lam=0)", "R119\n(1-bit, lam=1)", "R121 CHAMPION\n(2-bit, lam=1)"]
worst_progress     = [1.92, 3.65, 3.45]
side_max_progress  = [-4.51, -6.60, -6.05]
side_mean_progress = [-15.75, -23.70, -30.84]

# Cross-incidence (R123)
inc_angles = [0, 30, 51, 70]
inc_baseline = [1.23, 1.91, 1.92, 1.11]
inc_r121     = [3.48, 2.94, 3.45, 3.61]
inc_baseline_flat = [0/5, 2/5, 5/5, 1/5]   # flat-top compliance
inc_r121_flat     = [4/5, 5/5, 5/5, 4/5]

# Cross-frequency (R124)
freq_labels = ["5.8GHz", "28GHz", "38GHz", "60GHz"]
freq_baseline = [0.59, 1.66, 1.92, 2.09]
freq_r121     = [2.72, 3.39, 3.45, 4.16]
freq_baseline_flat = [0/5, 2/5, 5/5, 3/5]
freq_r121_flat     = [5/5, 5/5, 5/5, 4/5]

# Cross-steering (R125)
steer_angles = [-30, -15, 0, 15, 30, 45]
steer_baseline = [0.92, 1.74, 2.11, 2.21, 1.09, 1.22]
steer_r121     = [1.87, 2.85, 3.30, 3.00, 2.38, 1.17]

# Aperture sweep (R127)
n_values = [31, 51, 71]
aperture_broadside = [0.86, 3.30, 8.77]
aperture_45deg     = [0.42, 1.17, 2.32]
aperture_broadside_flat = [4/5, 5/5, 0/5]
aperture_45deg_flat     = [5/5, 5/5, 3/5]

# +45 deg boundary probe (R126)
boundary_recipes = ["A: R121\n(2-bit, lam=1, rw=2)", "B: 3-bit\nupgrade",
                    "C: rw=3\nstronger ripple", "D: lam=1.5\nstronger mean",
                    "E: continuous\nphase (theory max)"]
boundary_worst   = [1.17, 1.33, 0.75, 2.05, 1.32]
boundary_smean   = [-28.64, -31.48, -27.68, -34.14, -32.62]
boundary_flat    = [5, 5, 5, 3, 5]

# ===============================================================
# FIGURE 1: Recipe progression
# ===============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(len(recipes))
colors = ["#c44e52", "#4c72b0", "#55a868"]

axes[0].bar(x, worst_progress, color=colors, edgecolor="black", linewidth=1.2)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_ylabel("worst suppression (dB)", fontsize=11)
axes[0].set_title("Worst-case improvement\n(higher = better)", fontsize=11, fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(recipes, fontsize=9)
for i, v in enumerate(worst_progress):
    axes[0].text(i, v + 0.1, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, 5)

axes[1].bar(x, side_max_progress, color=colors, edgecolor="black", linewidth=1.2)
axes[1].set_ylabel("side_max (dB)", fontsize=11)
axes[1].set_title("Worst sidelobe\n(lower = better)", fontsize=11, fontweight="bold")
axes[1].set_xticks(x); axes[1].set_xticklabels(recipes, fontsize=9)
for i, v in enumerate(side_max_progress):
    axes[1].text(i, v - 0.3, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(-8, 0)

axes[2].bar(x, side_mean_progress, color=colors, edgecolor="black", linewidth=1.2)
axes[2].set_ylabel("side_mean (dB)", fontsize=11)
axes[2].set_title("Sidelobe distribution mean\n(lower = ENTIRE sidelobe pushed down)", fontsize=11, fontweight="bold")
axes[2].set_xticks(x); axes[2].set_xticklabels(recipes, fontsize=9)
for i, v in enumerate(side_mean_progress):
    axes[2].text(i, v - 1.2, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)
axes[2].set_ylim(-35, 0)
# Highlight the -15 dB total improvement
axes[2].annotate("", xy=(2, -30.84), xytext=(0, -15.75),
                 arrowprops=dict(arrowstyle="<->", color="darkred", lw=2))
axes[2].text(1, -22, "-15.09 dB\ntotal", color="darkred", fontsize=10,
             fontweight="bold", ha="center",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="darkred"))

fig.suptitle("Figure 1 — Recipe progression: R94 baseline -> R121 CHAMPION (n=51, 38GHz, inc=51deg, broadside)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/report_fig1_recipe_progression.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_fig1_recipe_progression.png")

# ===============================================================
# FIGURE 2: 4-axis universal validation overview
# ===============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
bar_w = 0.35

# (a) Incidence
ax = axes[0, 0]
x = np.arange(len(inc_angles))
ax.bar(x - bar_w/2, inc_baseline, bar_w, label="1-bit baseline", color="#c44e52", edgecolor="black")
ax.bar(x + bar_w/2, inc_r121, bar_w, label="R121 CHAMPION", color="#55a868", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels([f"{a}deg" for a in inc_angles])
ax.set_ylabel("worst suppression (dB)")
ax.set_title("(a) Cross-incidence (R123)\nR121 universally rescues baseline", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
for i, (b, r) in enumerate(zip(inc_baseline, inc_r121)):
    ax.text(i - bar_w/2, b + 0.1, f"{b:+.1f}", ha="center", fontsize=8)
    ax.text(i + bar_w/2, r + 0.1, f"{r:+.1f}", ha="center", fontsize=8, fontweight="bold")
# Mark baseline failures (flat-top < 5/5)
for i, (b, r, bf, rf) in enumerate(zip(inc_baseline, inc_r121, inc_baseline_flat, inc_r121_flat)):
    if bf < 1.0:
        ax.text(i - bar_w/2, -0.4, f"{int(bf*5)}/5", ha="center", fontsize=8, color="darkred", fontweight="bold")
    if rf >= 1.0:
        ax.text(i + bar_w/2, -0.4, "OK", ha="center", fontsize=8, color="darkgreen", fontweight="bold")
    else:
        ax.text(i + bar_w/2, -0.4, f"{int(rf*5)}/5", ha="center", fontsize=8, color="darkorange", fontweight="bold")

# (b) Frequency
ax = axes[0, 1]
x = np.arange(len(freq_labels))
ax.bar(x - bar_w/2, freq_baseline, bar_w, label="1-bit baseline", color="#c44e52", edgecolor="black")
ax.bar(x + bar_w/2, freq_r121, bar_w, label="R121 CHAMPION", color="#55a868", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(freq_labels)
ax.set_ylabel("worst suppression (dB)")
ax.set_title("(b) Cross-frequency (R124)\nsub-6G to 60GHz all robust", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
for i, (b, r) in enumerate(zip(freq_baseline, freq_r121)):
    ax.text(i - bar_w/2, b + 0.1, f"{b:+.1f}", ha="center", fontsize=8)
    ax.text(i + bar_w/2, r + 0.1, f"{r:+.1f}", ha="center", fontsize=8, fontweight="bold")
for i, (bf, rf) in enumerate(zip(freq_baseline_flat, freq_r121_flat)):
    if bf < 1.0:
        ax.text(i - bar_w/2, -0.4, f"{int(bf*5)}/5", ha="center", fontsize=8, color="darkred", fontweight="bold")
    if rf >= 1.0:
        ax.text(i + bar_w/2, -0.4, "OK", ha="center", fontsize=8, color="darkgreen", fontweight="bold")
    else:
        ax.text(i + bar_w/2, -0.4, f"{int(rf*5)}/5", ha="center", fontsize=8, color="darkorange", fontweight="bold")

# (c) Steering
ax = axes[1, 0]
x = np.arange(len(steer_angles))
ax.bar(x - bar_w/2, steer_baseline, bar_w, label="1-bit baseline", color="#c44e52", edgecolor="black")
ax.bar(x + bar_w/2, steer_r121, bar_w, label="R121 CHAMPION", color="#55a868", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels([f"{a:+d}d" for a in steer_angles])
ax.set_ylabel("worst suppression (dB)")
ax.set_title("(c) Cross-steering (R125)\nUniversal in [-30,+30]; +45deg = TIE (R126 boundary)", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)
for i, (b, r) in enumerate(zip(steer_baseline, steer_r121)):
    ax.text(i - bar_w/2, b + 0.1, f"{b:+.1f}", ha="center", fontsize=8)
    ax.text(i + bar_w/2, r + 0.1, f"{r:+.1f}", ha="center", fontsize=8, fontweight="bold")
# Highlight +45 boundary
ax.axvspan(4.5, 5.5, color="orange", alpha=0.15, label="boundary")
ax.text(5, 4.0, "BOUNDARY\n(physical aperture limit)", ha="center", fontsize=8,
        color="darkred", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="darkred"))

# (d) Aperture
ax = axes[1, 1]
x = np.arange(len(n_values))
ax.bar(x - bar_w/2, aperture_broadside, bar_w, label="broadside (0deg)", color="#4c72b0", edgecolor="black")
ax.bar(x + bar_w/2, aperture_45deg, bar_w, label="+45deg steering", color="#c44e52", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in n_values])
ax.set_ylabel("worst suppression (dB)")
ax.set_title("(d) Aperture sweep (R127)\nBigger n breaks +45deg boundary, but n=71 broadside loses flat-top", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, (b, s) in enumerate(zip(aperture_broadside, aperture_45deg)):
    ax.text(i - bar_w/2, b + 0.15, f"{b:+.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + bar_w/2, s + 0.15, f"{s:+.2f}", ha="center", fontsize=9, fontweight="bold")
# Mark n=71 broadside flat-top failure
ax.text(2 - bar_w/2, 9.0, "flat-top\n0/5 !!", ha="center", fontsize=9, color="darkred",
        fontweight="bold", bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="darkred"))

fig.suptitle("Figure 2 — R121 CHAMPION universal validation across 4 physical axes",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/report_fig2_4axis_validation.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_fig2_4axis_validation.png")

# ===============================================================
# FIGURE 3: +45 deg boundary probe (R126)
# ===============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(boundary_recipes))
colors_b = ["#55a868", "#4c72b0", "#937860", "#c44e52", "#8172b3"]

ax = axes[0]
ax.bar(x, boundary_worst, color=colors_b, edgecolor="black", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(boundary_recipes, fontsize=8)
ax.set_ylabel("worst suppression (dB)", fontsize=11)
ax.set_title("Worst-case at +45deg steering\nContinuous phase saturates -> physical aperture limit",
             fontweight="bold", fontsize=11)
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(boundary_worst):
    ax.text(i, v + 0.05, f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(boundary_worst[4], color="purple", linestyle="--", linewidth=1.5,
           label=f"continuous phase = +{boundary_worst[4]:.2f} (theory max)")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, 2.5)

ax = axes[1]
flat_marker = ["OK" if f == 5 else f"{f}/5" for f in boundary_flat]
ax.bar(x, boundary_smean, color=colors_b, edgecolor="black", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(boundary_recipes, fontsize=8)
ax.set_ylabel("side_mean (dB)", fontsize=11)
ax.set_title("Sidelobe distribution mean at +45deg\n(lambda=1.5 trades flat-top for lower mean)",
             fontweight="bold", fontsize=11)
ax.grid(axis="y", alpha=0.3)
for i, (v, f) in enumerate(zip(boundary_smean, flat_marker)):
    ax.text(i, v - 1.0, f"{v:+.2f}\nflat: {f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(-40, 0)

fig.suptitle("Figure 3 — R126 +45deg boundary probe: hardware upgrades cannot break the limit",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/report_fig3_45deg_boundary.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_fig3_45deg_boundary.png")

# ===============================================================
# FIGURE 4: Aperture scaling line plot
# ===============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(n_values, aperture_broadside, "o-", linewidth=2.5, markersize=12, color="#4c72b0", label="broadside (0deg)")
ax.plot(n_values, aperture_45deg, "s-", linewidth=2.5, markersize=12, color="#c44e52", label="+45deg steering")
for n, b in zip(n_values, aperture_broadside):
    ax.annotate(f"+{b:.2f}", (n, b), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")
for n, s in zip(n_values, aperture_45deg):
    ax.annotate(f"+{s:.2f}", (n, s), textcoords="offset points", xytext=(8, -15), fontsize=10, fontweight="bold")
ax.set_xlabel("aperture n (n x n element grid)", fontsize=11)
ax.set_ylabel("worst suppression (dB)", fontsize=11)
ax.set_title("Aperture scaling: bigger n = more headroom\n(R127 verifies R126 physical-limit hypothesis)", fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="upper left")
ax.set_xticks(n_values)
ax.set_ylim(0, 10)

ax = axes[1]
flat_b = [f * 100 for f in aperture_broadside_flat]
flat_45 = [f * 100 for f in aperture_45deg_flat]
bar_w = 0.35
x = np.arange(len(n_values))
ax.bar(x - bar_w/2, flat_b, bar_w, color="#4c72b0", edgecolor="black", label="broadside")
ax.bar(x + bar_w/2, flat_45, bar_w, color="#c44e52", edgecolor="black", label="+45deg")
ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in n_values])
ax.set_ylabel("flat-top compliance (%)", fontsize=11)
ax.set_title("Flat-top compliance across n\n(n=71 broadside collapses -> R121 recipe needs re-tune)",
             fontweight="bold")
for i, (b, s) in enumerate(zip(flat_b, flat_45)):
    ax.text(i - bar_w/2, b + 2, f"{int(b)}%", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + bar_w/2, s + 2, f"{int(s)}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 115)
ax.axhline(100, color="green", linestyle=":", linewidth=1)
ax.legend(fontsize=10, loc="lower left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 4 — R127 aperture sweep: hypothesis verified + new edge case found",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/report_fig4_aperture_scaling.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_fig4_aperture_scaling.png")

print("\nAll 4 figures generated successfully.")
