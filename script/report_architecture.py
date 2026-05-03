"""Generate architecture diagram + development timeline figures for the
R94-R156 comprehensive report.

Two figures:
  1. report_arch_pipeline.png  — deployment pipeline flow + loss design
  2. report_arch_timeline.png  — development timeline with phases
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# FIGURE 1: Deployment pipeline architecture
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left panel: pipeline flow
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 12)
ax.axis("off")
ax.set_title("Deployment pipeline: optimize_ris_1bit()",
             fontsize=13, fontweight="bold", pad=10)


def box(ax, x, y, w, h, text, color="#e6f0fa", edge="black", textsize=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=textsize, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, label=None, color="black"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="->", mutation_scale=18,
                        color=color, linewidth=1.5)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2,
                label, fontsize=8, color=color, style="italic")


# Input
box(ax, 5, 11, 7, 0.7, "INPUT: (n, inc, freq, width, steering_center_deg)",
    color="#fff3b0", textsize=10)
arrow(ax, 5, 10.6, 5, 10.0)

# Selector
box(ax, 5, 9.6, 5.5, 0.7,
    "select_1bit_recipe()  [R134/R135]\n→ (rw, λ_mean) per (n, inc, freq, w)",
    color="#cce5ff", textsize=9)
arrow(ax, 5, 9.2, 5, 8.6)

# Forward fn split
box(ax, 2.5, 8.2, 3.5, 0.7,
    "RISSimulator\n(analytical)", color="#d4edda", textsize=9)
box(ax, 7.5, 8.2, 3.5, 0.7,
    "WarmStartSurrogate\n(R146 - exact match)", color="#f8d7da", textsize=9)
arrow(ax, 5, 9.2, 2.5, 8.6, color="gray")
arrow(ax, 5, 9.2, 7.5, 8.6, color="gray")
ax.text(5, 8.85, "forward_fn=", ha="center", fontsize=8, style="italic", color="gray")

# Optimization loop
box(ax, 5, 7.2, 8, 0.7,
    "Adam(lr=0.05) × N restarts × 1500 GD steps",
    color="#cce5ff", textsize=10)
arrow(ax, 2.5, 7.85, 5, 7.55, color="gray")
arrow(ax, 7.5, 7.85, 5, 7.55, color="gray")

# Loss
box(ax, 5, 6.0, 8.5, 1.0,
    "loss = -(soft_min(main) - soft_max(side))      # R94 worst-case\n"
    "     + rw  · (soft_max(main) - soft_min(main))  # R94 ripple\n"
    "     + λₘ · side.mean()                         # R119 distribution",
    color="#fff3b0", textsize=8)
arrow(ax, 5, 6.85, 5, 6.5)
arrow(ax, 5, 5.5, 5, 5.0)

# Joint early-stop
box(ax, 5, 4.6, 7.5, 0.9,
    "Joint early-stop  [R140]\n"
    "every 50 steps: track best worst AMONG flat-top valid snapshots\n"
    "(critical safety net — R150 confirms at perfect surrogate too)",
    color="#cce5ff", textsize=8.5)
arrow(ax, 5, 4.15, 5, 3.7)

# Quantize at eval
box(ax, 5, 3.3, 6, 0.7,
    "Quantize: phase = (params·π) mod 2π\n→ binary (0 if (π/2,3π/2) else 1)",
    color="#d4edda", textsize=8)
arrow(ax, 5, 2.95, 5, 2.5)

# Output
box(ax, 5, 2.1, 7.5, 0.7,
    "OUTPUT: best binary pattern + per-seed metrics + recipe",
    color="#fff3b0", textsize=10)

# Validated envelope footnote
ax.text(5, 1.0, "Validated envelope:  n ∈ {15, 31, 51, 71}   |   inc ∈ [0, 70°]   "
        "|   freq ∈ [5.8, 60] GHz   |   width ∈ [10, 30°]   |   steering ∈ [-30, +30°]",
        ha="center", fontsize=9, style="italic")
ax.text(5, 0.5, "Edge cases: +45° steering = aperture limit (R126)   |   width=30° = recipe boundary (R129)   "
        "|   inc=0+60GHz = needs n=71 (R133)",
        ha="center", fontsize=8, color="darkred", style="italic")

# Right panel: loss component breakdown
ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 12)
ax.set_title("Loss components: design rationale",
             fontsize=13, fontweight="bold", pad=10)

components = [
    ("R94 - Worst-case",
     "-(soft_min(main) - soft_max(side))",
     "Forces: min of main beam region must beat max of sidelobes.\n"
     "Avoids R57-R63's max-max loss that rewards single peak.",
     "#d4edda"),
    ("R94 - Ripple penalty",
     "rw * (soft_max(main) - soft_min(main))",
     "Forces: main beam region stays flat (low ripple).\n"
     "Maps to 'main beam region close to cap' user requirement.",
     "#cce5ff"),
    ("R119 - Distribution penalty",
     "lambda_m * side.mean()",
     "Pushes ENTIRE sidelobe region down, not just the worst bin.\n"
     "R118 found mean(side) is the right operator (vs L2, ReLU).\n"
     "side_mean: -15.75 -> -23.70 -> -30.84 (R94->R119->R121).",
     "#fff3b0"),
    ("R140 - Joint early-stop",
     "track best worst AMONG flat-valid snapshots",
     "Picks pattern from trajectory peak that ALSO has flat-top OK.\n"
     "Naive 'just track best worst' (R138) sacrificed flat-top in 3/4.\n"
     "R148: even survives 20% surrogate weight noise.",
     "#f8d7da"),
    ("R154 - Multi-freq sum",
     "sum over in-band freqs of R119 recipe",
     "Same recipe summed across freqs -> broadband (10% rel BW PASSes).\n"
     "Joint optimization is regularization: BETTER than single-freq even\n"
     "at the design freq (+0.46 dB at 38GHz, +1.16 dB off-band).",
     "#e2d9f3"),
]

y = 11
for title, formula, desc, color in components:
    box(ax, 5, y, 9, 1.7, "", color=color, textsize=0)
    ax.text(0.6, y + 0.55, title, fontsize=11, fontweight="bold", ha="left")
    ax.text(0.6, y + 0.15, f"  {formula}", fontsize=9.5,
            family="monospace", color="darkblue", ha="left")
    ax.text(0.6, y - 0.45, desc, fontsize=8.5, ha="left")
    y -= 2.1

ax.text(5, 0.5,
        "Same loss design transfers to surrogate-in-the-loop (R147) and\n"
        "patch S-parameters (R151 plan: in-band/out-of-band frequencies)",
        ha="center", fontsize=9, style="italic")

fig.tight_layout()
fig.savefig("outputs/report_arch_pipeline.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_arch_pipeline.png")

# ============================================================
# FIGURE 2: Development timeline
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 100); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("R94 → R156: 3-phase development timeline",
             fontsize=14, fontweight="bold", pad=15)

phases = [
    {"name": "Phase 0\nR57-R63",
     "x_start": 0, "x_end": 6,
     "y": 8, "color": "#d3d3d3",
     "label": "max-max loss\n(metric-cheating\nstarting failure)"},
    {"name": "Phase 1: RIS playground methodology",
     "x_start": 6, "x_end": 60,
     "y": 8, "color": "#cce5ff",
     "label": "R94-R141"},
    {"name": "Phase 2: Surrogate-in-the-loop",
     "x_start": 60, "x_end": 78,
     "y": 8, "color": "#d4edda",
     "label": "R142-R149"},
    {"name": "Phase 3: Patch bridge + broadband",
     "x_start": 78, "x_end": 100,
     "y": 8, "color": "#fff3b0",
     "label": "R150-R156"},
]

for p in phases:
    rect = FancyBboxPatch((p["x_start"], p["y"] - 0.6),
                          p["x_end"] - p["x_start"], 1.2,
                          boxstyle="round,pad=0.05",
                          facecolor=p["color"], edgecolor="black", linewidth=1.5)
    ax.add_patch(rect)
    ax.text((p["x_start"] + p["x_end"]) / 2, p["y"] + 0.15, p["name"],
            ha="center", fontsize=11, fontweight="bold")
    ax.text((p["x_start"] + p["x_end"]) / 2, p["y"] - 0.3, p["label"],
            ha="center", fontsize=8, style="italic")

# Detailed milestones below
milestones = [
    # (x, label, sub_label, color)
    (3, "R57-R63",
     "max-max loss\n+30 dB illusion\n(real -18 dB)", "#666"),
    (10, "R94",
     "worst-case loss\n+ ripple penalty\nworst +1.92, flat OK", "#0066cc"),
    (18, "R118-R119",
     "discover\nmean(side) penalty\nside_mean -15.75 -> -23.70", "#0066cc"),
    (24, "R121",
     "2-bit + lam=1\nside_mean -30.84\n(non-deployable later)", "#0066cc"),
    (30, "R123-R127",
     "4-axis universal\nvalidation\n(inc/freq/steer/n)", "#0066cc"),
    (37, "R128",
     "1-bit pivot\n(0 or pi only)\n+ width x steering", "#cc0033"),
    (43, "R134-R135",
     "selector codified\nwidth>12 -> R129\nelse -> R119", "#0066cc"),
    (50, "R136-R140",
     "fab tolerance OK\njoint early-stop OK", "#0066cc"),
    (56, "R141",
     "deployment API\n6/6 PASS\non held-out", "#009933"),
    (64, "R142-R145",
     "surrogate fail\n4 negative results\n(arch/data both)", "#cc6600"),
    (70, "R146-R147",
     "warm-start\nR^2 = 1.000000\nsurrogate-loop OK", "#009933"),
    (75, "R148-R149",
     "20% noise OK\n4 configs PASS\n3-8x speedup", "#009933"),
    (82, "R150-R153",
     "unified pipeline\npatch bridge plan\nn=15 extension", "#0066cc"),
    (88, "R154",
     "multi-freq joint\n> single-freq\n10% BW PASS", "#009933"),
    (94, "R155-R156",
     "BW limit found\n10% PASS / 32% FAIL\nvisualization", "#cc6600"),
]

for x, label, sub, color in milestones:
    ax.plot(x, 6.8, "o", markersize=11, color=color, zorder=5)
    ax.text(x, 7.3, label, ha="center", fontsize=8.5, fontweight="bold", color=color)
    ax.text(x, 6.3, sub, ha="center", fontsize=7, color="black")

# Bottom section: outcomes / key findings
ax.text(50, 4.5, "Key takeaways", ha="center", fontsize=13, fontweight="bold")

outcomes = [
    ("Loss design", "soft-min main vs soft-max side + ripple + mean(side)\n"
     "framework-agnostic; transfers to surrogate AND broadband", "#d4edda"),
    ("Recipe selector", "select_1bit_recipe(n, inc, freq, width)\n"
     "decision tree from R119/R129/R131/R133/R135\n"
     "validated envelope: n∈{15,31,51,71}, w∈[10,30°]", "#cce5ff"),
    ("Joint early-stop", "max worst AMONG flat-top-valid trajectory snapshots\n"
     "the critical safety net for surrogate-in-the-loop\n"
     "survives 20% surrogate weight perturbation", "#fff3b0"),
    ("Multi-freq broadband", "loss summed across freqs: 10% BW clean PASS\n"
     "joint > single-freq EVEN at design freq\n"
     "32%+ BW needs aperture upgrade", "#f8d7da"),
]

x_starts = [3, 27, 51, 75]
for x, (title, body, color) in zip(x_starts, outcomes):
    rect = FancyBboxPatch((x, 1.5), 22, 2.5,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + 11, 3.5, title, fontsize=11, fontweight="bold", ha="center")
    ax.text(x + 11, 2.5, body, fontsize=8.5, ha="center")

# Bottom note
ax.text(50, 0.5,
        "Cumulative: 156 rounds, 200+ commits, branch ricky/modernize. "
        "1-bit RIS + 4-axis selector + joint early-stop + multi-freq BW = "
        "patch-transition-ready methodology.",
        ha="center", fontsize=9, style="italic")

fig.tight_layout()
fig.savefig("outputs/report_arch_timeline.png", dpi=120, bbox_inches="tight")
print("Saved: outputs/report_arch_timeline.png")
