"""Inference visualization for R121 CHAMPION (2-bit, historical).

This script visualizes the R121 2-bit RIS patterns. R121 is NOT
hardware-deployable per the user's 1-bit constraint (only 0/pi phase),
but is kept for methodology comparison.

Pattern visualization uses:
  - ListedColormap with 4 DISCRETE colors (one per phase level)
  - interpolation="nearest" (no smoothing)
  - colorbar ticks at exactly the 4 levels: 0, pi/2, pi, 3pi/2

Configurations (all n=51 to avoid VRAM ceiling on n=71):
  Row A: n=51, broadside
  Row B: n=51, +30 deg steering
  Row C: n=51, +45 deg steering (boundary)
  Row D: n=51, +15 deg steering (alternative, was previously n=71)
"""
import sys
sys.path.insert(0, "script")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
freq = 38e9
inc = 51.0
gd_steps = 1500
n_restarts = 5
THETA_DEG = np.arange(-90, 90.1, 0.5)


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def quantize_2bit(params):
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi*3/4, 4, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi


def run_r121(n, main_lo, main_hi, n_restarts_local=None):
    """R121 CHAMPION: 2-bit + lambda=1 + rw=2."""
    if n_restarts_local is None:
        n_restarts_local = n_restarts
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts_local):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            quantized = quantize_2bit(params)
            resp_b = sim(quantized)["response"].cpu().numpy()
        main_arr = resp_b[main_lo:main_hi]
        side_arr = np.delete(resp_b, np.arange(main_lo, main_hi))
        score = {
            "worst": float(main_arr.min() - side_arr.max()),
            "side_max": float(side_arr.max()),
            "side_mean": float(side_arr.mean()),
            "ripple": float(main_arr.max() - main_arr.min()),
            "main_min": float(main_arr.min()),
            "flat_top": int(np.sum(main_arr < -3)) == 0,
            "main_below_3": int(np.sum(main_arr < -3)),
        }
        if best is None or score["worst"] > best["score"]["worst"]:
            best = {
                "pat": quantized.cpu().numpy(),
                "resp": resp_b,
                "score": score,
                "seed": seed,
            }
    return best


# 4 configs (all n=51 — avoid VRAM ceiling on n=71)
configs = [
    {"label": "(A) n=51, broadside (R121 sweet spot)",
     "n": 51, "main_lo": 172, "main_hi": 188, "color": "#55a868"},
    {"label": "(B) n=51, +30 deg steering (last universal point)",
     "n": 51, "main_lo": 232, "main_hi": 248, "color": "#4c72b0"},
    {"label": "(C) n=51, +45 deg steering (boundary case)",
     "n": 51, "main_lo": 262, "main_hi": 278, "color": "#dd8452"},
    {"label": "(D) n=51, +15 deg steering",
     "n": 51, "main_lo": 202, "main_hi": 218, "color": "#c44e52"},
]

print("=" * 90)
print("Running R121 CHAMPION inference at 4 configurations...")
print("=" * 90)

results = []
for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/4] {cfg['label']}", flush=True)
    nr = cfg.get("n_restarts", n_restarts)
    r = run_r121(cfg["n"], cfg["main_lo"], cfg["main_hi"], nr)
    s = r["score"]
    flat_str = "OK" if s['flat_top'] else "fail ({} bins)".format(s['main_below_3'])
    print(f"  worst={s['worst']:+.2f}, side_max={s['side_max']:+.2f}, "
          f"side_mean={s['side_mean']:+.2f}, ripple={s['ripple']:.2f}, "
          f"flat-top={flat_str}", flush=True)
    results.append((cfg, r))


# Render
fig, axes = plt.subplots(4, 3, figsize=(17, 14))

for row, (cfg, r) in enumerate(results):
    pat = r["pat"]
    resp = r["resp"]
    s = r["score"]
    n = cfg["n"]
    main_lo, main_hi = cfg["main_lo"], cfg["main_hi"]
    main_arr = resp[main_lo:main_hi]
    side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    main_lo_deg = -90 + main_lo * 0.5
    main_hi_deg = -90 + main_hi * 0.5

    # Col 0: 2-bit phase pattern (4 DISCRETE levels, no smoothing)
    ax = axes[row, 0]
    # 4 distinct colors: white, light gray, dark gray, black
    cmap_4lvl = mcolors.ListedColormap(["#ffffff", "#bcbcbc", "#5c5c5c", "#000000"])
    bounds = [-0.25, 0.25, 0.75, 1.25, 1.75]   # 4 bins centered on 0, 0.5, 1, 1.5
    norm = mcolors.BoundaryNorm(bounds, cmap_4lvl.N)
    im = ax.imshow(pat, cmap=cmap_4lvl, norm=norm, aspect="equal",
                   interpolation="nearest")
    ax.set_title(f"2-bit phase pattern ({n} x {n}, seed={r['seed']})\n"
                 f"4 phase levels: 0, pi/2, pi, 3pi/2",
                 fontsize=9)
    ax.set_xticks([0, n//2, n-1])
    ax.set_yticks([0, n//2, n-1])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels(["0", "pi/2", "pi", "3pi/2"])
    cbar.set_label("phase", fontsize=8)

    # Col 1: far-field response
    ax = axes[row, 1]
    ax.plot(THETA_DEG, resp, color=cfg["color"], linewidth=1.4)
    ax.axvspan(main_lo_deg, main_hi_deg, color="green", alpha=0.15, label="main region")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-3, color="red", linewidth=0.8, linestyle="--", label="-3 dB cap")
    ax.axhline(s["side_mean"], color="purple", linewidth=1, linestyle=":",
               label=f"side_mean={s['side_mean']:+.1f}")
    ax.axhline(s["side_max"], color="darkred", linewidth=1, linestyle=":",
               label=f"side_max={s['side_max']:+.1f}")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-50, 5)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("response (dB)")
    flat_str = "OK" if s["flat_top"] else f"FAIL ({s['main_below_3']}/30)"
    ax.set_title(f"far-field — worst={s['worst']:+.2f} dB, flat-top: {flat_str}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # Col 2: distribution histogram
    ax = axes[row, 2]
    ax.hist(side_arr, bins=40, color=cfg["color"], alpha=0.7,
            edgecolor="black", label="sidelobe")
    ax.hist(main_arr, bins=15, color="lightgreen", alpha=0.8,
            edgecolor="black", label="main")
    ax.axvline(s["side_mean"], color="purple", linewidth=2, linestyle=":",
               label=f"side_mean={s['side_mean']:+.1f}")
    ax.axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    ax.set_xlim(-50, 5)
    ax.set_xlabel("response (dB)")
    ax.set_ylabel("count")
    ax.set_title(f"distribution — side_mean={s['side_mean']:+.2f}, ripple={s['ripple']:.2f}",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # Row label on the left
    fig.text(0.005, 1.0 - (row + 0.5) / 4, cfg["label"], rotation=90, ha="center", va="center",
             fontsize=11, fontweight="bold")

fig.suptitle("R121 CHAMPION inference at 4 representative configurations\n"
             "(2-bit phase, lambda_mean=1.0, rw=2.0)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0.01, 0, 1, 0.97])
out = "outputs/report_fig5_inference_examples.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}")
