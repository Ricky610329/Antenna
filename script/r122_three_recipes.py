"""Round 122 — Visual progression: R94 baseline → R119 → R121 champion."""

import sys
sys.path.insert(0, "script")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import evaluate_metrics

config.device = "cuda:0"
n = 51; freq = 38e9; inc = 51.0; main_lo, main_hi = 162, 192
gd_steps = 1500; n_restarts = 5
THETA_DEG = np.arange(-90, 90.1, 0.5)

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def quantize(params, n_levels):
    phase = (params * torch.pi) % (2 * torch.pi)
    if n_levels == 0: return phase / torch.pi
    levels = torch.linspace(0, 2*torch.pi*(n_levels-1)/n_levels, n_levels, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(rw, mean_w, n_levels):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + rw * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            quantized = quantize(params, n_levels)
            resp_b = sim(quantized)["response"].cpu().numpy()
        m = evaluate_metrics(resp_b, main_lo, main_hi)
        m["side_mean"] = float(np.delete(resp_b, np.arange(main_lo, main_hi)).mean())
        if best is None or m["worst_supp"] > best["m"]["worst_supp"]:
            best = {"pat": quantized.cpu().numpy(), "resp": resp_b, "m": m, "seed": seed}
    return best

print("Generating 3 recipes...")
r94 = deploy(rw=2.0, mean_w=0.0, n_levels=2);  print(f"  R94: {r94['m']}")
r119 = deploy(rw=2.0, mean_w=1.0, n_levels=2); print(f"  R119: {r119['m']}")
r121 = deploy(rw=2.0, mean_w=1.0, n_levels=4); print(f"  R121: {r121['m']}")

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
recipes = [
    (r94,  "R94 baseline (1-bit, lambda=0)",          "darkred",   "lightcoral"),
    (r119, "R119 (1-bit, lambda=1, mean penalty)",   "darkblue",  "lightblue"),
    (r121, "R121 CHAMPION (2-bit, lambda=1)",        "darkgreen", "lightgreen"),
]

for row, (b, label, color, hist_color) in enumerate(recipes):
    pat = b["pat"]; resp = b["resp"]; m = b["m"]
    main = resp[main_lo:main_hi]; side = np.delete(resp, np.arange(main_lo, main_hi))
    
    axes[row, 0].imshow(pat, cmap="viridis" if row==2 else "binary",
                        vmin=0, vmax=2 if row==2 else 1, aspect="equal")
    axes[row, 0].set_title(f"Pattern (seed={b['seed']})", fontsize=10)
    
    axes[row, 1].plot(THETA_DEG, resp, color=color, linewidth=1.4)
    axes[row, 1].axvspan(-90+main_lo*0.5, -90+main_hi*0.5, color="green", alpha=0.15)
    axes[row, 1].axhline(0, color="black", linewidth=0.5)
    axes[row, 1].axhline(-3, color="red", linewidth=0.5, linestyle="--")
    axes[row, 1].axhline(m["side_mean"], color="purple", linewidth=1.2, linestyle=":",
                          label=f"side_mean={m['side_mean']:+.2f}")
    axes[row, 1].set_ylim(-50, 5)
    axes[row, 1].set_xlabel("theta (deg)"); axes[row, 1].set_ylabel("response (dB)")
    axes[row, 1].set_title(
        f"{label}\nworst={m['worst_supp']:+.2f}, side_max={m['side_max']:+.2f}, "
        f"side_mean={m['side_mean']:+.2f}", fontsize=10)
    axes[row, 1].legend(loc="lower right", fontsize=8); axes[row, 1].grid(alpha=0.3)
    
    axes[row, 2].hist(side, bins=30, color=hist_color, alpha=0.8, edgecolor="black", label="side")
    axes[row, 2].hist(main, bins=15, color="lightyellow", alpha=0.8, edgecolor="black", label="main")
    axes[row, 2].axvline(m["side_mean"], color="purple", linewidth=2, linestyle=":",
                          label=f"mean={m['side_mean']:+.1f}")
    axes[row, 2].axvline(-3, color="red", linewidth=1, linestyle="--")
    axes[row, 2].set_xlabel("response (dB)")
    axes[row, 2].set_title(f"Distribution (main<-3: {m['main_below_3dB']}/30)", fontsize=10)
    axes[row, 2].legend(fontsize=8); axes[row, 2].set_xlim(-50, 5)

fig.suptitle(
    f"Recipe Progression: R94 baseline -> R119 (mean penalty) -> R121 CHAMPION (2-bit)\n"
    f"side_mean: {r94['m']['side_mean']:+.2f} -> {r119['m']['side_mean']:+.2f} -> {r121['m']['side_mean']:+.2f} dB "
    f"(total {r121['m']['side_mean']-r94['m']['side_mean']:+.2f} dB shift)",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
fig.savefig("outputs/r122_three_recipes.png", dpi=110, bbox_inches="tight")
print("\nSaved: outputs/r122_three_recipes.png")
print(f"\nProgression: side_mean {r94['m']['side_mean']:+.2f} -> "
      f"{r119['m']['side_mean']:+.2f} -> {r121['m']['side_mean']:+.2f}")
