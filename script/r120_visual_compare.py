"""Round 120 — Visual comparison: R94 baseline vs R119 winning recipe."""

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

n = 51
freq = 38e9
inc = 51.0
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5
THETA_DEG = np.arange(-90, 90.1, 0.5)


def soft_max(x, beta=20.0):
    return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0):
    return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def deploy(rw, mean_w):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            main_min = soft_min(main); side_max = soft_max(side); main_max = soft_max(main)
            loss = -(main_min - side_max) + rw * (main_max - main_min) + mean_w * side.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()
        m = evaluate_metrics(resp_bin, main_lo, main_hi)
        side_arr = np.delete(resp_bin, np.arange(main_lo, main_hi))
        m["side_mean"] = float(side_arr.mean())
        if best is None or m["worst_supp"] > best["m"]["worst_supp"]:
            best = {"pat": binary.cpu().numpy(), "resp": resp_bin, "m": m, "seed": seed}
    return best


print("Generating R94 baseline...")
b94 = deploy(rw=2.0, mean_w=0.0)
print(f"  worst={b94['m']['worst_supp']:+.2f}, ripple={b94['m']['main_ripple']:.2f}")

print("Generating R119 winner...")
b119 = deploy(rw=2.0, mean_w=1.0)
print(f"  worst={b119['m']['worst_supp']:+.2f}, ripple={b119['m']['main_ripple']:.2f}")

# Render
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

for row, (b, label, color, color_hist) in enumerate([
    (b94,  "R94 baseline (rw=2, no mean penalty)",  "darkred",   "lightcoral"),
    (b119, "R119 NEW (rw=2, λ_mean=1.0)",            "darkgreen", "lightblue"),
]):
    pat = b["pat"]; resp = b["resp"]; m = b["m"]
    main = resp[main_lo:main_hi]
    side = np.delete(resp, np.arange(main_lo, main_hi))
    side_mean = float(side.mean())
    
    axes[row, 0].imshow(pat, cmap="binary", vmin=0, vmax=1, aspect="equal")
    axes[row, 0].set_title(f"Binary 51×51 (seed={b['seed']}, on-rate={pat.mean()*100:.1f}%)")
    
    axes[row, 1].plot(THETA_DEG, resp, color=color, linewidth=1.4)
    axes[row, 1].axvspan(-90 + main_lo*0.5, -90 + main_hi*0.5, color="green", alpha=0.15, label="main")
    axes[row, 1].axhline(0, color="black", linewidth=0.5)
    axes[row, 1].axhline(-3, color="red", linewidth=0.5, linestyle="--", label="-3 dB cap")
    axes[row, 1].axhline(side_mean, color="purple", linewidth=1, linestyle=":", label=f"side_mean={side_mean:+.1f}")
    axes[row, 1].set_ylim(-50, 5)
    axes[row, 1].set_xlabel("θ (deg)")
    axes[row, 1].set_ylabel("response (dB)")
    axes[row, 1].set_title(
        f"{label}\n"
        f"worst={m['worst_supp']:+.2f}, side_max={float(side.max()):+.2f}, side_mean={side_mean:+.2f}"
    )
    axes[row, 1].legend(fontsize=8, loc="lower right")
    axes[row, 1].grid(alpha=0.3)
    
    axes[row, 2].hist(side, bins=30, color=color_hist, alpha=0.8, edgecolor="black", label="sidelobe")
    axes[row, 2].hist(main, bins=15, color="lightgreen", alpha=0.7, edgecolor="black", label="main")
    axes[row, 2].axvline(side_mean, color="purple", linewidth=2, linestyle=":", label=f"side_mean={side_mean:+.1f}")
    axes[row, 2].axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    axes[row, 2].set_xlabel("response (dB)")
    axes[row, 2].set_title(f"Distribution (main<-3: {m['main_below_3dB']}/30)")
    axes[row, 2].legend(fontsize=8)
    axes[row, 2].set_xlim(-50, 5)

fig.suptitle(
    f"R94 baseline vs R119 NEW recipe — n=51 broadside flat-top\n"
    f"R119 加 mean(side) penalty: side_mean shift {b94['m']['side_mean']:+.2f} → "
    f"{b119['m']['side_mean']:+.2f} (Δ {b119['m']['side_mean']-b94['m']['side_mean']:+.2f} dB)",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
out = "outputs/r120_baseline_vs_winner.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved: {out}")

print("\n=== Summary ===")
print(f"{'metric':<15} | {'R94 baseline':>14} | {'R119 winner':>14} | {'Δ':>8}")
print("-" * 60)
for k, name in [("worst_supp", "worst"), ("main_ripple", "ripple"),
                ("side_max", "side_max"), ("side_mean", "side_mean")]:
    delta = b119['m'][k] - b94['m'][k]
    sign = "+" if delta > 0 else ""
    print(f"{name:<15} | {b94['m'][k]:>+14.2f} | {b119['m'][k]:>+14.2f} | {sign}{delta:>+7.2f}")
