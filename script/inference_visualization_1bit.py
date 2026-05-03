"""1-bit RIS inference visualization (FIXES previous 2-bit Figure 5).

User constraint: hardware only supports 0 or 180-deg phase (1-bit binary).
The previous report_fig5_inference_examples.png used quantize_2bit() which
shows 4 phase levels — that's R121 (2-bit), not the deployable 1-bit pipeline.

This script regenerates inference figures using the actual 1-bit pipeline:
  - Recipe selector (R141 / R134-R135)
  - Joint early-stop (R140)
  - Hard 1-bit quantization (phase = 0 or pi only)

Pattern displayed in BLACK (1) / WHITE (0) so the binary-ness is unambiguous.
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
gd_steps = 1500
n_restarts = 5
THETA_DEG = np.arange(-90, 90.1, 0.5)


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def select_1bit_recipe(n, inc_deg, freq_hz, width_deg):
    """R134/R135 selector."""
    if n == 71:
        if inc_deg == 0 and freq_hz >= 50e9:
            return {"rw": 5.0, "lam": 0.3, "tier": "R133 n=71 inc=0 mmWave"}
        if width_deg <= 15:
            return {"rw": 5.0, "lam": 0.5, "tier": "n=71 narrow"}
        return {"rw": 7.0, "lam": 0.5, "tier": "n=71 wide"}
    if width_deg > 12:
        if width_deg <= 20:
            return {"rw": 3.0, "lam": 1.0, "tier": "R129 wide cap"}
        return {"rw": 3.0, "lam": 0.5, "tier": "R129 marginal"}
    if inc_deg == 0 and freq_hz >= 20e9:
        if freq_hz >= 50e9: raise ValueError("inc=0 + 50GHz at n=51 -> use n=71")
        if freq_hz >= 35e9: return {"rw": 2.0, "lam": 0.5, "tier": "R131 38GHz rescue"}
        return {"rw": 2.0, "lam": 0.3, "tier": "R131 28GHz rescue"}
    return {"rw": 2.0, "lam": 1.0, "tier": "R119 baseline"}


def quantize_1bit(params):
    """Hard 1-bit: phase mod 2pi -> 0 if (pi/2, 3pi/2) else 1.
    Returns float tensor in {0.0, 1.0} representing phase in {0, pi}."""
    phase = (params * torch.pi) % (2 * torch.pi)
    binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
    return binary


def eval_binary_metrics(resp_np, main_lo, main_hi):
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "worst": float(main.min() - side.max()),
        "side_max": float(side.max()),
        "side_mean": float(side.mean()),
        "ripple": float(main.max() - main.min()),
        "main_min": float(main.min()),
        "flat_top": int(np.sum(main < -3)) == 0,
        "main_below_3": int(np.sum(main < -3)),
    }


def optimize_1bit(n, inc_deg, freq_hz, width_deg, n_restarts_local=None):
    """Full 1-bit pipeline: selector + joint early-stop."""
    if n_restarts_local is None:
        n_restarts_local = n_restarts
    recipe = select_1bit_recipe(n, inc_deg, freq_hz, width_deg)
    rw, lam = recipe["rw"], recipe["lam"]
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    main_lo, main_hi = steer_to_indices(0, width_deg)

    best = None
    for seed in range(n_restarts_local):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        # Joint early-stop
        best_joint_worst = -1e9; best_state = None
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + rw * (mx - mm) + lam * side.mean()
            loss.backward(); opt.step()
            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    binary = quantize_1bit(params)
                    r = sim(binary)["response"].cpu().numpy()
                m = eval_binary_metrics(r, main_lo, main_hi)
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_state = params.detach().clone()

        # Final eval
        eval_state = best_state if best_state is not None else params.detach()
        with torch.no_grad():
            binary = quantize_1bit(eval_state)
            r = sim(binary)["response"].cpu().numpy()
        m = eval_binary_metrics(r, main_lo, main_hi)
        m["used_early_stop"] = best_state is not None
        if best is None or m["worst"] > best["score"]["worst"]:
            best = {
                "binary": binary.cpu().numpy(),
                "resp": r,
                "score": m,
                "seed": seed,
                "recipe": recipe,
                "main_lo": main_lo,
                "main_hi": main_hi,
            }
    return best


# 4 representative configs (1-BIT now)
configs = [
    {"label": "(A) n=51 broadside, R119 (sweet spot)",
     "n": 51, "inc": 51, "freq": 38e9, "width": 10, "color": "#55a868"},
    {"label": "(B) n=51 wide cap (w=18deg), R129",
     "n": 51, "inc": 51, "freq": 38e9, "width": 18, "color": "#4c72b0"},
    {"label": "(C) n=51 inc=0 + 28GHz, R131 rescue",
     "n": 51, "inc": 0,  "freq": 28e9, "width": 10, "color": "#dd8452"},
    {"label": "(D) n=71 broadside, n=71 extrapolation",
     "n": 71, "inc": 51, "freq": 38e9, "width": 10, "color": "#c44e52",
     "n_restarts": 3},
]

print("=" * 100, flush=True)
print(f"Regenerating inference figure with HARD 1-bit (0 or pi only) quantization", flush=True)
print(f"Recipe selector + joint early-stop pipeline (R141)", flush=True)
print("=" * 100, flush=True)

results = []
for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/4] {cfg['label']}", flush=True)
    nr = cfg.get("n_restarts", n_restarts)
    r = optimize_1bit(cfg["n"], cfg["inc"], cfg["freq"], cfg["width"], nr)
    s = r["score"]
    flat_str = "OK" if s["flat_top"] else "fail ({} bins)".format(s["main_below_3"])
    print(f"  recipe: {r['recipe']['tier']} (rw={r['recipe']['rw']}, lam={r['recipe']['lam']})", flush=True)
    print(f"  worst={s['worst']:+.2f}, side_max={s['side_max']:+.2f}, "
          f"side_mean={s['side_mean']:+.2f}, ripple={s['ripple']:.2f}, flat={flat_str}", flush=True)
    # Sanity check: pattern should ONLY have 0 or 1
    unique_vals = np.unique(r["binary"])
    print(f"  pattern unique values: {unique_vals.tolist()}  (must be [0., 1.] for 1-bit)", flush=True)
    results.append((cfg, r))


# Render: 4 rows x 3 columns (binary pattern | far-field | distribution)
fig, axes = plt.subplots(4, 3, figsize=(17, 14))

# Black-and-white colormap for unambiguous binary display
binary_cmap = mcolors.ListedColormap(["white", "black"])

for row, (cfg, r) in enumerate(results):
    pat = r["binary"]
    resp = r["resp"]
    s = r["score"]
    n = cfg["n"]
    main_lo, main_hi = r["main_lo"], r["main_hi"]
    main_arr = resp[main_lo:main_hi]
    side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    main_lo_deg = -90 + main_lo * 0.5
    main_hi_deg = -90 + main_hi * 0.5

    # Col 0: BINARY pattern (black=1=pi, white=0=phase 0)
    ax = axes[row, 0]
    im = ax.imshow(pat, cmap=binary_cmap, vmin=0, vmax=1, aspect="equal",
                   interpolation="nearest")
    on_rate = pat.mean() * 100
    ax.set_title(f"binary pattern ({n} x {n}, 1-bit, seed={r['seed']})\n"
                 f"black=pi, white=0  (on-rate {on_rate:.1f}%)", fontsize=10)
    ax.set_xticks([0, n//2, n-1])
    ax.set_yticks([0, n//2, n-1])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1])
    cbar.set_ticklabels(["0  (phase=0)", "1  (phase=pi)"])

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
    ax.set_xlim(-90, 90); ax.set_ylim(-50, 5)
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("response (dB)")
    flat_str = "OK" if s["flat_top"] else f"FAIL ({s['main_below_3']}/{main_hi-main_lo})"
    ax.set_title(f"far-field — worst={s['worst']:+.2f} dB, flat-top: {flat_str}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # Col 2: distribution histogram
    ax = axes[row, 2]
    ax.hist(side_arr, bins=40, color=cfg["color"], alpha=0.7, edgecolor="black", label="sidelobe")
    ax.hist(main_arr, bins=15, color="lightgreen", alpha=0.8, edgecolor="black", label="main")
    ax.axvline(s["side_mean"], color="purple", linewidth=2, linestyle=":",
               label=f"side_mean={s['side_mean']:+.1f}")
    ax.axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    ax.set_xlim(-50, 5)
    ax.set_xlabel("response (dB)"); ax.set_ylabel("count")
    ax.set_title(f"distribution — side_mean={s['side_mean']:+.2f}, ripple={s['ripple']:.2f}",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # Row label
    fig.text(0.005, 1.0 - (row + 0.5) / 4, cfg["label"], rotation=90,
             ha="center", va="center", fontsize=11, fontweight="bold")

fig.suptitle("R141 deployment pipeline at 4 representative configs — 1-BIT (0 or pi ONLY)\n"
             "Selector + joint early-stop, hardware-deployable",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0.01, 0, 1, 0.96])
out = "outputs/report_fig5_inference_1bit.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved: {out}", flush=True)
