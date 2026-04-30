"""
Round 91 — End-to-End Methodology Demo

挑具體 deployment spec, 跑 patch-methodology 推薦 pipeline:
1. Free-phase parameterization (R57)
2. Worst-case + ripple penalty loss (R64)
3. Multi-restart 10 seeds (R44/R56)
4. Direct GD through real RIS sim (no surrogate, R90 dichotomy)
5. Optimal 1-bit quantization

Output: binary pattern + response + metrics + 視覺化, 作 patch team 的
"recommended deployment example" reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config


THETA_DEG = np.arange(-90, 90.1, 0.5)


def soft_max(x, beta=20.0):
    return (1/beta) * torch.logsumexp(beta * x, dim=-1)


def soft_min(x, beta=20.0):
    return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0):
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    main_min = soft_min(main, beta)
    side_max = soft_max(side, beta)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max(main, beta)
        loss = loss + ripple_weight * (main_max - main_min)
    return loss


def evaluate_metrics(resp_np, main_lo, main_hi):
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "headline_supp": float(main.max() - side.max()),
        "worst_supp": float(main.min() - side.max()),
        "main_max": float(main.max()),
        "main_min": float(main.min()),
        "main_ripple": float(main.max() - main.min()),
        "side_max": float(side.max()),
        "main_below_3dB": int((main < -3.0).sum()),
        "main_total": len(main),
        "flat_top_compliant": int((main < -3.0).sum()) == 0,
    }


def deploy_one_target(spec, n_restarts=10, gd_steps=1500, lr=0.05, ripple_weight=2.0,
                      device="cuda:0"):
    """Recommended deployment pipeline for single target spec."""
    config.device = device
    sim = RISSimulator(
        element_num=spec["n"],
        freq_hz=spec["freq_ghz"] * 1e9,
        inc_theta_deg=spec["inc_theta"],
    )
    main_lo = spec["main_lo"]
    main_hi = spec["main_hi"]

    print(f"\n--- Deploy spec: {spec['name']} ---")
    print(f"  freq={spec['freq_ghz']}GHz, n={spec['n']}, inc={spec['inc_theta']}, "
          f"main idx [{main_lo}, {main_hi}]")
    print(f"  ripple_weight={ripple_weight}, n_restarts={n_restarts}, gd_steps={gd_steps}")

    best = None
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(spec["n"], spec["n"], device=device) * 2.0)
        opt = torch.optim.Adam([params], lr=lr)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_weight)
            loss.backward()
            opt.step()

        # Optimal 1-bit quantization (R57)
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()

        m = evaluate_metrics(resp_bin, main_lo, main_hi)
        seed_results.append({"seed": seed, **m})
        if best is None or m["worst_supp"] > best["metrics"]["worst_supp"]:
            best = {
                "seed": seed,
                "binary_pattern": binary.cpu().numpy(),
                "response": resp_bin,
                "metrics": m,
            }
        print(f"  seed {seed}: worst={m['worst_supp']:+.2f} dB, ripple={m['main_ripple']:.2f}, "
              f"flat-top={'yes' if m['flat_top_compliant'] else 'no'}")

    return best, seed_results


def render_deployment(best, spec, out_path):
    pat = best["binary_pattern"]
    resp = best["response"]
    m = best["metrics"]
    main_lo, main_hi = spec["main_lo"], spec["main_hi"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(pat, cmap="binary", vmin=0, vmax=1, aspect="equal")
    axes[0].set_title(f"Binary Pattern {spec['n']}×{spec['n']}\n"
                      f"on-rate={pat.mean()*100:.1f}%, seed={best['seed']}")
    axes[0].set_xlabel("element x")
    axes[0].set_ylabel("element y")

    axes[1].plot(THETA_DEG, resp, "b-", linewidth=1.4)
    axes[1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5,
                     color="green", alpha=0.15, label="main beam region")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(-3, color="red", linewidth=0.6, linestyle="--", label="-3 dB cap")
    axes[1].set_ylim(-50, 5)
    axes[1].set_xlabel("θ (deg)")
    axes[1].set_ylabel("response (dB)")
    axes[1].set_title(
        f"Response curve\n"
        f"worst supp={m['worst_supp']:+.2f}, ripple={m['main_ripple']:.2f}, "
        f"flat-top={'yes' if m['flat_top_compliant'] else 'no'}",
    )
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3)

    side_resp = np.delete(resp, np.arange(main_lo, main_hi))
    main_resp = resp[main_lo:main_hi]
    axes[2].hist(side_resp, bins=30, color="steelblue", alpha=0.7, label="sidelobe", edgecolor="black")
    axes[2].hist(main_resp, bins=30, color="lightgreen", alpha=0.7, label="main beam", edgecolor="black")
    axes[2].axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    axes[2].set_xlabel("response (dB)")
    axes[2].set_title("Response distribution")
    axes[2].legend(fontsize=9)

    fig.suptitle(
        f"Recommended Deployment Demo — {spec['name']}\n"
        f"Pipeline: free-phase + worst-case loss + multi-restart × 10 seeds + 1-bit quantize",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"  saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n_restarts", type=int, default=10)
    p.add_argument("--gd_steps", type=int, default=1500)
    args = p.parse_args()

    # Three representative deployment specs
    specs = [
        {
            "name": "narrow_steering_38GHz",
            "freq_ghz": 38.0, "n": 31, "inc_theta": 51.0,
            "main_lo": 173, "main_hi": 187,  # 14 samples = 7° wide
            "ripple_weight": 0.0,  # steering, allow ripple
        },
        {
            "name": "flat_top_38GHz",
            "freq_ghz": 38.0, "n": 41, "inc_theta": 51.0,
            "main_lo": 162, "main_hi": 192,  # 30 samples = 15° wide
            "ripple_weight": 2.0,  # flat-top, suppress ripple
        },
        {
            "name": "off_axis_28GHz",
            "freq_ghz": 28.0, "n": 31, "inc_theta": 51.0,
            "main_lo": 110, "main_hi": 130,  # 20 samples at θ_c=-30°
            "ripple_weight": 1.0,
        },
    ]

    out_dir = Path("outputs/r91_deployment_demos")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Round 91 — End-to-End Methodology Demo")
    print("=" * 70)

    for spec in specs:
        rw = spec.pop("ripple_weight")
        best, seed_results = deploy_one_target(
            spec, n_restarts=args.n_restarts, gd_steps=args.gd_steps,
            ripple_weight=rw, device=args.device,
        )
        render_deployment(best, spec, out_dir / f"{spec['name']}.png")
        np.save(out_dir / f"{spec['name']}_pattern.npy", best["binary_pattern"])
        np.save(out_dir / f"{spec['name']}_response.npy", best["response"])

        # Stats
        worsts = [r["worst_supp"] for r in seed_results]
        print(f"\n  Stats for {spec['name']} (across {len(seed_results)} seeds):")
        print(f"    best worst_supp:  {best['metrics']['worst_supp']:+.2f}")
        print(f"    median:           {np.median(worsts):+.2f}")
        print(f"    mean ± std:       {np.mean(worsts):+.2f} ± {np.std(worsts):.2f}")
        print(f"    flat-top hit:     {sum(1 for r in seed_results if r['flat_top_compliant'])}/{len(seed_results)}")


if __name__ == "__main__":
    main()
