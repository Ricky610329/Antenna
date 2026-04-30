"""
Round 77 — Surrogate-in-loop GD vs Real-sim GD comparison

對相同 test target，跑兩個 optimization:
1. real-sim GD: GD through differentiable RIS sim (R64 baseline)
2. surrogate-in-loop GD: GD through frozen surrogate (R72 CNN trained on dataset_v2)
   然後 final pattern 跑 real sim eval

如果 surrogate-in-loop 達到接近 real-sim 的 worst_supp，
證明 patch methodology (surrogate-in-loop) 在 RIS 上 self-consistent。

對 patch: 同樣架構，real-sim 替換為 HFSS-on-demand 即可。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config


# Import surrogate from train_surrogate.py
import sys
sys.path.insert(0, "script")
from train_surrogate import SurrogateCNN


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


def supp_metrics(resp_np, main_lo, main_hi):
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "worst": float(main.min() - side.max()),
        "ripple": float(main.max() - main.min()),
        "main_min": float(main.min()),
        "side_max": float(side.max()),
    }


def gd_through_real_sim(sim, n, main_lo, main_hi, ripple_w, n_seeds=5, steps=1500, lr=0.05, device="cuda:0"):
    """R64 baseline: real-sim GD with worst-case loss."""
    best = None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device=device) * 2.0)
        opt = torch.optim.Adam([params], lr=lr)
        for step in range(steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_w)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()
            real_resp = sim(binary)["response"].cpu().numpy()
        m = supp_metrics(real_resp, main_lo, main_hi)
        if best is None or m["worst"] > best["worst"]:
            best = {**m, "seed": seed, "binary": binary.cpu().numpy()}
    return best


def gd_through_surrogate(surrogate, sim_for_eval, n, max_n, freq, inc_theta, theta_c, target_w_deg,
                         ripple_w, n_seeds=5, steps=1500, lr=0.05, device="cuda:0"):
    """Same as above but loss computed through frozen surrogate (R76 methodology test)."""
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    main_lo, main_hi = build_target_idx(theta_c, target_w_deg)

    # Build padded mask
    offset = (max_n - n) // 2
    pad_mask = torch.zeros(max_n, max_n, device=device)
    pad_mask[offset:offset + n, offset:offset + n] = 1.0

    # config_vec for surrogate (matches RISDataset format in train_surrogate.py)
    cfg_vec = torch.tensor([
        freq / 100e9,         # 0.38 for 38 GHz
        n / 41.0,
        theta_c / 90.0,
        target_w_deg / 90.0,
        inc_theta / 90.0,
        ripple_w / 5.0,
    ], dtype=torch.float32, device=device).unsqueeze(0)

    best = None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        params_n = nn.Parameter(torch.rand(n, n, device=device) * 2.0)
        opt = torch.optim.Adam([params_n], lr=lr)
        for step in range(steps):
            opt.zero_grad()
            # quantize via STE-like: forward use binary, backward through params
            phase = (params_n * torch.pi) % (2 * torch.pi)
            soft_bin = torch.sigmoid(10 * torch.cos(phase))  # smooth approx of binary indicator
            # ↑ smoother approximation: cos(phase) ∈ [-1, 1], threshold ~0 → sigmoid≈step
            # But we want "binary={0,1}" representation.
            # Actually let's do simpler: just feed continuous to surrogate (OOD but smoother)
            pat_padded = torch.zeros(1, max_n, max_n, device=device)
            pat_padded[0, offset:offset+n, offset:offset+n] = soft_bin
            mask_b = pad_mask.unsqueeze(0)
            resp_pred = surrogate(pat_padded, mask_b, cfg_vec)
            loss = worst_case_loss(resp_pred[0], main_lo, main_hi, beta=20.0, ripple_weight=ripple_w)
            loss.backward()
            opt.step()
        # Final binary eval through REAL sim
        with torch.no_grad():
            phase = (params_n * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()
            real_resp = sim_for_eval(binary)["response"].cpu().numpy()
        m = supp_metrics(real_resp, main_lo, main_hi)
        if best is None or m["worst"] > best["worst"]:
            best = {**m, "seed": seed, "binary": binary.cpu().numpy()}
    return best


def build_target_idx(theta_c, width):
    sample_per_deg = 2
    center = int(round((theta_c + 90) * sample_per_deg))
    half = int(round(width * sample_per_deg / 2))
    return max(0, center - half), min(361, center + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate_pt", type=str, default="outputs/r72_cnn_v2/surrogate.pt")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()

    config.device = args.device
    device = args.device

    # Load surrogate
    surrogate = SurrogateCNN(channels=32, depth=4).to(device)
    surrogate.load_state_dict(torch.load(args.surrogate_pt, map_location=device))
    print(f"Loaded surrogate from {args.surrogate_pt}")

    # Test targets (subset of dataset_v2 covering different configs)
    targets = [
        # (freq_hz, n, theta_c, width_deg, inc_theta, ripple_weight)
        (38e9, 31, 0.0, 20.0, 51.0, 2.0),   # broadside flat-top
        (38e9, 31, -30.0, 10.0, 51.0, 2.0),  # off-axis narrow flat-top
        (28e9, 31, 0.0, 20.0, 51.0, 2.0),   # 28 GHz broadside flat-top
        (38e9, 41, 0.0, 10.0, 51.0, 2.0),   # n=41 narrow flat-top
        (38e9, 31, 30.0, 30.0, 51.0, 0.0),  # off-axis wide steering
    ]

    print(f"\n{'config':<45} | {'real-sim':>12} | {'surrogate':>12} | {'gap':>6}")
    print(f"{'(freq, n, θc, w, rw)':<45} | {'(R64)':>12} | {'(R76)':>12} | {'dB':>6}")
    print("-" * 90)

    for freq, n, tc, tw, inc, rw in targets:
        sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
        main_lo, main_hi = build_target_idx(tc, tw)

        # Method A: real-sim GD (R64)
        a = gd_through_real_sim(sim, n, main_lo, main_hi, rw, args.n_seeds, args.steps, device=device)

        # Method B: surrogate-in-loop GD (R76)
        b = gd_through_surrogate(surrogate, sim, n, max_n=41,
                                  freq=freq, inc_theta=inc, theta_c=tc,
                                  target_w_deg=tw, ripple_w=rw,
                                  n_seeds=args.n_seeds, steps=args.steps, device=device)

        cfg_str = f"f={freq/1e9:.0f}G n={n} θc={tc:+.0f} w={tw:.0f} rw={rw}"
        gap = a["worst"] - b["worst"]
        print(f"{cfg_str:<45} | {a['worst']:+12.2f} | {b['worst']:+12.2f} | {gap:+6.2f}")


if __name__ == "__main__":
    main()
