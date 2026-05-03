"""
Round 57 — 1-bit binary vs full continuous phase comparison

Three modes for the same (freq, n, inc, width, target) record config:
  (a) sigmoid  — pattern in [0, 1] → phase in [0, π]   (half-circle, our default)
  (b) free     — phase in [0, 2π]                       (full continuous, lit upper bound)
  (c) binary   — {0, π}                                 (1-bit hardware)

文獻 3 dB quantization loss 是 (b) - (c)。我們要實證這個 gap。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def run_gd(sim, target, element_num, mode: str, seed: int, steps: int, lr: float, device: str,
           main_lo: int = None, main_hi: int = None, loss_kind: str = "tolerance"):
    """mode in {'sigmoid', 'free'}; loss_kind in {'tolerance', 'direct'}"""
    torch.manual_seed(seed)
    N = element_num**2
    if mode == "sigmoid":
        params = nn.Parameter(torch.randn(N, device=device))
        def to_pattern(p):
            return torch.sigmoid(p)
    elif mode == "free":
        params = nn.Parameter(torch.rand(N, device=device) * 2.0)
        def to_pattern(p):
            return p
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam([params], lr=lr)
    best_loss = float("inf")
    best_params = params.detach().clone()
    for step in range(steps):
        opt.zero_grad()
        pat = to_pattern(params)
        resp = sim(pat.reshape(element_num, element_num))["response"]
        if loss_kind == "tolerance":
            loss = custom_loss_tolerance(resp, target, sidelobe_threshold=-25.0, main_target=0.0, main_weight=5.0)
        elif loss_kind == "direct":
            # log-sum-exp soft-max over main and side. Maximize main - side.
            mask = torch.zeros_like(resp, dtype=torch.bool)
            mask[main_lo:main_hi] = True
            beta = 5.0
            main_soft = (1.0 / beta) * torch.logsumexp(beta * resp[mask], dim=0)
            side_soft = (1.0 / beta) * torch.logsumexp(beta * resp[~mask], dim=0)
            loss = -(main_soft - side_soft)
        else:
            raise ValueError(loss_kind)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()
    return best_params, best_loss


def supp(resp, main_lo, main_hi):
    main = resp[main_lo:main_hi].max()
    side = np.delete(resp, np.arange(main_lo, main_hi)).max()
    return float(main - side)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=13)
    p.add_argument("--freq", type=float, default=28e9)
    p.add_argument("--inc_theta", type=float, default=51.0)
    p.add_argument("--plateau_start", type=int, default=137)
    p.add_argument("--plateau_w", type=int, default=80)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device

    sim = RISSimulator(
        element_num=args.element_num,
        freq_hz=args.freq,
        inc_theta_deg=args.inc_theta,
    )

    target = torch.full((361,), -25.0, device=args.device)
    main_lo = args.plateau_start
    main_hi = args.plateau_start + args.plateau_w
    target[main_lo:main_hi] = 0.0

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        # mode (a) sigmoid: half-circle [0, π]
        sig_logits, _ = run_gd(sim, target, args.element_num, "sigmoid", seed, args.steps, args.lr, args.device)
        with torch.no_grad():
            sig_pat = torch.sigmoid(sig_logits).reshape(args.element_num, args.element_num)
            sig_resp = sim(sig_pat)["response"].cpu().numpy()
            bin_pat = (torch.sigmoid(sig_logits) > 0.5).float().reshape(args.element_num, args.element_num)
            bin_resp = sim(bin_pat)["response"].cpu().numpy()

        # mode (b) free phase: full [0, 2π] — try direct loss
        free_params, _ = run_gd(sim, target, args.element_num, "free", seed, args.steps, args.lr, args.device,
                                main_lo=main_lo, main_hi=main_hi, loss_kind="direct")
        with torch.no_grad():
            free_pat = free_params.reshape(args.element_num, args.element_num)
            free_resp = sim(free_pat)["response"].cpu().numpy()
            # binarize the free phase: phase < π/2 or phase > 3π/2 → 0, else π. Wraps the unit circle:
            phase = (free_params * torch.pi) % (2 * torch.pi)  # in [0, 2π)
            bin_from_free = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float().reshape(args.element_num, args.element_num)
            binf_resp = sim(bin_from_free)["response"].cpu().numpy()

        s_half = supp(sig_resp, main_lo, main_hi)
        s_bin = supp(bin_resp, main_lo, main_hi)
        s_free = supp(free_resp, main_lo, main_hi)
        s_binf = supp(binf_resp, main_lo, main_hi)
        rows.append((seed, s_half, s_bin, s_free, s_binf))
        print(
            f"seed={seed}: half-circle={s_half:+.2f} bin-from-half={s_bin:+.2f} "
            f"full-cont={s_free:+.2f} bin-from-full={s_binf:+.2f}"
        )

    print()
    print(f"{'seed':>4} | {'half-π':>7} | {'bin/half':>8} | {'full-2π':>7} | {'bin/full':>8} | "
          f"{'gap (full-binfull)':>18}")
    print("-" * 75)
    for seed, sh, sb, sf, sbf in rows:
        print(f"{seed:4d} | {sh:+7.2f} | {sb:+8.2f} | {sf:+7.2f} | {sbf:+8.2f} | {sf - sbf:+18.2f}")
    arr = np.array(rows)
    print(
        f"\nmean: half={arr[:, 1].mean():+.2f} bin-half={arr[:, 2].mean():+.2f} "
        f"full={arr[:, 3].mean():+.2f} bin-full={arr[:, 4].mean():+.2f}"
    )
    print(f"max: half={arr[:, 1].max():+.2f} bin-half={arr[:, 2].max():+.2f} "
          f"full={arr[:, 3].max():+.2f} bin-full={arr[:, 4].max():+.2f}")
    gap_mean = (arr[:, 3] - arr[:, 4]).mean()
    print(f"\ntheoretical 3 dB quantization loss check: full-cont vs bin-from-full = {gap_mean:+.2f} dB (mean)")


if __name__ == "__main__":
    main()
