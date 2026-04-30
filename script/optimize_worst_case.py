"""
Round 64 — Worst-case loss optimizer (Chebyshev / minimax 風格)

新 loss: maximize min(main) - max(side)
用 logsumexp 平滑 min/max。

對比舊 loss (max-max)，新 loss 強迫 main region 整片貼上蓋。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config


def soft_max(x: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    return (1.0 / beta) * torch.logsumexp(beta * x, dim=0)


def soft_min(x: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    return -(1.0 / beta) * torch.logsumexp(-beta * x, dim=0)


def worst_case_loss(
    resp: torch.Tensor,
    main_mask: torch.Tensor,
    beta: float = 5.0,
    ripple_weight: float = 0.0,
) -> torch.Tensor:
    """maximize min(main) - max(side); 可選 ripple penalty."""
    main_min = soft_min(resp[main_mask], beta)
    side_max = soft_max(resp[~main_mask], beta)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max(resp[main_mask], beta)
        ripple = main_max - main_min
        loss = loss + ripple_weight * ripple
    return loss


def evaluate_full(resp_np: np.ndarray, main_lo: int, main_hi: int) -> dict:
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "headline": float(main.max() - side.max()),
        "worst": float(main.min() - side.max()),
        "main_min": float(main.min()),
        "main_max": float(main.max()),
        "main_ripple": float(main.max() - main.min()),
        "side_max": float(side.max()),
        "side_mean": float(side.mean()),
        "main_below_3": int((main < -3.0).sum()),
        "main_total": len(main),
    }


def run_one(
    sim: RISSimulator,
    element_num: int,
    main_lo: int,
    main_hi: int,
    seed: int,
    steps: int,
    lr: float,
    beta: float,
    ripple_weight: float,
    device: str,
) -> tuple[np.ndarray, dict, dict]:
    torch.manual_seed(seed)
    N = element_num**2
    params = nn.Parameter(torch.rand(N, device=device) * 2.0)
    opt = torch.optim.Adam([params], lr=lr)
    best_loss = float("inf")
    best_params = params.detach().clone()

    for step in range(steps):
        opt.zero_grad()
        resp = sim(params.reshape(element_num, element_num))["response"]
        mask = torch.zeros_like(resp, dtype=torch.bool)
        mask[main_lo:main_hi] = True
        loss = worst_case_loss(resp, mask, beta=beta, ripple_weight=ripple_weight)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

    with torch.no_grad():
        phase = (best_params * torch.pi) % (2 * torch.pi)
        bin_pat = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float().reshape(
            element_num, element_num
        )
        cont_resp = sim(params.reshape(element_num, element_num))["response"].cpu().numpy()
        bin_resp = sim(bin_pat)["response"].cpu().numpy()

    cont_metrics = evaluate_full(cont_resp, main_lo, main_hi)
    bin_metrics = evaluate_full(bin_resp, main_lo, main_hi)
    return bin_pat.cpu().numpy(), cont_metrics, bin_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=41)
    p.add_argument("--freq", type=float, default=38e9)
    p.add_argument("--inc_theta", type=float, default=51.0)
    p.add_argument("--plateau_start", type=int, default=137)
    p.add_argument("--plateau_w", type=int, default=80)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=5.0,
                   help="logsumexp sharpness; higher = closer to true min/max")
    p.add_argument("--ripple_weight", type=float, default=0.0)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default="outputs/r64_worst_case")
    args = p.parse_args()

    config.device = args.device
    sim = RISSimulator(element_num=args.element_num, freq_hz=args.freq, inc_theta_deg=args.inc_theta)

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== worst-case loss: n={args.element_num}, freq={args.freq/1e9:.1f} GHz, "
          f"inc={args.inc_theta}, plateau {args.plateau_start}-{args.plateau_start + args.plateau_w}, "
          f"beta={args.beta}, ripple_w={args.ripple_weight} ===\n")
    print(f"{'seed':>4} | {'cont_worst':>10} | {'cont_main_min':>13} | {'cont_main_max':>13} | "
          f"{'bin_worst':>9} | {'bin_main_min':>12} | {'bin_ripple':>10} | {'bin_below3':>10}")
    print("-" * 110)
    rows = []
    for seed in seeds:
        bin_pat, cont, binm = run_one(
            sim, args.element_num, args.plateau_start, args.plateau_start + args.plateau_w,
            seed, args.steps, args.lr, args.beta, args.ripple_weight, args.device,
        )
        np.save(out / f"bin_seed{seed}.npy", bin_pat)
        rows.append((seed, cont, binm))
        print(
            f"{seed:4d} | {cont['worst']:+10.2f} | {cont['main_min']:+13.2f} | "
            f"{cont['main_max']:+13.2f} | {binm['worst']:+9.2f} | "
            f"{binm['main_min']:+12.2f} | {binm['main_ripple']:10.2f} | "
            f"{binm['main_below_3']:>4}/{binm['main_total']}"
        )

    print()
    best = max(rows, key=lambda r: r[2]["worst"])
    print(f"best worst-case (binary): seed={best[0]} → worst_supp={best[2]['worst']:+.2f} dB")
    print(f"  headline (max-max):      {best[2]['headline']:+.2f} dB")
    print(f"  main_min:                {best[2]['main_min']:+.2f} dB")
    print(f"  main_max:                {best[2]['main_max']:+.2f} dB")
    print(f"  main ripple:             {best[2]['main_ripple']:.2f} dB")
    print(f"  side_max:                {best[2]['side_max']:+.2f} dB")
    print(f"  main < -3 dB count:      {best[2]['main_below_3']}/{best[2]['main_total']}")


if __name__ == "__main__":
    main()
