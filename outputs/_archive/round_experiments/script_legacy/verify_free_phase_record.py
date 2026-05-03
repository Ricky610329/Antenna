"""
Round 57 verification — free-phase GD + 1-bit quantization 是否真能達 +21 dB。

獨立 evaluator，明確存出 binary pattern + 各步驟 suppression。
也測試 SA fine-tune 是否進一步提升。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def supp(resp: np.ndarray, main_lo: int, main_hi: int) -> tuple[float, float, float]:
    main = float(resp[main_lo:main_hi].max())
    side = float(np.delete(resp, np.arange(main_lo, main_hi)).max())
    return main - side, main, side


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=13)
    p.add_argument("--freq", type=float, default=28e9)
    p.add_argument("--inc_theta", type=float, default=51.0)
    p.add_argument("--plateau_start", type=int, default=137)
    p.add_argument("--plateau_w", type=int, default=80)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default="outputs/r57_free_phase_verify")
    p.add_argument("--sa_steps", type=int, default=8000)
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

    # === Step 1: free-phase GD with direct loss ===
    torch.manual_seed(args.seed)
    N = args.element_num**2
    params = nn.Parameter(torch.rand(N, device=args.device) * 2.0)
    opt = torch.optim.Adam([params], lr=args.lr)

    best_loss = float("inf")
    best_params = params.detach().clone()
    for step in range(args.steps):
        opt.zero_grad()
        pat = params.reshape(args.element_num, args.element_num)
        resp = sim(pat)["response"]
        mask = torch.zeros_like(resp, dtype=torch.bool)
        mask[main_lo:main_hi] = True
        beta = 5.0
        main_soft = (1.0 / beta) * torch.logsumexp(beta * resp[mask], dim=0)
        side_soft = (1.0 / beta) * torch.logsumexp(beta * resp[~mask], dim=0)
        loss = -(main_soft - side_soft)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

    with torch.no_grad():
        free_pat = best_params.reshape(args.element_num, args.element_num)
        free_resp = sim(free_pat)["response"].cpu().numpy()
        free_supp, free_main, free_side = supp(free_resp, main_lo, main_hi)

    print(f"=== seed={args.seed} ===")
    print(f"[free continuous] suppression={free_supp:+.2f} (main={free_main:+.2f}, side={free_side:+.2f})")

    # === Step 2: optimal 1-bit quantization (closest to {0, π}) ===
    with torch.no_grad():
        phase = (best_params * torch.pi) % (2 * torch.pi)
        bin_pat = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float().reshape(
            args.element_num, args.element_num)
        bin_resp = sim(bin_pat)["response"].cpu().numpy()
        bin_supp, bin_main, bin_side = supp(bin_resp, main_lo, main_hi)

    print(f"[1-bit quantize] suppression={bin_supp:+.2f} (main={bin_main:+.2f}, side={bin_side:+.2f})")
    print(f"  quantization loss: {free_supp - bin_supp:+.2f} dB")

    # === Step 3: SA fine-tune from the 1-bit start ===
    try:
        from binary_sa_finetune import sa_finetune, evaluate
        sa_pat, _ = sa_finetune(
            sim, target, bin_pat,
            main_lo=main_lo, main_hi=main_hi,
            steps=args.sa_steps, T0=20.0, T_final=0.001,
            flip_n=3, log_every=args.sa_steps + 1,
            reheat_cycles=2,
        )
        with torch.no_grad():
            sa_resp = sim(sa_pat)["response"].cpu().numpy()
            sa_supp, sa_main, sa_side = supp(sa_resp, main_lo, main_hi)
        print(f"[+ SA fine-tune] suppression={sa_supp:+.2f} (main={sa_main:+.2f}, side={sa_side:+.2f})")
        print(f"  SA gain: {sa_supp - bin_supp:+.2f} dB")
    except Exception as e:
        print(f"SA failed: {e}")
        sa_pat = bin_pat
        sa_supp = bin_supp

    # === save ===
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"free_pattern_seed{args.seed}.npy", best_params.detach().cpu().numpy())
    np.save(out / f"bin_pattern_seed{args.seed}.npy", bin_pat.cpu().numpy())
    if 'sa_pat' in dir():
        np.save(out / f"sa_pattern_seed{args.seed}.npy", sa_pat.cpu().numpy())
    print(f"saved to {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "script")
    main()
