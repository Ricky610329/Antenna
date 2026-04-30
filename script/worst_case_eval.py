"""
Round 64 — 多 metric 評估器，暴露 max-max loss 的虛胖。

對任意 binary pattern 計算：
- headline_supp:  max(main) − max(side)   ← R57-R63 用的虛胖 metric
- worst_supp:     min(main) − max(side)   ← 真實可部署 metric
- main_ripple:    max(main) − min(main)   ← 主波束平坦度
- side_mean:      mean(side)              ← sidelobe 平均能量
- main_below_3dB: count of main points < -3 dB  ← 違反帽蓋的點數
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from antenna.ris import RISSimulator
from antenna.utils.config import config


def evaluate_pattern(
    pattern: np.ndarray | torch.Tensor,
    sim: RISSimulator,
    main_lo: int,
    main_hi: int,
) -> dict:
    """對 binary pattern 算所有 metric。"""
    if isinstance(pattern, np.ndarray):
        pattern = torch.from_numpy(pattern).float().to(config.device)
    with torch.no_grad():
        resp = sim(pattern)["response"].cpu().numpy()

    main = resp[main_lo:main_hi]
    side = np.delete(resp, np.arange(main_lo, main_hi))

    main_max = float(main.max())
    main_min = float(main.min())
    side_max = float(side.max())
    side_mean = float(side.mean())

    return {
        "headline_supp": main_max - side_max,
        "worst_supp": main_min - side_max,
        "main_max": main_max,
        "main_min": main_min,
        "side_max": side_max,
        "main_ripple": main_max - main_min,
        "side_mean": side_mean,
        "main_below_3dB": int((main < -3.0).sum()),
        "main_below_1dB": int((main < -1.0).sum()),
        "main_total": len(main),
        "resp": resp,
    }


def regenerate_r63_record(device: str = "cuda:0") -> np.ndarray:
    """重跑 R63 best record (38 GHz × n=41 × seed=0) 拿到 binary pattern。"""
    config.device = device
    sim = RISSimulator(element_num=41, freq_hz=38e9, inc_theta_deg=51.0)

    target = torch.full((361,), -25.0, device=device)
    target[137:217] = 0.0

    torch.manual_seed(0)
    params = torch.nn.Parameter(torch.rand(41**2, device=device) * 2.0)
    opt = torch.optim.Adam([params], lr=0.05)
    best_loss = float("inf")
    best_params = params.detach().clone()

    main_lo, main_hi = 137, 217
    for step in range(3000):
        opt.zero_grad()
        resp = sim(params.reshape(41, 41))["response"]
        beta = 5.0
        mask = torch.zeros_like(resp, dtype=torch.bool)
        mask[main_lo:main_hi] = True
        main_soft = (1.0 / beta) * torch.logsumexp(beta * resp[mask], dim=0)
        side_soft = (1.0 / beta) * torch.logsumexp(beta * resp[~mask], dim=0)
        loss = -(main_soft - side_soft)  # OLD max-max loss
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

    with torch.no_grad():
        phase = (best_params * torch.pi) % (2 * torch.pi)
        bin_pat = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float().reshape(41, 41)
    return bin_pat.cpu().numpy(), sim, main_lo, main_hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    print("=== Round 64: R63 record 在新 metric 下的真實表現 ===\n")
    pattern, sim, main_lo, main_hi = regenerate_r63_record(args.device)
    metrics = evaluate_pattern(pattern, sim, main_lo, main_hi)

    print(f"配置: 38 GHz × n=41 × inc=+51° × broadside × main beam region 80 samples")
    print()
    print(f"  headline supp (R63 報告):     max(main) - max(side) = {metrics['headline_supp']:+.2f} dB")
    print(f"  worst supp (真實可部署):      min(main) - max(side) = {metrics['worst_supp']:+.2f} dB")
    print()
    print(f"  main max (peak):              {metrics['main_max']:+.2f} dB")
    print(f"  main min (worst point):       {metrics['main_min']:+.2f} dB")
    print(f"  main ripple:                  {metrics['main_ripple']:.2f} dB")
    print(f"  side max (worst sidelobe):    {metrics['side_max']:+.2f} dB")
    print(f"  side mean:                    {metrics['side_mean']:+.2f} dB")
    print()
    print(f"  main 區內 < -3 dB 的角度數:   {metrics['main_below_3dB']} / {metrics['main_total']}")
    print(f"  main 區內 < -1 dB 的角度數:   {metrics['main_below_1dB']} / {metrics['main_total']}")
    print()
    headline = metrics["headline_supp"]
    worst = metrics["worst_supp"]
    delta = headline - worst
    print(f"==> 虛胖差距: {delta:+.2f} dB  ({headline:.2f} - {worst:.2f})")
    print(f"    這就是 R57-R63 max-max loss 的 systematic overestimation。")

    # save
    out = Path("outputs/r64_eval")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "r63_pattern.npy", pattern)
    np.save(out / "r63_response.npy", metrics["resp"])
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
