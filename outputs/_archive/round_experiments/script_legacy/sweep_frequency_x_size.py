"""探頻率 × element_num 二維對 RIS suppression 的影響。

Round 17 發現：15×15 在 28 GHz + inc_θ=+60° 是物理最佳。
推測：element spacing = λ/2 與 element_num 的乘積（aperture 大小）才是真正
變因。同一個 aperture 在不同頻率下需要不同 element_num。

實驗：3 frequencies (5.6/28/60 GHz) × 3 sizes (10/15/20) × 3 seeds × {GD, SA}

用法：
    python script/sweep_frequency_x_size.py --device cuda:0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config
import importlib.util

_spec = importlib.util.spec_from_file_location("_sa", Path(__file__).parent / "binary_sa_finetune.py")
sa_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sa_mod)


def gd_one(sim, target, element_num, main_lo, main_hi, steps, lr, seed):
    pattern_size = element_num * element_num
    torch.manual_seed(seed)
    logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
    optimizer = torch.optim.Adam([logits], lr=lr)
    best_loss = float("inf"); best_logits = logits.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad()
        soft = torch.sigmoid(logits)
        pat = soft.reshape(element_num, element_num)
        resp = sim(pat)["response"]
        loss = custom_loss_tolerance(
            resp, target, sidelobe_threshold=-25.0, main_target=0.0, main_weight=5.0,
        )
        loss.backward(); optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_logits = logits.detach().clone()
    with torch.no_grad():
        hard = (torch.sigmoid(best_logits) > 0.5).float().reshape(element_num, element_num)
        _, supp = sa_mod.evaluate(sim, hard, target, main_lo, main_hi)
    return hard, supp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frequencies", type=float, nargs="+", default=[5.6e9, 28e9, 60e9])
    p.add_argument("--element_nums", type=int, nargs="+", default=[10, 15, 20])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_freq_x_size")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaResponse.registerLabels("response", x="ris")
    target_np = np.full(361, -20.0, dtype=np.float32)
    target_np[args.plateau_start:args.plateau_start + args.plateau_w] = 0.0
    target = torch.tensor(target_np, device=config.device)
    main_lo, main_hi = args.plateau_start, args.plateau_start + args.plateau_w

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for freq in args.frequencies:
        results[freq] = {}
        for n in args.element_nums:
            AntennaPattern.setDefaultCoordinate((0, n, 0, n))
            sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=args.inc_theta)
            sa_supps = []
            for s in range(args.n_seeds):
                hard, gd_s = gd_one(sim, target, n, main_lo, main_hi, args.steps, args.lr, seed=s)
                sa_pat, _ = sa_mod.sa_finetune(
                    sim, target, hard, main_lo=main_lo, main_hi=main_hi,
                    steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
                    flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,
                )
                _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
                sa_supps.append(sa_s)
                logger.info(f"  freq={freq / 1e9:.1f}GHz, n={n}, seed {s}: GD={gd_s:+.2f} → SA={sa_s:+.2f}")
            results[freq][n] = sa_supps

    # Markdown
    lines = [
        f"# 頻率 × element_num sweep（GD+SA, {args.n_seeds} seeds, inc_θ={args.inc_theta}°）",
        "",
        "**SA mean (max) suppression in dB:**",
        "",
    ]
    header = ["frequency"] + [f"{n}×{n}" for n in args.element_nums]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for freq in args.frequencies:
        row = [f"{freq / 1e9:.1f} GHz"]
        for n in args.element_nums:
            arr = np.array(results[freq][n])
            row.append(f"{arr.mean():+.2f} ({arr.max():+.2f})")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 物理常數",
        f"Element spacing = λ/2",
        f"Apertures (n × spacing × wavelength_factor):",
    ]
    for freq in args.frequencies:
        wavelength = 3e8 / freq
        for n in args.element_nums:
            aperture = n * 0.5 * wavelength
            lines.append(f"- {freq / 1e9:.1f} GHz × {n}×{n}: aperture = {n * 0.5:.1f}λ = {aperture * 1000:.1f} mm")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Heatmap (mean suppression)
    Z = np.zeros((len(args.frequencies), len(args.element_nums)))
    for i, freq in enumerate(args.frequencies):
        for j, n in enumerate(args.element_nums):
            Z[i, j] = np.mean(results[freq][n])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=4, vmax=9)
    plt.colorbar(im, ax=ax, label="mean suppression (dB)")
    ax.set_xticks(range(len(args.element_nums)))
    ax.set_xticklabels([f"{n}×{n}" for n in args.element_nums])
    ax.set_yticks(range(len(args.frequencies)))
    ax.set_yticklabels([f"{f / 1e9:.1f} GHz" for f in args.frequencies])
    ax.set_title("Mean suppression: frequency × element_num (GD+SA)")
    for i, _ in enumerate(args.frequencies):
        for j, _ in enumerate(args.element_nums):
            ax.text(j, i, f"{Z[i, j]:+.2f}", ha="center", va="center",
                    color="white" if Z[i, j] < 5.5 or Z[i, j] > 8 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
