"""探 inc_θ × element_num 二維 heatmap — 量化 bimodal valley 移動規律。

Round 22 發現：
- inc_θ=+60° valley 在 17×17
- inc_θ=-40° valley 在 13×13
**Valley 隨 inc_θ 改變位置**。本 script 跑完整 2D grid 量化這個關係。

如果 valley 沿 (inc_θ, size) 空間中對角線跑，可能有 grating lobe 數學
公式可推導。如果沒明顯規律，則需要 case-by-case sweep。

用法：
    python script/sweep_inc_x_size_2d.py --device cuda:0
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


def gd_one(sim, target, n, main_lo, main_hi, steps, lr, seed):
    pattern_size = n * n
    torch.manual_seed(seed)
    logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
    optimizer = torch.optim.Adam([logits], lr=lr)
    best_loss = float("inf"); best_logits = logits.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad()
        soft = torch.sigmoid(logits)
        pat = soft.reshape(n, n)
        resp = sim(pat)["response"]
        loss = custom_loss_tolerance(
            resp, target, sidelobe_threshold=-25.0, main_target=0.0, main_weight=5.0,
        )
        loss.backward(); optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_logits = logits.detach().clone()
    with torch.no_grad():
        hard = (torch.sigmoid(best_logits) > 0.5).float().reshape(n, n)
        _, supp = sa_mod.evaluate(sim, hard, target, main_lo, main_hi)
    return hard, supp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inc_thetas", type=float, nargs="+",
                   default=[-60, -40, -20, 0, 20, 40, 60])
    p.add_argument("--element_nums", type=int, nargs="+", default=[11, 13, 15, 17, 19])
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--freq", type=float, default=28e9)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_inc_x_size_2d")
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

    Z = np.zeros((len(args.inc_thetas), len(args.element_nums)))
    total = len(args.inc_thetas) * len(args.element_nums)
    cnt = 0
    for i, inc_th in enumerate(args.inc_thetas):
        for j, n in enumerate(args.element_nums):
            cnt += 1
            AntennaPattern.setDefaultCoordinate((0, n, 0, n))
            sim = RISSimulator(element_num=n, freq_hz=args.freq, inc_theta_deg=inc_th)
            seeds_supp = []
            for s in range(args.n_seeds):
                hard, _ = gd_one(sim, target, n, main_lo, main_hi, args.steps, args.lr, seed=s)
                sa_pat, _ = sa_mod.sa_finetune(
                    sim, target, hard, main_lo=main_lo, main_hi=main_hi,
                    steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
                    flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,
                )
                _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
                seeds_supp.append(sa_s)
            Z[i, j] = np.mean(seeds_supp)
            logger.info(f"[{cnt}/{total}] inc={inc_th:+.0f}°, n={n}: SA mean={Z[i, j]:+.2f} dB")

    # Heatmap
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=2, vmax=10)
    plt.colorbar(im, ax=ax, label="suppression mean (dB)")
    ax.set_xticks(range(len(args.element_nums)))
    ax.set_xticklabels([f"{n}×{n}" for n in args.element_nums])
    ax.set_yticks(range(len(args.inc_thetas)))
    ax.set_yticklabels([f"{t:+.0f}°" for t in args.inc_thetas])
    ax.set_xlabel("element_num"); ax.set_ylabel("inc_θ")
    ax.set_title(f"2D sweep: inc_θ × size at {args.freq / 1e9:.1f} GHz (mean over {args.n_seeds} seeds)")
    for i, _ in enumerate(args.inc_thetas):
        for j, _ in enumerate(args.element_nums):
            v = Z[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if v < 4 or v > 8 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)

    # Summary
    lines = [f"# inc_θ × element_num 2D heatmap (freq={args.freq / 1e9:.1f} GHz)", "",
             f"Setting: plateau {main_lo}-{main_hi}, GD {args.steps} + SA {args.sa_steps} steps",
             f"Each cell averaged over {args.n_seeds} seeds.", ""]
    header = ["inc_θ \\ size"] + [f"{n}×{n}" for n in args.element_nums]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, t in enumerate(args.inc_thetas):
        row = [f"{t:+.0f}°"] + [f"{Z[i, j]:+.2f}" for j in range(len(args.element_nums))]
        lines.append("| " + " | ".join(row) + " |")
    # Best per row
    lines.append("")
    lines.append("## 各 inc_θ 最佳 size")
    for i, t in enumerate(args.inc_thetas):
        best_j = int(np.argmax(Z[i]))
        worst_j = int(np.argmin(Z[i]))
        lines.append(
            f"- inc_θ={t:+.0f}°: best={args.element_nums[best_j]}×{args.element_nums[best_j]} "
            f"({Z[i, best_j]:+.2f}), worst={args.element_nums[worst_j]}×{args.element_nums[worst_j]} "
            f"({Z[i, worst_j]:+.2f})"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
