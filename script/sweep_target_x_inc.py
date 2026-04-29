"""探 target 方向 × inc_θ 配對：是否每個 target 都有 sweet spot inc_θ。

Round 32 發現 +11.82 dB 只在 broadside (θ_center=-1.5°) × inc_θ=+60° 達到。
其他 target 方向在同一 inc_θ 只 +0.6~+6.7 dB。

物理推測：specular reflection 在 -inc_θ 方向；target 越遠離 specular 越容易做
directional shaping。不同 target 應有不同最佳 inc_θ。

實驗：5 target plateau × 5 inc_θ × 5.6 GHz × 19×19 × seed 0:
- target plateau center θ ∈ {-50°, -25°, 0°, +25°, +50°}
- inc_θ ∈ {-60°, -30°, 0°, +30°, +60°}

如果 target × inc_θ heatmap 有對角線結構（target 越偏正、inc_θ 越偏負），
就有 specular-avoidance 物理規律。
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
        resp = sim(hard)["response"].cpu().numpy()
        main_idx = np.arange(main_lo, main_hi)
        side_idx = np.array([i for i in range(len(resp)) if i not in set(main_idx.tolist())])
        mp = float(resp[main_idx].max())
        sm = float(resp[side_idx].max())
    return mp - sm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=19)
    p.add_argument("--freq", type=float, default=5.6e9)
    p.add_argument("--target_thetas", type=float, nargs="+", default=[-50, -25, 0, 25, 50])
    p.add_argument("--inc_thetas", type=float, nargs="+", default=[-60, -30, 0, 30, 60])
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_target_x_inc")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.zeros((len(args.target_thetas), len(args.inc_thetas)))
    total = len(args.target_thetas) * len(args.inc_thetas)
    cnt = 0
    for i, t_theta in enumerate(args.target_thetas):
        # target plateau idx 對應 θ_center = t_theta：idx = (theta + 90) / 0.5
        center_idx = int(round((t_theta + 90) / 0.5))
        plateau_start = max(0, center_idx - args.plateau_w // 2)
        target_np = np.full(361, -20., dtype=np.float32)
        target_np[plateau_start:plateau_start + args.plateau_w] = 0.
        target = torch.tensor(target_np, device=config.device)
        for j, inc in enumerate(args.inc_thetas):
            cnt += 1
            sim = RISSimulator(element_num=args.element_num, freq_hz=args.freq, inc_theta_deg=inc)
            supp = gd_one(sim, target, args.element_num,
                          plateau_start, plateau_start + args.plateau_w,
                          args.steps, 0.05, args.seed)
            Z[i, j] = supp
            logger.info(f"[{cnt}/{total}] target_θ={t_theta:+.0f}°, inc_θ={inc:+.0f}°: "
                        f"suppression={supp:+.2f} dB")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=-2, vmax=12)
    plt.colorbar(im, ax=ax, label="suppression (dB)")
    ax.set_xticks(range(len(args.inc_thetas)))
    ax.set_xticklabels([f"{t:+.0f}°" for t in args.inc_thetas])
    ax.set_yticks(range(len(args.target_thetas)))
    ax.set_yticklabels([f"{t:+.0f}°" for t in args.target_thetas])
    ax.set_xlabel("inc_θ"); ax.set_ylabel("target_θ_center")
    ax.set_title(f"target × inc_θ sweep ({args.element_num}×{args.element_num}, "
                 f"{args.freq / 1e9:.1f} GHz, GD only seed=0)")
    for i, _ in enumerate(args.target_thetas):
        for j, _ in enumerate(args.inc_thetas):
            v = Z[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if v < 4 or v > 9 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)

    # Markdown
    lines = [f"# target × inc_θ sweep ({args.element_num}×{args.element_num}, "
             f"{args.freq / 1e9:.1f} GHz, GD seed={args.seed})", "",
             "| target_θ \\ inc_θ | " + " | ".join([f"{t:+.0f}°" for t in args.inc_thetas]) + " |",
             "|" + "|".join(["---"] * (len(args.inc_thetas) + 1)) + "|"]
    for i, t in enumerate(args.target_thetas):
        row = [f"{t:+.0f}°"] + [f"{Z[i, j]:+.2f}" for j in range(len(args.inc_thetas))]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## 每個 target 的最佳 inc_θ")
    for i, t in enumerate(args.target_thetas):
        best_j = int(np.argmax(Z[i]))
        lines.append(f"- target_θ={t:+.0f}°: best inc_θ={args.inc_thetas[best_j]:+.0f}° "
                     f"({Z[i, best_j]:+.2f} dB)")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
