"""探不同入射角 θ_i 對 RIS 可達 suppression 的影響。

RIS 預設 inc_theta=-40°, inc_phi=90°。物理直覺：入射方向決定 specular reflection
方向，當 target 與 specular reflection 對齊時 suppression 應最高。本 script 量化
這個關係。

用法：
    python script/sweep_incidence_angle.py --device cuda:0
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


def design_one(
    sim: RISSimulator, target: torch.Tensor,
    *, element_num: int, main_lo: int, main_hi: int,
    steps: int, lr: float, n_restarts: int, seed: int,
) -> dict:
    pattern_size = element_num * element_num
    best_supp = -np.inf
    best_pattern = None
    best_resp = None
    for r in range(n_restarts):
        torch.manual_seed(seed + r)
        logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
        optimizer = torch.optim.Adam([logits], lr=lr)
        best_loss_local = float("inf")
        best_logits_local = logits.detach().clone()
        for _ in range(steps):
            optimizer.zero_grad()
            soft = torch.sigmoid(logits)
            pat = soft.reshape(element_num, element_num)
            resp = sim(pat)["response"]
            loss = custom_loss_tolerance(
                resp, target, sidelobe_threshold=-25.0, main_target=0.0, main_weight=5.0,
            )
            loss.backward(); optimizer.step()
            if loss.item() < best_loss_local:
                best_loss_local = loss.item()
                best_logits_local = logits.detach().clone()
        with torch.no_grad():
            hard = (torch.sigmoid(best_logits_local) > 0.5).float().reshape(element_num, element_num)
            hard_resp = sim(hard)["response"].cpu().numpy()
            main_idx = np.arange(main_lo, min(main_hi, len(hard_resp)))
            mp = float(hard_resp[main_idx].max())
            sm = float(np.delete(hard_resp, main_idx).max())
            supp = mp - sm
            if supp > best_supp:
                best_supp = supp
                best_pattern = hard.cpu().numpy()
                best_resp = hard_resp
    return {"suppression": best_supp, "pattern": best_pattern, "hard_resp": best_resp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inc_thetas", type=float, nargs="+",
                   default=[-60, -40, -20, 0, 20, 40, 60])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_incidence")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 三組 target（左/中/右）
    target_configs = [
        ("left",   91,  46),
        ("center", 154, 46),
        ("right",  217, 46),
    ]

    results = {}
    for inc_th in args.inc_thetas:
        sim = RISSimulator(element_num=args.element_num, inc_theta_deg=inc_th)
        results[inc_th] = {}
        for name, st, w in target_configs:
            target_np = np.full(361, -20.0, dtype=np.float32)
            target_np[st:st + w] = 0.0
            target = torch.tensor(target_np, device=config.device)
            info = design_one(
                sim, target, element_num=args.element_num,
                main_lo=st, main_hi=st + w,
                steps=args.steps, lr=args.lr,
                n_restarts=args.n_restarts, seed=args.seed,
            )
            results[inc_th][name] = info
            logger.info(
                f"inc_θ={inc_th:+.0f}°, target={name}: suppression={info['suppression']:+.2f} dB"
            )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, _, _ in target_configs:
        xs = list(results.keys())
        ys = [results[t][name]["suppression"] for t in xs]
        ax.plot(xs, ys, marker="o", label=f"target {name}")
    ax.axvline(0, color="gray", alpha=0.3)
    ax.set_xlabel("incidence theta (deg)")
    ax.set_ylabel("suppression (dB)")
    ax.set_title(f"Suppression vs incidence θ ({args.element_num}×{args.element_num} RIS)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vs_inc_theta.png", dpi=110)
    plt.close(fig)

    # Heatmap
    targets_names = [n for n, _, _ in target_configs]
    Z = np.array([[results[t][n]["suppression"] for n in targets_names] for t in args.inc_thetas])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   extent=[0, len(targets_names), args.inc_thetas[0], args.inc_thetas[-1]],
                   vmin=-2, vmax=8)
    plt.colorbar(im, ax=ax, label="suppression (dB)")
    ax.set_xticks([i + 0.5 for i in range(len(targets_names))])
    ax.set_xticklabels(targets_names)
    ax.set_xlabel("target"); ax.set_ylabel("incidence θ (deg)")
    ax.set_title("Heatmap: suppression by inc_θ × target")
    for i, t in enumerate(args.inc_thetas):
        for j, n in enumerate(targets_names):
            v = Z[i, j]
            ax.text(j + 0.5, t, f"{v:+.1f}", ha="center", va="center",
                    fontsize=9, color="white" if v < 1 or v > 6 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)

    # Markdown
    lines = ["# 入射角 sweep — suppression vs inc_θ", "",
             f"Setting: {args.element_num}×{args.element_num}, "
             f"{args.steps} steps × {args.n_restarts} restarts (取最佳)", ""]
    lines += [
        "| inc_θ | target_left | target_center | target_right |",
        "|-------|-------------|---------------|--------------|",
    ]
    for t in args.inc_thetas:
        l = results[t]["left"]["suppression"]
        c = results[t]["center"]["suppression"]
        r = results[t]["right"]["suppression"]
        lines.append(f"| {t:+.0f}° | {l:+.2f} | {c:+.2f} | {r:+.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
