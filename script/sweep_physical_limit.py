"""對 N 組不同 (plateau_start, plateau_w) 跑 direct GD，畫物理可達 suppression 分布。

**動機**：generator-based 的 −0.46 dB 上限已驗證為架構限制。但 direct GD per-target
能達 +2.76 dB（plateau 140-180）。物理上不同方向的 target，suppression 上限會
有差異——靠近鏡面反射方向會比較容易、離得遠的會更難。本 script 量化這個分布。

用法：
    python script/sweep_physical_limit.py --steps 1500 --n_runs 24
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
    steps: int = 1500, lr: float = 0.05, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    pattern_size = element_num * element_num
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
        hard_resp = sim(hard)["response"].cpu().numpy()
        main_idx = np.arange(main_lo, min(main_hi, len(hard_resp)))
        side_idx = np.array([i for i in range(len(hard_resp)) if i not in set(main_idx.tolist())])
        mp = float(hard_resp[main_idx].max()) if len(main_idx) else float("nan")
        sm = float(hard_resp[side_idx].max()) if len(side_idx) else float("nan")
    return {"main_peak": mp, "side_max": sm, "suppression": mp - sm,
            "on_rate": float(hard.mean()), "best_loss": best_loss,
            "hard_resp": hard_resp, "pattern": hard.cpu().numpy()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_widths", type=int, default=4, help="幾種 plateau 寬度（20/30/40/60 系列）")
    p.add_argument("--n_positions", type=int, default=8, help="幾個 plateau 起點位置")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_physical")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    sim = RISSimulator(element_num=args.element_num)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # plateau 寬度與位置 grid
    plateau_widths = np.linspace(20, 60, args.n_widths).astype(int)
    plateau_starts = np.linspace(60, 280, args.n_positions).astype(int)
    THETA_DEG = np.arange(-90, 90.1, 0.5)

    results = []
    total = len(plateau_widths) * len(plateau_starts)
    cnt = 0
    for w in plateau_widths:
        for st in plateau_starts:
            cnt += 1
            main_lo, main_hi = st, st + w
            response_size = 361
            target_np = np.full(response_size, -20.0, dtype=np.float32)
            target_np[main_lo:min(main_hi, response_size)] = 0.0
            target = torch.tensor(target_np, device=config.device)
            info = design_one(
                sim, target, element_num=args.element_num,
                main_lo=main_lo, main_hi=main_hi,
                steps=args.steps, lr=args.lr, seed=args.seed,
            )
            theta_main_center = -90 + (main_lo + w / 2) * 0.5
            results.append({
                "plateau_w": int(w), "plateau_start": int(st),
                "theta_center": float(theta_main_center),
                "suppression": info["suppression"],
                "main_peak": info["main_peak"],
                "side_max": info["side_max"],
                "on_rate": info["on_rate"],
            })
            logger.info(
                f"[{cnt:3d}/{total}] w={w}, start={st} (θ_center={theta_main_center:+.1f}°): "
                f"suppression={info['suppression']:+.2f} dB, on={info['on_rate']:.0%}"
            )

    # Heatmap
    Z = np.zeros((len(plateau_widths), len(plateau_starts)))
    for r in results:
        i = list(plateau_widths).index(r["plateau_w"])
        j = list(plateau_starts).index(r["plateau_start"])
        Z[i, j] = r["suppression"]

    fig, ax = plt.subplots(figsize=(11, 4))
    theta_centers = [-90 + (s + w / 2) * 0.5 for s in plateau_starts for w in plateau_widths[:1]]
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   extent=[plateau_starts[0], plateau_starts[-1], plateau_widths[0], plateau_widths[-1]],
                   vmin=-5, vmax=5)
    plt.colorbar(im, ax=ax, label="suppression (dB)")
    ax.set_xlabel("plateau start idx (theta_center 軸 ≈ -90 + 0.5×idx)")
    ax.set_ylabel("plateau width (samples)")
    ax.set_title("Direct GD physical limit — suppression heatmap (binary RIS 15×15, 28 GHz)")
    # 標記每格的數值
    for r in results:
        i = list(plateau_widths).index(r["plateau_w"])
        j = list(plateau_starts).index(r["plateau_start"])
        ax.text(plateau_starts[j], plateau_widths[i], f"{r['suppression']:+.1f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(r["suppression"]) > 3 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)
    logger.info(f"heatmap → {out_dir}/heatmap.png")

    # Suppression vs theta_center 折線圖（每 width 一條）
    fig, ax = plt.subplots(figsize=(10, 4))
    for w in plateau_widths:
        xs = []; ys = []
        for r in results:
            if r["plateau_w"] == w:
                xs.append(r["theta_center"]); ys.append(r["suppression"])
        ax.plot(xs, ys, marker="o", label=f"plateau width = {w} samples")
    ax.axhline(0, color="gray", alpha=0.3)
    ax.set_xlabel("main beam center theta (deg)")
    ax.set_ylabel("suppression (dB)")
    ax.set_title("Physical-limit suppression vs main beam direction")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vs_theta.png", dpi=110)
    plt.close(fig)

    # Markdown summary
    sup_arr = np.array([r["suppression"] for r in results])
    lines = ["# Physical-limit sweep — direct GD per-target", "",
             f"setting: 15×15 RIS, 28 GHz, inc θ=-40°/φ=90°, "
             f"{args.steps} steps, lr={args.lr}", "",
             f"runs: {len(results)} (widths={list(plateau_widths)}, "
             f"starts={list(plateau_starts)})", "",
             f"**suppression**: mean={sup_arr.mean():+.2f}, "
             f"min={sup_arr.min():+.2f}, max={sup_arr.max():+.2f}", "",
             "| width | start | θ_center | suppression dB | on% |",
             "|-------|-------|----------|----------------|-----|"]
    for r in sorted(results, key=lambda x: -x["suppression"]):
        lines.append(
            f"| {r['plateau_w']} | {r['plateau_start']} | "
            f"{r['theta_center']:+.1f}° | {r['suppression']:+.2f} | {r['on_rate']:.0%} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
