"""為**單一指定 target** 直接搜尋最佳 RIS binary pattern（單目標最佳化工具）。

**這是 round 1-8 實驗後得出的最務實工具**：
- 11 個 generator-based run 最佳僅 -0.46 dB（v6），距物理上限 +3.05 dB 還差 3.5 dB
- Hamming 統計顯示 generator 對任意輸入都收斂到相同 pattern（conditioning failure）
- 真正的 root cause 是 generator 必須妥協 32 個 target → 找通用最佳 pattern
- **若使用者每次部署只服務一個 target**，per-target direct GD 才是正解：5 秒、+3 dB

用法（指定 plateau 位置）：
    python script/design_pattern_for_target.py --plateau_start 140 --plateau_w 40
    # 或 by reflection angle 度數：
    python script/design_pattern_for_target.py --reflection_idx 160 --plateau_w 40

輸出：
    outputs/per_target_design/<tag>/pattern.png
    outputs/per_target_design/<tag>/response.png
    outputs/per_target_design/<tag>/pattern_binary.npy   ← 直接給硬體用
    outputs/per_target_design/<tag>/summary.md
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


def design(
    sim: RISSimulator,
    target: torch.Tensor,
    *,
    element_num: int,
    main_lo: int,
    main_hi: int,
    steps: int = 2000,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[torch.Tensor, dict]:
    """連續 GD 搜尋 + 後處理量化（已驗證為最佳路線）。"""
    torch.manual_seed(seed)
    pattern_size = element_num * element_num
    logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
    optimizer = torch.optim.Adam([logits], lr=lr)

    best_loss = float("inf")
    best_logits = logits.detach().clone()
    history = []

    for step in range(steps):
        optimizer.zero_grad()
        soft = torch.sigmoid(logits)
        pat = soft.reshape(element_num, element_num)
        resp = sim(pat)["response"]
        loss = custom_loss_tolerance(
            resp, target,
            sidelobe_threshold=-25.0,
            main_target=0.0,
            main_weight=5.0,
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_logits = logits.detach().clone()

    # Post-quantize: best soft → hard
    with torch.no_grad():
        best_hard = (torch.sigmoid(best_logits) > 0.5).float().reshape(element_num, element_num)
        hard_resp = sim(best_hard)["response"].cpu().numpy()
        main_idx = np.arange(main_lo, min(main_hi, len(hard_resp)))
        side_idx = np.array([i for i in range(len(hard_resp)) if i not in set(main_idx.tolist())])
        main_peak = float(hard_resp[main_idx].max()) if len(main_idx) else float("nan")
        side_max = float(hard_resp[side_idx].max()) if len(side_idx) else float("nan")
        suppression = main_peak - side_max
        on_rate = float(best_hard.mean())

    return best_hard, {
        "best_loss": best_loss,
        "main_peak": main_peak,
        "side_max": side_max,
        "suppression": suppression,
        "on_rate": on_rate,
        "history": history,
        "hard_resp": hard_resp,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--plateau_start", type=int, default=140, help="main beam plateau 起點 idx (0-360)")
    p.add_argument("--plateau_w", type=int, default=40, help="main beam plateau 寬度 (sample 數)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")

    # 構造 target
    main_lo = args.plateau_start
    main_hi = main_lo + args.plateau_w
    response_size = 361
    side, center = -20.0, 0.0
    target_np = np.full(response_size, side, dtype=np.float32)
    target_np[main_lo:main_hi] = center
    target = torch.tensor(target_np, device=config.device)

    sim = RISSimulator(element_num=args.element_num)

    logger.info(f"設計 target: plateau idx {main_lo}-{main_hi} ({args.plateau_w} samples)")
    pattern, info = design(
        sim, target,
        element_num=args.element_num,
        main_lo=main_lo, main_hi=main_hi,
        steps=args.steps, lr=args.lr, seed=args.seed,
    )

    tag = f"plateau_{main_lo}_{args.plateau_w}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/per_target_design") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 存 pattern
    np.save(out_dir / "pattern_binary.npy", pattern.cpu().numpy().astype(np.uint8))

    # 視覺化
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(pattern.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
    on = int(pattern.sum())
    axes[0].set_title(f"Optimized binary pattern\n{on}/{args.element_num**2} on ({info['on_rate']:.0%})")
    axes[0].axis("off")

    THETA_DEG = np.arange(-90, 90.1, 0.5)
    axes[1].plot(THETA_DEG, info["hard_resp"], label="actual", linewidth=1.5)
    axes[1].plot(THETA_DEG, target_np, label="target", linewidth=1.5, alpha=0.6)
    axes[1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5, color="green", alpha=0.1, label="main beam region")
    axes[1].set_title(
        f"Response — main_peak={info['main_peak']:+.2f}, side_max={info['side_max']:+.2f}, "
        f"suppression={info['suppression']:+.2f} dB"
    )
    axes[1].set_xlabel("theta (deg)"); axes[1].set_ylabel("dB"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "design.png", dpi=110)
    plt.close(fig)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(info["history"], linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("GD loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=110)
    plt.close(fig)

    summary = (
        f"# Per-target pattern design — plateau {main_lo}–{main_hi}\n\n"
        f"- **element_num**: {args.element_num}×{args.element_num}\n"
        f"- **main beam region**: theta {-90 + main_lo * 0.5:+.1f}° ~ {-90 + main_hi * 0.5:+.1f}°\n"
        f"- **steps**: {args.steps}, lr={args.lr}, seed={args.seed}\n"
        f"- **best loss**: {info['best_loss']:.4f}\n"
        f"- **on-rate**: {info['on_rate']:.0%}\n"
        f"- **main_peak**: {info['main_peak']:+.2f} dB\n"
        f"- **side_max**: {info['side_max']:+.2f} dB\n"
        f"- **suppression**: {info['suppression']:+.2f} dB\n\n"
        f"## 跟 generator-based 路線比較\n"
        f"- v6 (best generator-based): -0.46 dB suppression\n"
        f"- this run: {info['suppression']:+.2f} dB suppression\n"
        f"- gain over generator: {info['suppression'] - (-0.46):+.2f} dB\n\n"
        f"## 部署用\n"
        f"binary pattern 已存於 `pattern_binary.npy`，shape ({args.element_num}, {args.element_num})\n"
        f"元素值 ∈ {{0, 1}}，0=phase 0、1=phase π。\n"
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
