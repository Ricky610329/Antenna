"""對單一 target 直接做 gradient descent on pattern logits — 驗證 RIS 物理極限。

**動機**：v1–v4 的 generator 都對所有 target 給相似 pattern（conditioning failure）。
這套程式繞過 generator 完全，直接以 BinarySTE + RISSimulator 的可微路徑對單一目標
target 優化一個 pattern logits tensor。

如果 direct GD 也達不到合理 suppression，代表 RIS+target 配對本身有物理極限；
如果 direct GD 達得到，那 generator 是學習問題。

用法：
    python script/direct_pattern_search.py --steps 2000 --device cuda:0
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
from antenna.models.autograd import BinarySTE
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def direct_search(
    sim: RISSimulator,
    target: torch.Tensor,
    *,
    element_num: int,
    steps: int = 2000,
    lr: float = 0.05,
    binary_mode: bool = True,
    sidelobe_threshold: float = -25.0,
    main_target: float = 0.0,
    main_weight: float = 5.0,
    log_every: int = 100,
) -> tuple[torch.Tensor, list[float]]:
    """對單一 target 直接優化 pattern logits（不經 generator）。

    Returns:
        (best_pattern, loss_history): best_pattern 是 binarized 後的 (E, E) tensor
    """
    pattern_size = element_num * element_num
    # 隨機 init logits（≈ N(0, 1)）→ sigmoid 後 ≈ 0.5 中心
    logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
    optimizer = torch.optim.Adam([logits], lr=lr)

    loss_history = []
    best_loss = float("inf")
    best_logits = logits.detach().clone()

    for step in range(steps):
        optimizer.zero_grad()
        soft = torch.sigmoid(logits)
        if binary_mode:
            pat = BinarySTE.apply(soft).reshape(element_num, element_num)
        else:
            pat = soft.reshape(element_num, element_num)
        resp = sim(pat)["response"]

        loss = custom_loss_tolerance(
            resp,
            target,
            sidelobe_threshold=sidelobe_threshold,
            main_target=main_target,
            main_weight=main_weight,
        )
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_logits = logits.detach().clone()

        if (step + 1) % log_every == 0 or step == 0:
            with torch.no_grad():
                hard = (torch.sigmoid(logits) > 0.5).float().reshape(element_num, element_num)
                hard_resp = sim(hard)["response"].cpu().numpy()
                main_lo = 140
                main_hi = 180
                mp = float(hard_resp[main_lo:main_hi].max())
                sm = float(np.delete(hard_resp, slice(main_lo, main_hi)).max())
                supp = mp - sm
                on_rate = hard.mean().item()
            logger.info(
                f"step {step + 1:4d}: loss={loss.item():.2f}, best={best_loss:.2f}, "
                f"on={on_rate:.0%}, main_peak={mp:+.2f} dB, side_max={sm:+.2f} dB, "
                f"suppression={supp:+.2f} dB"
            )

    with torch.no_grad():
        best_pat = (torch.sigmoid(best_logits) > 0.5).float().reshape(element_num, element_num)
    return best_pat, loss_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element_num", type=int, default=15)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/direct_search",
    )
    args = parser.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    torch.manual_seed(args.seed)

    # 設定 pattern / target — 跟 train_ris_binary_v2 / phase2 一致
    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    sim = RISSimulator(element_num=args.element_num)
    target = AntennaResponse.target.concat().to(config.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run 1: binary mode（與 trainer 一致）──
    logger.info("=== Direct GD with BinarySTE (matches trainer) ===")
    best_pat_bin, hist_bin = direct_search(
        sim, target,
        element_num=args.element_num,
        steps=args.steps,
        lr=args.lr,
        binary_mode=True,
    )

    # ── Run 2: 連續模式（觀察沒有量化干擾的物理上限）──
    logger.info("\n=== Direct GD with continuous (no STE) ===")
    best_pat_cont, hist_cont = direct_search(
        sim, target,
        element_num=args.element_num,
        steps=args.steps,
        lr=args.lr,
        binary_mode=False,
    )

    # 評估 binary 結果
    with torch.no_grad():
        hard = best_pat_bin
        hard_resp = sim(hard)["response"].cpu().numpy()
        main_lo, main_hi = 140, 180
        mp = float(hard_resp[main_lo:main_hi].max())
        sm = float(np.delete(hard_resp, slice(main_lo, main_hi)).max())
        bin_supp = mp - sm
        bin_on = float(hard.mean())

        # 連續結果二值化後評估
        hard_cont = (best_pat_cont > 0.5).float()
        hard_cont_resp = sim(hard_cont)["response"].cpu().numpy()
        mp_c = float(hard_cont_resp[main_lo:main_hi].max())
        sm_c = float(np.delete(hard_cont_resp, slice(main_lo, main_hi)).max())
        cont_supp = mp_c - sm_c
        cont_on = float(hard_cont.mean())

    logger.success(
        f"\n========================================\n"
        f"Binary direct GD:    on={bin_on:.0%}, suppression={bin_supp:+.2f} dB\n"
        f"Continuous→hard:     on={cont_on:.0%}, suppression={cont_supp:+.2f} dB\n"
        f"========================================\n"
    )

    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(best_pat_bin.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Binary GD pattern  (on={bin_on:.0%}, supp={bin_supp:+.2f} dB)")
    axes[0, 0].axis("off")

    THETA_DEG = np.arange(-90, 90.1, 0.5)
    axes[0, 1].plot(THETA_DEG, hard_resp, label="actual", linewidth=1.5)
    axes[0, 1].plot(THETA_DEG, target.cpu().numpy(), label="target", linewidth=1.5, alpha=0.6)
    axes[0, 1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5, color="green", alpha=0.1)
    axes[0, 1].set_title(f"Binary GD response: main_peak={mp:+.2f}, side_max={sm:+.2f}")
    axes[0, 1].set_xlabel("theta (deg)"); axes[0, 1].set_ylabel("dB"); axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].imshow(hard_cont.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Continuous→hard pattern  (on={cont_on:.0%}, supp={cont_supp:+.2f} dB)")
    axes[1, 0].axis("off")
    axes[1, 1].plot(THETA_DEG, hard_cont_resp, label="actual", linewidth=1.5)
    axes[1, 1].plot(THETA_DEG, target.cpu().numpy(), label="target", linewidth=1.5, alpha=0.6)
    axes[1, 1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5, color="green", alpha=0.1)
    axes[1, 1].set_title(f"Continuous→hard response: main_peak={mp_c:+.2f}, side_max={sm_c:+.2f}")
    axes[1, 1].set_xlabel("theta (deg)"); axes[1, 1].set_ylabel("dB"); axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "direct_search.png"
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    logger.info(f"視覺化 → {fig_path}")

    # loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(hist_bin, label="binary STE", alpha=0.8)
    ax.semilogy(hist_cont, label="continuous", alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Direct GD loss curve")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
