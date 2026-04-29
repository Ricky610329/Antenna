"""對 binary RIS pattern 做 simulated annealing fine-tune。

**動機**：round 13 顯示 GD on logits + post-quantize 有強烈 local-minima 問題
（5 restarts 中 1 個達 +9.51 dB，4 個卡 +2~+6 dB）。SA 在 binary domain 做
random pixel flip，理論上能突破 GD 卡死的 local minima。

策略：對每個 GD restart 結果做 N 步 SA fine-tune，希望從 +5 dB 推到 +9 dB
級別，達到「每個 restart 都接近上限」的效果。

用法：
    # 對 design_pattern_for_target 的輸出做 SA
    python script/binary_sa_finetune.py \
        --pattern outputs/per_target_design/best_combo_15x15/pattern_binary.npy \
        --plateau_start 154 --plateau_w 46 --inc_theta 60 \
        --steps 5000 --device cuda:0
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def evaluate(sim, pattern: torch.Tensor, target: torch.Tensor,
             main_lo: int, main_hi: int) -> tuple[float, float]:
    """回傳 (loss, suppression)"""
    with torch.no_grad():
        resp = sim(pattern)["response"]
        loss = custom_loss_tolerance(
            resp, target, sidelobe_threshold=-25.0,
            main_target=0.0, main_weight=5.0,
        )
        resp_np = resp.cpu().numpy()
        main_idx = np.arange(main_lo, min(main_hi, len(resp_np)))
        side_idx = np.array([i for i in range(len(resp_np)) if i not in set(main_idx.tolist())])
        mp = float(resp_np[main_idx].max())
        sm = float(resp_np[side_idx].max())
    return float(loss.item()), mp - sm


def sa_finetune(
    sim: RISSimulator, target: torch.Tensor, init_pattern: torch.Tensor,
    *, main_lo: int, main_hi: int,
    steps: int = 5000, T0: float = 50.0, T_final: float = 0.01,
    flip_n: int = 1, log_every: int = 500,
    reheat_cycles: int = 1,
) -> tuple[torch.Tensor, list[float]]:
    """SA：每步隨機翻 flip_n 個 pixel，按 Metropolis 接受。

    用 loss 作為 cost（越低越好），用 exp(-(L_new - L_old)/T) 接受 worse moves。

    reheat_cycles=1 時是單輪 cooling，>1 時把 steps 平分為多輪 cooling，每輪
    從 T0 降到 T_final，回到 best pattern（不是當前 pattern）作為新 cycle 起點。
    這個 schedule 能 escape local minima 比單輪更穩。
    """
    n = init_pattern.shape[0]
    pattern = init_pattern.clone()
    cur_loss, cur_supp = evaluate(sim, pattern, target, main_lo, main_hi)
    best_loss = cur_loss; best_supp = cur_supp
    best_pattern = pattern.clone()

    history = []
    rng = np.random.default_rng(0)

    cycles = max(1, int(reheat_cycles))
    steps_per_cycle = max(1, steps // cycles)
    total_step = 0

    for cycle in range(cycles):
        # Reheat 開始：每輪用同個 T0→T_final schedule，但從 best pattern 重啟
        if cycle > 0:
            pattern = best_pattern.clone()
            cur_loss, cur_supp = best_loss, best_supp
        for s in range(steps_per_cycle):
            total_step += 1
            T = T0 * (T_final / T0) ** (s / max(steps_per_cycle - 1, 1))
            flat = pattern.flatten()
            idxs = rng.choice(flat.numel(), size=flip_n, replace=False)
            flat_new = flat.clone()
            flat_new[idxs] = 1.0 - flat_new[idxs]
            new_pattern = flat_new.reshape(n, n)
            new_loss, new_supp = evaluate(sim, new_pattern, target, main_lo, main_hi)

            delta = new_loss - cur_loss
            accept = (delta < 0) or (rng.random() < math.exp(-delta / T))
            if accept:
                pattern = new_pattern
                cur_loss = new_loss
                cur_supp = new_supp
                if new_supp > best_supp:
                    best_supp = new_supp
                    best_loss = new_loss
                    best_pattern = pattern.clone()

            history.append(cur_supp)
            if total_step % log_every == 0:
                logger.info(
                    f"  cycle {cycle + 1}/{cycles} step {s + 1:5d}/{steps_per_cycle}: "
                    f"T={T:.3f}, cur={cur_supp:+.2f}, best={best_supp:+.2f}"
                )

    return best_pattern, history


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", type=str, required=True, help="初始 binary pattern (.npy)")
    p.add_argument("--element_num", type=int, default=None, help="如果不指定從 pattern shape 推")
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--inc_theta", type=float, default=None)
    p.add_argument("--inc_phi", type=float, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--T0", type=float, default=50.0)
    p.add_argument("--T_final", type=float, default=0.01)
    p.add_argument("--flip_n", type=int, default=1)
    p.add_argument("--reheat_cycles", type=int, default=1,
                   help="多輪 reheat：把 steps 切成 N 輪，每輪從 best pattern 重啟")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    init_np = np.load(args.pattern)
    if args.element_num is None:
        args.element_num = init_np.shape[0]
    init_pattern = torch.tensor(init_np, dtype=torch.float32, device=config.device)

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")

    sim_kwargs = {"element_num": args.element_num}
    if args.inc_theta is not None: sim_kwargs["inc_theta_deg"] = args.inc_theta
    if args.inc_phi is not None: sim_kwargs["inc_phi_deg"] = args.inc_phi
    sim = RISSimulator(**sim_kwargs)

    target_np = np.full(361, -20.0, dtype=np.float32)
    target_np[args.plateau_start:args.plateau_start + args.plateau_w] = 0.0
    target = torch.tensor(target_np, device=config.device)

    main_lo = args.plateau_start
    main_hi = main_lo + args.plateau_w

    init_loss, init_supp = evaluate(sim, init_pattern, target, main_lo, main_hi)
    logger.info(f"初始 pattern: suppression={init_supp:+.2f} dB, loss={init_loss:.2f}")

    best_pattern, history = sa_finetune(
        sim, target, init_pattern,
        main_lo=main_lo, main_hi=main_hi,
        steps=args.steps, T0=args.T0, T_final=args.T_final,
        flip_n=args.flip_n,
        reheat_cycles=args.reheat_cycles,
    )
    final_loss, final_supp = evaluate(sim, best_pattern, target, main_lo, main_hi)
    gain = final_supp - init_supp
    logger.success(
        f"SA 完成: init={init_supp:+.2f} → final={final_supp:+.2f} dB "
        f"(gain {gain:+.2f} dB)"
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.pattern).parent / "sa_finetune"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pattern_binary.npy", best_pattern.cpu().numpy().astype(np.uint8))

    # 視覺化
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(init_np, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"Init (GD): suppression={init_supp:+.2f} dB")
    axes[0].axis("off")
    axes[1].imshow(best_pattern.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title(f"Post-SA: suppression={final_supp:+.2f} dB (gain {gain:+.2f})")
    axes[1].axis("off")
    axes[2].plot(history, linewidth=0.8)
    axes[2].set_title("SA suppression trajectory")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("dB")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sa_finetune.png", dpi=110)
    plt.close(fig)
    logger.info(f"輸出 → {out_dir}/")


if __name__ == "__main__":
    main()
