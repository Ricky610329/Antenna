"""試激進 SA schedules 是否能突破 +11.82 deeper basin。

Round 28 發現 5.6 GHz × 19×19 的 GD 命中率 10%——seed 0 直接 GD 達 +11.82，
但其他 9 seeds 經 SA 仍卡 +6.84~+8.57。+11.82 與 +9 之間有 deeper attraction
basin，SA 一般 schedule（flip_n=3, T0=20）跨不過去。

實驗：對固定 sub-optimal seed (seed 1, GD=+5.82) 試多種激進 SA schedule：
- 標準: flip_n=3, T0=20, 8000 steps, reheat=2
- 大 flip: flip_n=10, T0=50, 15000 steps, reheat=2
- 超大 flip: flip_n=20, T0=100, 15000 steps, reheat=4
- 階梯式: flip_n 從 20→10→3 漸減

如果某個 schedule 從 +5.82 直推到 +11，就找到突破方法了。

用法：
    python script/benchmark_sa_aggressive.py --device cuda:0
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config

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


def staged_sa(sim, target, init, main_lo, main_hi, total_steps, T0_list, flip_n_list, reheat_per_stage):
    """多階段 SA：先大 flip_n 大 T0 跨 basin，後小 flip_n 細調。"""
    n_stages = len(T0_list)
    pattern = init.clone()
    best_pattern = pattern.clone()
    _, best_supp = sa_mod.evaluate(sim, pattern, target, main_lo, main_hi)
    steps_per_stage = total_steps // n_stages

    for stage, (T0, flip_n) in enumerate(zip(T0_list, flip_n_list)):
        logger.info(f"  stage {stage + 1}/{n_stages}: T0={T0}, flip_n={flip_n}, steps={steps_per_stage}")
        sa_pat, _ = sa_mod.sa_finetune(
            sim, target, best_pattern,
            main_lo=main_lo, main_hi=main_hi,
            steps=steps_per_stage, T0=T0, T_final=0.001,
            flip_n=flip_n, log_every=steps_per_stage + 1,
            reheat_cycles=reheat_per_stage,
        )
        _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
        if sa_s > best_supp:
            best_supp = sa_s
            best_pattern = sa_pat
        logger.info(f"    → suppression={sa_s:+.2f}, best so far={best_supp:+.2f}")
    return best_pattern, best_supp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=19)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--freq", type=float, default=5.6e9)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--seed", type=int, default=1, help="哪個 GD seed 當 init（建議用 round 28 sub-optimal one）")
    p.add_argument("--gd_steps", type=int, default=1500)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/benchmark_sa_aggressive")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    target_np = np.full(361, -20.0, dtype=np.float32)
    target_np[args.plateau_start:args.plateau_start + args.plateau_w] = 0.0
    target = torch.tensor(target_np, device=config.device)
    main_lo, main_hi = args.plateau_start, args.plateau_start + args.plateau_w
    sim = RISSimulator(element_num=args.element_num, freq_hz=args.freq, inc_theta_deg=args.inc_theta)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # GD init
    init_pat, gd_supp = gd_one(sim, target, args.element_num, main_lo, main_hi,
                                args.gd_steps, 0.05, args.seed)
    logger.info(f"GD seed={args.seed}: suppression={gd_supp:+.2f} dB（init）")

    schedules = [
        ("std reheat=2", dict(steps=8000, T0=20, flip_n=3, reheat_cycles=2)),
        ("big flip", dict(steps=15000, T0=50, flip_n=10, reheat_cycles=2)),
        ("huge flip", dict(steps=15000, T0=100, flip_n=20, reheat_cycles=4)),
    ]
    results = [("init GD", gd_supp)]
    for name, kwargs in schedules:
        logger.info(f"\n=== {name}: {kwargs} ===")
        sa_pat, _ = sa_mod.sa_finetune(
            sim, target, init_pat,
            main_lo=main_lo, main_hi=main_hi, T_final=0.001,
            log_every=kwargs["steps"] + 1,
            **kwargs,
        )
        _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
        gain = sa_s - gd_supp
        results.append((name, sa_s))
        logger.info(f"  → suppression={sa_s:+.2f} dB (gain {gain:+.2f})")

    # Staged
    logger.info("\n=== staged: flip 20→10→3 ===")
    _, staged_supp = staged_sa(
        sim, target, init_pat, main_lo, main_hi,
        total_steps=20000,
        T0_list=[100, 30, 10],
        flip_n_list=[20, 10, 3],
        reheat_per_stage=2,
    )
    results.append(("staged 20→10→3", staged_supp))

    # Markdown
    lines = [f"# Aggressive SA schedule benchmark — seed={args.seed} on 5.6 GHz × {args.element_num}×{args.element_num}",
             "",
             f"Setting: inc_θ={args.inc_theta}°, plateau {main_lo}-{main_hi}",
             f"GD seed init: suppression={gd_supp:+.2f} dB",
             "",
             "| schedule | suppression | gain over GD |",
             "|----------|-------------|--------------|"]
    for name, supp in results:
        lines.append(f"| {name} | {supp:+.2f} | {supp - gd_supp:+.2f} |")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
