"""量化 SA fine-tune 對 GD 結果的「保底機率」。

實驗設計：對固定 target，跑 N 次 GD（不同 seed）拿 sub-optimal local minima
的分布；對每個結果做 SA fine-tune，看 SA 後分布如何變化。

關鍵指標：
- GD-only mean / std / 達到 +7 dB 的比率
- GD+SA mean / std / 達到 +7 dB 的比率
- SA 平均 gain

用法：
    python script/benchmark_gd_vs_sa.py --n_seeds 10 --device cuda:0
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
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--threshold_dB", type=float, default=7.0,
                   help="達標門檻：判定 +X dB 以上算「達上限級別」")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/benchmark_gd_vs_sa")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    sim = RISSimulator(element_num=args.element_num, inc_theta_deg=args.inc_theta)

    target_np = np.full(361, -20.0, dtype=np.float32)
    target_np[args.plateau_start:args.plateau_start + args.plateau_w] = 0.0
    target = torch.tensor(target_np, device=config.device)
    main_lo = args.plateau_start
    main_hi = main_lo + args.plateau_w

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    gd_supps = []
    sa_supps = []
    sa_gains = []
    for s in range(args.n_seeds):
        # GD only
        hard, gd_supp = gd_one(
            sim, target, args.element_num, main_lo, main_hi,
            args.steps, args.lr, seed=s,
        )
        # SA fine-tune
        sa_pat, _ = sa_mod.sa_finetune(
            sim, target, hard,
            main_lo=main_lo, main_hi=main_hi,
            steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
            flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,  # 不 log
        )
        _, sa_supp = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
        gain = sa_supp - gd_supp
        gd_supps.append(gd_supp)
        sa_supps.append(sa_supp)
        sa_gains.append(gain)
        logger.info(
            f"seed {s:2d}: GD={gd_supp:+.2f} → SA={sa_supp:+.2f} dB (gain {gain:+.2f})"
        )

    gd_arr = np.array(gd_supps); sa_arr = np.array(sa_supps); gain_arr = np.array(sa_gains)
    th = args.threshold_dB
    gd_pass = (gd_arr >= th).sum()
    sa_pass = (sa_arr >= th).sum()
    n = args.n_seeds

    summary = [
        f"# Benchmark: GD-only vs GD+SA ({n} seeds)",
        "",
        f"Setting: {args.element_num}×{args.element_num}, inc_θ={args.inc_theta}°, "
        f"plateau {main_lo}-{main_hi}",
        f"GD: {args.steps} steps lr={args.lr}; "
        f"SA: {args.sa_steps} steps T0={args.sa_T0} flip_n={args.sa_flip_n}",
        "",
        f"| seed | GD (dB) | GD+SA (dB) | gain |",
        f"|------|---------|-----------|------|",
    ]
    for s, g, sa, ga in zip(range(n), gd_supps, sa_supps, sa_gains):
        summary.append(f"| {s} | {g:+.2f} | {sa:+.2f} | {ga:+.2f} |")
    summary += [
        "",
        f"## 統計",
        f"- **GD-only**: mean={gd_arr.mean():+.2f}, std={gd_arr.std():.2f}, "
        f"min={gd_arr.min():+.2f}, max={gd_arr.max():+.2f}",
        f"- **GD+SA**: mean={sa_arr.mean():+.2f}, std={sa_arr.std():.2f}, "
        f"min={sa_arr.min():+.2f}, max={sa_arr.max():+.2f}",
        f"- **SA gain**: mean={gain_arr.mean():+.2f}, max={gain_arr.max():+.2f}",
        "",
        f"## 達標率（≥{th} dB）",
        f"- GD-only: {gd_pass}/{n} = {gd_pass / n:.0%}",
        f"- GD+SA: {sa_pass}/{n} = {sa_pass / n:.0%}",
        f"- **改善率**: {(sa_pass - gd_pass) / n:+.0%}",
    ]
    text = "\n".join(summary)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())

    # 視覺化
    fig, ax = plt.subplots(figsize=(10, 5))
    seeds = list(range(n))
    width = 0.35
    ax.bar([s - width / 2 for s in seeds], gd_supps, width, label="GD only", color="tab:orange")
    ax.bar([s + width / 2 for s in seeds], sa_supps, width, label="GD + SA", color="tab:green")
    ax.axhline(th, color="red", linestyle="--", alpha=0.5, label=f"threshold {th} dB")
    ax.axhline(0, color="gray", alpha=0.3)
    ax.set_xlabel("seed"); ax.set_ylabel("suppression (dB)")
    ax.set_title(f"GD-only vs GD+SA ({n} seeds)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
