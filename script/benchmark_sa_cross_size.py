"""驗證 SA 對不同 RIS 邊長的效果 — 解開 round 13「25×25 不如 15×15」謎題。

Round 13 發現：25×25 + inc_θ=+60° 只達 +7.42 dB（比 15×15 +9.51 還低）。
假說：25×25 cells 多 GD landscape 更崎嶇 → 卡 deeper local minima。
SA 應能跨越這些 local minima，讓 25×25 也達 +9 dB 級別。

實驗：4 sizes × 5 seeds × {GD only, GD+SA}

用法：
    python script/benchmark_sa_cross_size.py --device cuda:0
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
    p.add_argument("--element_nums", type=int, nargs="+", default=[10, 15, 20, 25])
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/benchmark_sa_cross_size")
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
    for n in args.element_nums:
        AntennaPattern.setDefaultCoordinate((0, n, 0, n))
        sim = RISSimulator(element_num=n, inc_theta_deg=args.inc_theta)
        gd_supps = []
        sa_supps = []
        for s in range(args.n_seeds):
            hard, gd_s = gd_one(sim, target, n, main_lo, main_hi, args.steps, args.lr, seed=s)
            sa_pat, _ = sa_mod.sa_finetune(
                sim, target, hard, main_lo=main_lo, main_hi=main_hi,
                steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
                flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,
            )
            _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
            gd_supps.append(gd_s); sa_supps.append(sa_s)
            logger.info(f"  n={n}, seed {s}: GD={gd_s:+.2f} → SA={sa_s:+.2f}")
        results[n] = {"gd": gd_supps, "sa": sa_supps}

    # Markdown
    lines = [f"# SA cross-size benchmark — {args.n_seeds} seeds × {len(args.element_nums)} sizes",
             "",
             f"Setting: inc_θ={args.inc_theta}°, plateau {main_lo}-{main_hi}",
             f"GD: {args.steps} steps; SA: {args.sa_steps} steps T0={args.sa_T0} flip_n={args.sa_flip_n}",
             "",
             "| element_num | GD mean | GD max | SA mean | SA max | SA-GD gain |",
             "|-------------|---------|--------|---------|--------|------------|"]
    for n in args.element_nums:
        gd = np.array(results[n]["gd"])
        sa = np.array(results[n]["sa"])
        gain = (sa - gd).mean()
        lines.append(
            f"| {n}×{n} | {gd.mean():+.2f} | {gd.max():+.2f} | "
            f"{sa.mean():+.2f} | {sa.max():+.2f} | {gain:+.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    data = []
    labels = []
    for i, n in enumerate(args.element_nums):
        positions += [i * 3, i * 3 + 1]
        data += [results[n]["gd"], results[n]["sa"]]
        labels += [f"{n}×{n}\nGD", f"{n}×{n}\nSA"]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, labels=labels)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor("tab:orange" if "GD" in lab else "tab:green")
    ax.set_ylabel("suppression (dB)")
    ax.set_title(f"SA cross-size effect ({args.n_seeds} seeds, inc_θ={args.inc_theta}°)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot.png", dpi=110)
    plt.close(fig)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
