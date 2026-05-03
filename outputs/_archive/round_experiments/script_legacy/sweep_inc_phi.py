"""探 inc_phi (入射方位角) 對 RIS 可達 suppression 的影響。

之前所有 sweep 都固定 inc_phi=90°（RISSimulator default）。本 script 補上
phi 影響——inc_phi 控制入射波在 RIS 表面 x/y 兩軸上的投影比例：
- phi=0°: 入射在 x-z 平面內
- phi=90°: 入射在 y-z 平面內（default）
- phi 變化 → element x/y 軸 grating 結構與入射方向 misalignment

5 phi × 5 sizes × 2 seeds × {GD+SA reheat=2} = 50 designs (~17 min on GPU)

用法：
    python script/sweep_inc_phi.py --device cuda:0
"""

import argparse
import importlib.util
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
    p.add_argument("--inc_phis", type=float, nargs="+", default=[0, 45, 90, 135, 180])
    p.add_argument("--element_nums", type=int, nargs="+", default=[11, 13, 15, 17, 19])
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--freq", type=float, default=28e9)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--sa_reheat_cycles", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_inc_phi")
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

    Z = np.zeros((len(args.inc_phis), len(args.element_nums)))
    total = len(args.inc_phis) * len(args.element_nums)
    cnt = 0
    for i, phi in enumerate(args.inc_phis):
        for j, n in enumerate(args.element_nums):
            cnt += 1
            AntennaPattern.setDefaultCoordinate((0, n, 0, n))
            sim = RISSimulator(element_num=n, freq_hz=args.freq,
                               inc_theta_deg=args.inc_theta, inc_phi_deg=phi)
            seeds_supp = []
            for s in range(args.n_seeds):
                hard, _ = gd_one(sim, target, n, main_lo, main_hi, args.steps, 0.05, seed=s)
                sa_pat, _ = sa_mod.sa_finetune(
                    sim, target, hard, main_lo=main_lo, main_hi=main_hi,
                    steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
                    flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,
                    reheat_cycles=args.sa_reheat_cycles,
                )
                _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
                seeds_supp.append(sa_s)
            Z[i, j] = np.mean(seeds_supp)
            logger.info(f"[{cnt}/{total}] phi={phi:+.0f}°, n={n}: mean={Z[i, j]:+.2f} dB")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=4, vmax=10)
    plt.colorbar(im, ax=ax, label="suppression mean (dB)")
    ax.set_xticks(range(len(args.element_nums)))
    ax.set_xticklabels([f"{n}×{n}" for n in args.element_nums])
    ax.set_yticks(range(len(args.inc_phis)))
    ax.set_yticklabels([f"{p:+.0f}°" for p in args.inc_phis])
    ax.set_xlabel("element_num"); ax.set_ylabel("inc_phi")
    ax.set_title(f"inc_phi × size at inc_θ={args.inc_theta}°, {args.freq / 1e9:.1f} GHz")
    for i, _ in enumerate(args.inc_phis):
        for j, _ in enumerate(args.element_nums):
            v = Z[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if v < 6 or v > 9 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=110)
    plt.close(fig)

    # Markdown
    lines = [f"# inc_phi × size sweep (inc_θ={args.inc_theta}°, {args.freq / 1e9:.1f} GHz)", "",
             f"GD+SA reheat={args.sa_reheat_cycles}, {args.n_seeds} seeds per cell", ""]
    header = ["inc_phi \\ size"] + [f"{n}×{n}" for n in args.element_nums]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, phi in enumerate(args.inc_phis):
        row = [f"{phi:+.0f}°"] + [f"{Z[i, j]:+.2f}" for j in range(len(args.element_nums))]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## 各 phi 最佳 size")
    for i, phi in enumerate(args.inc_phis):
        best_j = int(np.argmax(Z[i]))
        worst_j = int(np.argmin(Z[i]))
        lines.append(
            f"- phi={phi:+.0f}°: best={args.element_nums[best_j]} ({Z[i, best_j]:+.2f}), "
            f"worst={args.element_nums[worst_j]} ({Z[i, worst_j]:+.2f})"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
