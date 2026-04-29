"""對比 SA single-cycle vs reheat schedule — 量化 reheat 改善 mean。

從 round 16 知道單輪 SA 7/10 達 +7 dB。Reheat 多輪降溫應能 escape
deeper local minima，把 mean 從 +7.65 拉高。

實驗：
- 同一 init pattern（GD 卡 +2.42 dB local min）
- 跑 single-cycle SA 5 seeds vs reheat-4-cycle SA 5 seeds
- 同樣總 steps（8000 vs 8000），看 final suppression 差異
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--sa_steps", type=int, default=8000)
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--reheat_cycles_list", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/benchmark_sa_reheat")
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
    sim = RISSimulator(element_num=args.element_num, inc_theta_deg=args.inc_theta)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 對每個 seed 先跑 GD 拿初始，再用各 reheat_cycles 跑 SA
    results = {rc: [] for rc in args.reheat_cycles_list}
    gd_supps = []
    for s in range(args.n_seeds):
        hard, gd_s = gd_one(sim, target, args.element_num, main_lo, main_hi, args.steps, 0.05, seed=s)
        gd_supps.append(gd_s)
        for rc in args.reheat_cycles_list:
            sa_pat, _ = sa_mod.sa_finetune(
                sim, target, hard, main_lo=main_lo, main_hi=main_hi,
                steps=args.sa_steps, T0=args.sa_T0, T_final=0.001,
                flip_n=args.sa_flip_n, log_every=args.sa_steps + 1,
                reheat_cycles=rc,
            )
            _, sa_s = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
            results[rc].append(sa_s)
        logger.info(f"seed {s}: GD={gd_s:+.2f}, " +
                    ", ".join([f"reheat={rc}: {results[rc][-1]:+.2f}" for rc in args.reheat_cycles_list]))

    # 統計
    lines = ["# SA reheat schedule benchmark", "",
             f"Setting: {args.element_num}×{args.element_num}, inc_θ={args.inc_theta}°, "
             f"plateau {main_lo}-{main_hi}, total {args.sa_steps} SA steps",
             ""]
    lines += ["| seed | GD | " + " | ".join([f"reheat={rc}" for rc in args.reheat_cycles_list]) + " |",
              "|------|-----|" + "|".join(["-----"] * len(args.reheat_cycles_list)) + "|"]
    for s in range(args.n_seeds):
        row = [f"{s}", f"{gd_supps[s]:+.2f}"] + [f"{results[rc][s]:+.2f}" for rc in args.reheat_cycles_list]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 統計"]
    lines.append(f"- GD-only: mean={np.mean(gd_supps):+.2f}, std={np.std(gd_supps):.2f}")
    for rc in args.reheat_cycles_list:
        arr = np.array(results[rc])
        lines.append(f"- reheat={rc}: mean={arr.mean():+.2f}, std={arr.std():.2f}, "
                     f"min={arr.min():+.2f}, max={arr.max():+.2f}")
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
