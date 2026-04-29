"""批次處理多個 RIS target — 一次跑 N 個 design，整合成單一報告。

對使用者真實 use case「同一硬體服務多個方向」非常有用：例如希望生產 5 個
不同方向的 binary pattern，部署時 RIS 切換不同 pattern 對應不同 beam。

兩種輸入方式：

1. **CSV 規格**（推薦）：
   ```
   name,plateau_start,plateau_w,n_restarts
   north,140,46,5
   east,217,46,5
   south,250,40,5
   ```
   `python script/design_batch.py --csv targets.csv`

2. **CLI list**：
   ```
   python script/design_batch.py \
     --target north:140:46 \
     --target east:217:46 \
     --target south:250:40 \
     --n_restarts 5
   ```

輸出：
    outputs/design_batch_<timestamp>/
        <name>/                       — 每個 target 一個 dir（pattern + design.png）
        summary.md                    — 整合表，含 suppression 對照
        suppression_chart.png         — 直條圖
"""

import argparse
import csv
from datetime import datetime
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
    sa_mod=None, sa_steps: int = 0, sa_T0: float = 20.0,
    sa_flip_n: int = 3, sa_reheat_cycles: int = 2,
) -> dict:
    """每個 GD restart 立刻做 SA，取 best across（修 round 35 SA-per-restart bug）。"""
    pattern_size = element_num * element_num
    best_supp = -np.inf
    best_pattern = None
    best_resp = None
    best_loss_overall = float("inf")
    for restart in range(n_restarts):
        torch.manual_seed(seed + restart)
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
            side_idx = np.array([i for i in range(len(hard_resp)) if i not in set(main_idx.tolist())])
            mp = float(hard_resp[main_idx].max())
            sm = float(hard_resp[side_idx].max())
            supp = mp - sm

        # Round 36 修正：對每個 restart 立刻做 SA，不是只對 best GD
        if sa_mod is not None and sa_steps > 0:
            sa_pat, _ = sa_mod.sa_finetune(
                sim, target, hard,
                main_lo=main_lo, main_hi=main_hi,
                steps=sa_steps, T0=sa_T0, T_final=0.001,
                flip_n=sa_flip_n, log_every=sa_steps + 1,
                reheat_cycles=sa_reheat_cycles,
            )
            _, sa_supp = sa_mod.evaluate(sim, sa_pat, target, main_lo, main_hi)
            if sa_supp > supp:
                with torch.no_grad():
                    hard_resp = sim(sa_pat)["response"].cpu().numpy()
                hard = sa_pat
                supp = sa_supp
                mp = float(hard_resp[main_idx].max())
                sm_idx = np.array([i for i in range(len(hard_resp)) if i not in set(main_idx.tolist())])
                sm = float(hard_resp[sm_idx].max())

        if supp > best_supp:
            best_supp = supp
            best_pattern = hard.cpu().numpy() if isinstance(hard, torch.Tensor) else hard
            best_resp = hard_resp
            best_loss_overall = best_loss_local

    return {
        "suppression": best_supp,
        "pattern": best_pattern,
        "hard_resp": best_resp,
        "best_loss": best_loss_overall,
        "main_peak": float(best_resp[np.arange(main_lo, min(main_hi, len(best_resp)))].max()),
        "side_max": float(np.delete(best_resp, np.arange(main_lo, min(main_hi, len(best_resp)))).max()),
        "on_rate": float(best_pattern.mean()),
    }


def parse_target_str(s: str) -> tuple[str, int, int]:
    """parse 'name:start:width' → (name, start, width)"""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"target spec '{s}' 必須是 'name:start:width' 格式")
    return parts[0], int(parts[1]), int(parts[2])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element_num", type=int, default=15)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_restarts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inc_theta", type=float, default=None,
                   help="入射角 θ_i (deg)。round 12 sweep 顯示 ±60° 最佳，0° 最差")
    p.add_argument("--inc_phi", type=float, default=None)
    p.add_argument("--freq", type=float, default=None,
                   help="工作頻率 (Hz)。None=default 28e9")
    p.add_argument("--sa_steps", type=int, default=0,
                   help="GD 後加 SA fine-tune 步數（0=不做）。建議 8000 + flip_n=3 + T0=20")
    p.add_argument("--sa_T0", type=float, default=20.0)
    p.add_argument("--sa_flip_n", type=int, default=3)
    p.add_argument("--sa_reheat_cycles", type=int, default=2)
    p.add_argument("--csv", type=str, default=None,
                   help="CSV file with columns: name,plateau_start,plateau_w[,n_restarts]")
    p.add_argument("--target", action="append", default=[],
                   help="target spec 'name:start:width', 可多次")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")

    targets = []
    if args.csv:
        with open(args.csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                targets.append({
                    "name": row["name"],
                    "start": int(row["plateau_start"]),
                    "width": int(row["plateau_w"]),
                    "n_restarts": int(row.get("n_restarts", args.n_restarts)),
                })
    for s in args.target:
        name, start, width = parse_target_str(s)
        targets.append({"name": name, "start": start, "width": width, "n_restarts": args.n_restarts})

    if not targets:
        logger.error("未指定任何 target；請用 --csv 或 --target")
        return

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("outputs/design_batch") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"輸出 dir: {out_dir}")

    sim_kwargs = {"element_num": args.element_num}
    if args.inc_theta is not None:
        sim_kwargs["inc_theta_deg"] = args.inc_theta
    if args.inc_phi is not None:
        sim_kwargs["inc_phi_deg"] = args.inc_phi
    if args.freq is not None:
        sim_kwargs["freq_hz"] = args.freq
    sim = RISSimulator(**sim_kwargs)
    logger.info(f"RISSimulator: element_num={sim.element_num}, freq={sim.freq_hz / 1e9:.2f} GHz, "
                f"inc_θ={sim.inc_theta_deg}°, inc_φ={sim.inc_phi_deg}°")

    # 可選 SA fine-tune 模組（lazy import）
    sa_mod = None
    if args.sa_steps > 0:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_sa", Path(__file__).parent / "binary_sa_finetune.py")
        sa_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(sa_mod)
        logger.info(f"SA fine-tune enabled: {args.sa_steps} steps, T0={args.sa_T0}, flip_n={args.sa_flip_n}")
    THETA_DEG = np.arange(-90, 90.1, 0.5)

    results = []
    for i, t in enumerate(targets, start=1):
        name = t["name"]
        start = t["start"]
        width = t["width"]
        nr = t["n_restarts"]
        logger.info(
            f"[{i}/{len(targets)}] target={name} plateau idx {start}-{start + width} "
            f"({width} samples), n_restarts={nr}"
        )
        target_np = np.full(361, -20.0, dtype=np.float32)
        target_np[start:start + width] = 0.0
        target = torch.tensor(target_np, device=config.device)
        # SA-per-restart: design_one 內部處理（round 36 修正）
        info = design_one(
            sim, target, element_num=args.element_num,
            main_lo=start, main_hi=start + width,
            steps=args.steps, lr=args.lr, n_restarts=nr, seed=args.seed,
            sa_mod=sa_mod, sa_steps=args.sa_steps,
            sa_T0=args.sa_T0, sa_flip_n=args.sa_flip_n,
            sa_reheat_cycles=args.sa_reheat_cycles,
        )
        info["name"] = name
        info["plateau_start"] = start
        info["plateau_w"] = width
        results.append(info)

        # 個別 target dir
        sub = out_dir / name
        sub.mkdir(exist_ok=True)
        np.save(sub / "pattern_binary.npy", info["pattern"].astype(np.uint8))
        # 三聯圖
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].imshow(info["pattern"], cmap="gray_r", vmin=0, vmax=1)
        on = int(info["pattern"].sum())
        axes[0].set_title(f"{name} pattern\n{on}/{args.element_num**2} on ({info['on_rate']:.0%})")
        axes[0].axis("off")
        axes[1].plot(THETA_DEG, info["hard_resp"], label="actual", linewidth=1.5)
        axes[1].plot(THETA_DEG, target_np, label="target", linewidth=1.5, alpha=0.6)
        axes[1].axvspan(-90 + start * 0.5, -90 + (start + width) * 0.5, color="green", alpha=0.1)
        axes[1].set_title(f"{name}: suppression={info['suppression']:+.2f} dB")
        axes[1].set_xlabel("theta (deg)"); axes[1].set_ylabel("dB")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(sub / "design.png", dpi=110)
        plt.close(fig)
        logger.info(f"  → {info['suppression']:+.2f} dB, on={info['on_rate']:.0%}")

    # 整合 summary
    lines = ["# Batch design 結果", "",
             f"Setting: {args.element_num}×{args.element_num} RIS, "
             f"{args.steps} steps × n_restarts (各別)",
             "",
             "| name | plateau | θ_center | on% | main_peak | side_max | suppression |",
             "|------|---------|----------|-----|-----------|----------|-------------|"]
    for r in results:
        st, w = r["plateau_start"], r["plateau_w"]
        theta_c = -90 + (st + w / 2) * 0.5
        lines.append(
            f"| {r['name']} | {st}-{st + w} | {theta_c:+.1f}° | "
            f"{r['on_rate']:.0%} | {r['main_peak']:+.2f} | {r['side_max']:+.2f} | "
            f"{r['suppression']:+.2f} |"
        )
    sup_arr = np.array([r["suppression"] for r in results])
    lines += [
        "",
        f"**suppression**: mean={sup_arr.mean():+.2f}, "
        f"min={sup_arr.min():+.2f}, max={sup_arr.max():+.2f}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 直條圖
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 4))
    names = [r["name"] for r in results]
    sups = [r["suppression"] for r in results]
    bars = ax.bar(names, sups, color=["green" if s > 0 else "tomato" for s in sups])
    ax.axhline(0, color="gray", alpha=0.5)
    ax.set_ylabel("suppression (dB)")
    ax.set_title(f"Batch design suppression ({len(results)} targets)")
    for bar, s in zip(bars, sups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{s:+.2f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "suppression_chart.png", dpi=110)
    plt.close(fig)

    logger.success(
        f"完成 {len(results)} targets → {out_dir}/  "
        f"(suppression mean={sup_arr.mean():+.2f}, max={sup_arr.max():+.2f})"
    )


if __name__ == "__main__":
    main()
