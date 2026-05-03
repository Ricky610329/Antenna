"""測試 CUDA determinism 設定是否讓 GD 結果完全可重現。

Round 30 觀察：torch.manual_seed(0) 在 design tool vs benchmark 給不同結果。
推測 GPU CUDA non-determinism 是元兇。

實驗：跑兩次同 seed GD，看 suppression 是否 byte-identical：
- 模式 A: 預設（non-deterministic）
- 模式 B: torch.use_deterministic_algorithms(True) + cudnn.deterministic
- 模式 C: 模式 B + CUBLAS_WORKSPACE_CONFIG=:4096:8

用法：
    python script/test_determinism.py --device cuda:0
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def gd_one(sim, target, n, main_lo, main_hi, steps, lr, seed):
    pattern_size = n * n
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        resp = sim(hard)["response"].cpu().numpy()
        main_idx = np.arange(main_lo, main_hi)
        side_idx = np.array([i for i in range(len(resp)) if i not in set(main_idx.tolist())])
        mp = float(resp[main_idx].max())
        sm = float(resp[side_idx].max())
    return hard.cpu().numpy(), mp - sm, best_loss


def test_mode(name, target, sim, n, main_lo, main_hi, n_runs=3, **kwargs):
    """跑 N 次同 seed，回傳 (suppressions, patterns)"""
    logger.info(f"=== Mode: {name} ===")
    sups = []
    losses = []
    for run in range(n_runs):
        pat, supp, loss = gd_one(sim, target, n, main_lo, main_hi, kwargs["steps"], kwargs["lr"], seed=0)
        sups.append(supp)
        losses.append(loss)
        logger.info(f"  run {run + 1}: suppression={supp:+.4f} dB, loss={loss:.4f}")
    return sups, losses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element_num", type=int, default=19)
    p.add_argument("--inc_theta", type=float, default=60.0)
    p.add_argument("--freq", type=float, default=5.6e9)
    p.add_argument("--plateau_start", type=int, default=154)
    p.add_argument("--plateau_w", type=int, default=46)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/test_determinism")
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

    common_kwargs = {"steps": args.steps, "lr": args.lr}

    # Mode A: 預設
    sups_a, losses_a = test_mode("A: 預設 (non-det)", target, sim,
                                  args.element_num, main_lo, main_hi,
                                  n_runs=args.n_runs, **common_kwargs)

    # Mode B: cudnn.deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sups_b, losses_b = test_mode("B: cudnn.deterministic=True", target, sim,
                                  args.element_num, main_lo, main_hi,
                                  n_runs=args.n_runs, **common_kwargs)

    # Mode C: full deterministic
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        logger.warning(f"use_deterministic_algorithms failed: {e}")
    sups_c, losses_c = test_mode("C: use_deterministic_algorithms(True)", target, sim,
                                  args.element_num, main_lo, main_hi,
                                  n_runs=args.n_runs, **common_kwargs)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # Summary
    def summary(label, sups, losses):
        s_arr = np.array(sups)
        l_arr = np.array(losses)
        return (f"- **{label}**: sups range {s_arr.min():.4f}~{s_arr.max():.4f}, "
                f"std={s_arr.std():.4f}; losses range {l_arr.min():.4f}~{l_arr.max():.4f}, "
                f"std={l_arr.std():.4f}, "
                f"{'**reproducible ✓**' if l_arr.std() < 1e-6 else '✗ non-det'}")

    lines = ["# CUDA Determinism Test", "",
             f"Setting: {args.element_num}×{args.element_num}, freq={args.freq / 1e9} GHz, "
             f"inc_θ={args.inc_theta}°, GD seed=0, {args.n_runs} runs each", "",
             summary("Mode A 預設", sups_a, losses_a),
             summary("Mode B cudnn.deterministic", sups_b, losses_b),
             summary("Mode C full deterministic", sups_c, losses_c),
             "",
             "## Per-run suppression"]
    for label, sups in [("A", sups_a), ("B", sups_b), ("C", sups_c)]:
        lines.append(f"- {label}: " + ", ".join([f"{s:+.4f}" for s in sups]))

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
