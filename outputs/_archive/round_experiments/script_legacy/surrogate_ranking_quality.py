"""
Round 81 — Surrogate as ranking filter

R77-R80 證明 surrogate gradient 不可信 (cosine 0.001)。
但如果 surrogate 預測 function value 的 RANKING 跟 real sim 一致,
仍可用於 active learning / BO acquisition / pre-filter.

對 patch 移植: 即使 GD-through-surrogate 不行, 用 surrogate 篩選候選 geometry
後跑 HFSS verify 仍是 dataset-efficient strategy.

實驗:
  - 對 dataset_v2 全 108 entries:
    - True worst_supp (from JSONL, 已 stored)
    - Predicted worst_supp (surrogate forward + 後處理)
  - Compute Pearson + Spearman correlation
  - Plot scatter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import SurrogateCNN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate_pt", type=str, default="outputs/r72_cnn_v2/surrogate.pt")
    p.add_argument("--dataset", type=str, default="outputs/dataset_v2")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device
    device = args.device

    surrogate = SurrogateCNN(channels=32, depth=4).to(device)
    surrogate.load_state_dict(torch.load(args.surrogate_pt, map_location=device))
    surrogate.eval()

    root = Path(args.dataset)
    entries = []
    with open(root / "entries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    rows = []
    for e in entries:
        cfg_d = e["config"]
        main_lo, main_hi = e["main_idx_range"]
        for p_entry in e["pareto"]:
            pat = np.load(root / p_entry["pattern_file"]).astype(np.float32)
            n = pat.shape[0]
            max_n = 41
            offset = (max_n - n) // 2
            padded = np.zeros((max_n, max_n), dtype=np.float32)
            padded[offset:offset+n, offset:offset+n] = pat
            mask = np.zeros((max_n, max_n), dtype=np.float32)
            mask[offset:offset+n, offset:offset+n] = 1.0

            cfg_vec = np.array([
                cfg_d["freq_ghz"] / 100.0,
                cfg_d["n"] / 41.0,
                cfg_d["target_theta_c"] / 90.0,
                cfg_d["target_width_deg"] / 90.0,
                cfg_d["inc"] / 90.0,
                p_entry["ripple_weight"] / 5.0,
            ], dtype=np.float32)

            with torch.no_grad():
                pat_t = torch.from_numpy(padded).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                cfg_t = torch.from_numpy(cfg_vec).unsqueeze(0).to(device)
                resp_pred = surrogate(pat_t, mask_t, cfg_t)[0].cpu().numpy()

            # Compute predicted metrics
            main_pred = resp_pred[main_lo:main_hi]
            side_pred = np.delete(resp_pred, np.arange(main_lo, main_hi))
            worst_pred = float(main_pred.min() - side_pred.max())
            headline_pred = float(main_pred.max() - side_pred.max())

            true_metrics = p_entry["metrics"]
            rows.append({
                "freq_ghz": cfg_d["freq_ghz"],
                "n": cfg_d["n"],
                "theta_c": cfg_d["target_theta_c"],
                "width": cfg_d["target_width_deg"],
                "rw": p_entry["ripple_weight"],
                "true_worst": true_metrics["worst_supp"],
                "pred_worst": worst_pred,
                "true_headline": true_metrics["headline_supp"],
                "pred_headline": headline_pred,
            })

    print(f"Total entries: {len(rows)}")

    # Correlation analysis
    true_worst = np.array([r["true_worst"] for r in rows])
    pred_worst = np.array([r["pred_worst"] for r in rows])
    true_headline = np.array([r["true_headline"] for r in rows])
    pred_headline = np.array([r["pred_headline"] for r in rows])

    print("\n=== worst_supp correlation ===")
    pear_w, _ = pearsonr(true_worst, pred_worst)
    spear_w, _ = spearmanr(true_worst, pred_worst)
    print(f"  Pearson:  {pear_w:+.3f}")
    print(f"  Spearman: {spear_w:+.3f}")

    print("\n=== headline_supp correlation ===")
    pear_h, _ = pearsonr(true_headline, pred_headline)
    spear_h, _ = spearmanr(true_headline, pred_headline)
    print(f"  Pearson:  {pear_h:+.3f}")
    print(f"  Spearman: {spear_h:+.3f}")

    # Per-rw analysis
    print("\n=== worst_supp correlation per ripple_weight ===")
    for rw_val in sorted({r["rw"] for r in rows}):
        mask_rw = np.array([r["rw"] == rw_val for r in rows])
        if mask_rw.sum() < 3:
            continue
        tw = true_worst[mask_rw]
        pw = pred_worst[mask_rw]
        try:
            pe, _ = pearsonr(tw, pw)
            sp, _ = spearmanr(tw, pw)
        except Exception:
            pe = sp = float("nan")
        print(f"  rw={rw_val}: n={mask_rw.sum()}, Pearson={pe:+.3f}, Spearman={sp:+.3f}")

    # Top-K test: 如果用 surrogate 選 top-K, 真實 top-K 命中率多少？
    print("\n=== Top-K filter quality ===")
    print(f"{'K':>4} | {'pred top-K avg true worst':>25} | {'random K avg true worst':>22} | {'true top-K avg':>15}")
    print("-" * 80)
    for K in [5, 10, 20, 30, 50]:
        if K > len(rows):
            continue
        sorted_by_pred = np.argsort(-pred_worst)  # descending
        top_pred = sorted_by_pred[:K]
        top_pred_true_worst = true_worst[top_pred].mean()
        # Random K
        rng = np.random.RandomState(0)
        random_K = rng.choice(len(rows), K, replace=False)
        random_K_avg = true_worst[random_K].mean()
        # True top-K
        sorted_by_true = np.argsort(-true_worst)
        true_top_avg = true_worst[sorted_by_true[:K]].mean()
        print(f"{K:>4} | {top_pred_true_worst:>+25.2f} | {random_K_avg:>+22.2f} | "
              f"{true_top_avg:>+15.2f}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(true_worst, pred_worst, alpha=0.5)
    axes[0].plot([true_worst.min(), true_worst.max()],
                  [true_worst.min(), true_worst.max()], "k--", linewidth=1)
    axes[0].set_xlabel("True worst_supp (dB)")
    axes[0].set_ylabel("Predicted worst_supp (dB)")
    axes[0].set_title(f"worst_supp: Pearson {pear_w:+.3f}, Spearman {spear_w:+.3f}")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(true_headline, pred_headline, alpha=0.5, color="orange")
    axes[1].plot([true_headline.min(), true_headline.max()],
                  [true_headline.min(), true_headline.max()], "k--", linewidth=1)
    axes[1].set_xlabel("True headline_supp (dB)")
    axes[1].set_ylabel("Predicted headline_supp (dB)")
    axes[1].set_title(f"headline_supp: Pearson {pear_h:+.3f}, Spearman {spear_h:+.3f}")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Surrogate prediction vs true (R72 CNN trained on dataset_v2)")
    fig.tight_layout()
    fig.savefig("outputs/r81_surrogate_ranking.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved: outputs/r81_surrogate_ranking.png")


if __name__ == "__main__":
    main()
