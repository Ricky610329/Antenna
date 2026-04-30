"""
Round 86 — Active Learning with UCB Ensemble

R85 greedy fail (worse than random). 這版用 ensemble of 3 surrogates +
UCB acquisition 試 fix.

acquisition = ensemble_mean + κ × ensemble_std

對 patch BO: 同邏輯換 RISSimulator → HFSS。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import RISDataset, SurrogateCNN
from active_learning_demo import (
    train_surrogate_on_subset, predict_metrics, get_true_worst,
)


def predict_ensemble(ensemble, ds, indices, device, main_lo_per_idx, main_hi_per_idx):
    """Returns (mean_worst, std_worst) for each index."""
    all_preds = []  # [n_models, n_samples]
    for model in ensemble:
        preds = predict_metrics(model, ds, indices, device, main_lo_per_idx, main_hi_per_idx)
        all_preds.append(preds)
    arr = np.stack(all_preds)  # (n_models, n)
    return arr.mean(axis=0), arr.std(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v3")
    p.add_argument("--initial_size", type=int, default=20)
    p.add_argument("--iter_size", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--epochs_per_iter", type=int, default=80)
    p.add_argument("--n_ensemble", type=int, default=3)
    p.add_argument("--kappa", type=float, default=2.0,
                   help="UCB exploration weight: ucb = mean + κ*std")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config.device = args.device
    device = args.device

    ds = RISDataset(Path(args.dataset))
    n_total = len(ds)
    print(f"Pool size: {n_total}, ensemble n={args.n_ensemble}, κ={args.kappa}")

    main_lo_per_idx = {}
    main_hi_per_idx = {}
    for i in range(n_total):
        s = ds.entries[i]
        cfg_d = s["config"]
        sample_per_deg = 2
        center = int(round((cfg_d["target_theta_c"] + 90) * sample_per_deg))
        half = int(round(cfg_d["target_width_deg"] * sample_per_deg / 2))
        main_lo_per_idx[i] = max(0, center - half)
        main_hi_per_idx[i] = min(361, center + half)

    all_indices = list(range(n_total))
    all_true_worst = get_true_worst(ds, all_indices, main_lo_per_idx, main_hi_per_idx)
    true_max = all_true_worst.max()
    print(f"Pool max: {true_max:.2f}")

    rng = np.random.RandomState(args.seed)
    initial_idx = rng.choice(n_total, args.initial_size, replace=False).tolist()

    # === UCB Active Learning ===
    print("\n=== UCB Active Learning (ensemble) ===")
    ucb_labeled = list(initial_idx)
    ucb_best_per_iter = [get_true_worst(ds, ucb_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    print(f"Iter 0: n_labeled={len(ucb_labeled)}, best={ucb_best_per_iter[-1]:+.2f}")
    for it in range(args.n_iter):
        # Train ensemble
        ensemble = []
        for k in range(args.n_ensemble):
            torch.manual_seed(args.seed + k * 100 + it)
            model = train_surrogate_on_subset(
                ds, ucb_labeled, epochs=args.epochs_per_iter, device=device,
            )
            ensemble.append(model)
        # Predict ensemble on unlabeled
        unlabeled = [i for i in range(n_total) if i not in set(ucb_labeled)]
        means, stds = predict_ensemble(ensemble, ds, unlabeled, device,
                                        main_lo_per_idx, main_hi_per_idx)
        # UCB acquisition
        ucb = means + args.kappa * stds
        top_k = np.argsort(-ucb)[:args.iter_size]
        new_labels = [unlabeled[i] for i in top_k]
        ucb_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, ucb_labeled, main_lo_per_idx, main_hi_per_idx)
        ucb_best_per_iter.append(labeled_true.max())
        new_true = get_true_worst(ds, new_labels, main_lo_per_idx, main_hi_per_idx)
        print(f"Iter {it+1}: n={len(ucb_labeled)}, "
              f"new avg={new_true.mean():+.2f}, max={new_true.max():+.2f}, "
              f"std avg={stds[top_k].mean():.2f}, "
              f"best={ucb_best_per_iter[-1]:+.2f}")

    # === Greedy (R85 baseline) ===
    print("\n=== Greedy (single surrogate, R85 baseline) ===")
    greedy_labeled = list(initial_idx)
    greedy_best_per_iter = [get_true_worst(ds, greedy_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    for it in range(args.n_iter):
        torch.manual_seed(args.seed + it)
        model = train_surrogate_on_subset(
            ds, greedy_labeled, epochs=args.epochs_per_iter, device=device,
        )
        unlabeled = [i for i in range(n_total) if i not in set(greedy_labeled)]
        pred = predict_metrics(model, ds, unlabeled, device, main_lo_per_idx, main_hi_per_idx)
        top_k = np.argsort(-pred)[:args.iter_size]
        new_labels = [unlabeled[i] for i in top_k]
        greedy_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, greedy_labeled, main_lo_per_idx, main_hi_per_idx)
        greedy_best_per_iter.append(labeled_true.max())

    # === Random baseline ===
    print("\n=== Random Sampling ===")
    rand_labeled = list(initial_idx)
    rand_best_per_iter = [get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    rng_b = np.random.RandomState(args.seed + 100)
    for it in range(args.n_iter):
        unlabeled = [i for i in range(n_total) if i not in set(rand_labeled)]
        new_labels = rng_b.choice(unlabeled, args.iter_size, replace=False).tolist()
        rand_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx)
        rand_best_per_iter.append(labeled_true.max())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = list(range(args.n_iter + 1))
    ax.plot(iters, ucb_best_per_iter, "o-", label=f"UCB ensemble (κ={args.kappa})", linewidth=2.5, color="darkgreen")
    ax.plot(iters, greedy_best_per_iter, "s-", label="Greedy single (R85)", linewidth=2, color="darkred")
    ax.plot(iters, rand_best_per_iter, "^-", label="Random", linewidth=2, color="steelblue")
    ax.axhline(true_max, color="black", linestyle="--", label=f"Pool max: {true_max:.2f}")
    ax.set_xlabel(f"Iteration (each adds {args.iter_size} samples)")
    ax.set_ylabel("Best worst_supp found so far (dB)")
    ax.set_title(f"Active Learning Comparison — Pool {n_total}, "
                 f"Init {args.initial_size}, +{args.iter_size}/iter")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "outputs/r86_ucb_vs_greedy.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {out}")

    print(f"\n=== Final Comparison (gap to pool max {true_max:+.2f}) ===")
    print(f"  UCB ensemble:  best={ucb_best_per_iter[-1]:+.2f}, gap={true_max - ucb_best_per_iter[-1]:.2f}")
    print(f"  Greedy single: best={greedy_best_per_iter[-1]:+.2f}, gap={true_max - greedy_best_per_iter[-1]:.2f}")
    print(f"  Random:        best={rand_best_per_iter[-1]:+.2f}, gap={true_max - rand_best_per_iter[-1]:.2f}")


if __name__ == "__main__":
    main()
