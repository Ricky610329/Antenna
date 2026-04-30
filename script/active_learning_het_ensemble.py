"""
Round 89 — Heterogeneous Architecture Ensemble for UCB AL

R86 same-arch ensemble std 太小 (0.05-0.14). 試 different architectures:
- CNN c=16 d=3 (small)
- CNN c=32 d=4 (medium, R72 default)
- CNN c=64 d=5 (large)

不同 capacity 會學到不同 features → 更大 ensemble variance → useful UCB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import RISDataset, SurrogateCNN
from active_learning_demo import predict_metrics, get_true_worst


def train_with_arch(ds, indices, channels, depth, epochs=80, lr=1e-3, device="cuda:0", seed=0):
    subset = torch.utils.data.Subset(ds, indices)
    torch.set_default_device("cpu")

    class FixedSampler:
        def __init__(self, n, seed=0):
            self.n = n
            self.seed = seed
        def __iter__(self):
            return iter(np.random.RandomState(self.seed).permutation(self.n).tolist())
        def __len__(self):
            return self.n

    loader = DataLoader(subset, batch_size=8, sampler=FixedSampler(len(subset), seed))
    torch.set_default_device(device)

    torch.manual_seed(seed)
    model = SurrogateCNN(channels=channels, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            pat = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            cfg = batch["config"].to(device)
            tgt = batch["response"].to(device)
            pred = model(pat, mask, cfg)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_het_ensemble(ensemble, ds, indices, device, main_lo_per_idx, main_hi_per_idx):
    all_pred = []
    for model in ensemble:
        all_pred.append(predict_metrics(model, ds, indices, device, main_lo_per_idx, main_hi_per_idx))
    arr = np.stack(all_pred)
    return arr.mean(axis=0), arr.std(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v3")
    p.add_argument("--initial_size", type=int, default=20)
    p.add_argument("--iter_size", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--epochs_per_iter", type=int, default=80)
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config.device = args.device
    device = args.device
    ds = RISDataset(Path(args.dataset))
    n_total = len(ds)

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

    all_true = get_true_worst(ds, list(range(n_total)), main_lo_per_idx, main_hi_per_idx)
    true_max = all_true.max()
    print(f"Pool {n_total}, max {true_max:.2f}")

    rng = np.random.RandomState(args.seed)
    initial_idx = rng.choice(n_total, args.initial_size, replace=False).tolist()

    archs = [(16, 3), (32, 4), (64, 5)]  # heterogeneous
    print(f"\n=== Het Ensemble UCB (archs: {archs}, κ={args.kappa}) ===")

    labeled = list(initial_idx)
    best_per_iter = [get_true_worst(ds, labeled, main_lo_per_idx, main_hi_per_idx).max()]
    print(f"Iter 0: best={best_per_iter[-1]:+.2f}")
    avg_stds = []
    for it in range(args.n_iter):
        ensemble = []
        for k, (ch, dp) in enumerate(archs):
            model = train_with_arch(
                ds, labeled, channels=ch, depth=dp, epochs=args.epochs_per_iter,
                device=device, seed=args.seed + k + it * 10,
            )
            ensemble.append(model)
        unlabeled = [i for i in range(n_total) if i not in set(labeled)]
        means, stds = predict_het_ensemble(ensemble, ds, unlabeled, device,
                                            main_lo_per_idx, main_hi_per_idx)
        avg_stds.append(stds.mean())
        ucb = means + args.kappa * stds
        top_k = np.argsort(-ucb)[:args.iter_size]
        new_labels = [unlabeled[i] for i in top_k]
        labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, labeled, main_lo_per_idx, main_hi_per_idx)
        best_per_iter.append(labeled_true.max())
        new_true = get_true_worst(ds, new_labels, main_lo_per_idx, main_hi_per_idx)
        print(f"Iter {it+1}: n={len(labeled)}, "
              f"new max={new_true.max():+.2f}, std avg={stds[top_k].mean():.2f}, "
              f"best={best_per_iter[-1]:+.2f}")

    # Random baseline
    rand_labeled = list(initial_idx)
    rand_best = [get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    rng_b = np.random.RandomState(args.seed + 100)
    for it in range(args.n_iter):
        unlabeled = [i for i in range(n_total) if i not in set(rand_labeled)]
        new_labels = rng_b.choice(unlabeled, args.iter_size, replace=False).tolist()
        rand_labeled.extend(new_labels)
        rand_best.append(get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx).max())

    print(f"\n=== Final ===")
    print(f"  Het Ensemble UCB: {best_per_iter[-1]:+.2f}")
    print(f"  Random:           {rand_best[-1]:+.2f}")
    print(f"  R86 same ensemble: +4.42")
    print(f"  R88 MC Dropout:    +4.79")
    print(f"  Avg std per iter: {[f'{s:.2f}' for s in avg_stds]}")


if __name__ == "__main__":
    main()
