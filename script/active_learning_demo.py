"""
Round 85 — Active Learning Loop Demo

Use dataset_v3 (432 entries) as pool。模擬 BO acquisition (greedy top-K by surrogate
prediction) vs random sampling，看哪個更快找到高 worst_supp configs。

對 patch antenna: "HFSS run" 在這裡用 already-stored labels 替代, 但 logic 一致。
Patch 移植時換成: 跑 HFSS 拿真 response → metrics → 加入 training set。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import RISDataset, SurrogateCNN


def train_surrogate_on_subset(ds, indices, epochs=100, lr=1e-3, device="cuda:0"):
    """Train CNN surrogate on subset of dataset indexed by `indices`."""
    subset = torch.utils.data.Subset(ds, indices)
    torch.set_default_device("cpu")

    class FixedSampler:
        def __init__(self, n, seed=0):
            self.n = n
            self.seed = seed
        def __iter__(self):
            rng = np.random.RandomState(self.seed)
            return iter(rng.permutation(self.n).tolist())
        def __len__(self):
            return self.n

    loader = DataLoader(subset, batch_size=8, sampler=FixedSampler(len(subset)))
    torch.set_default_device(device)

    model = SurrogateCNN(channels=32, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
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


def predict_metrics(model, ds, indices, device, main_lo_per_idx, main_hi_per_idx):
    model.eval()
    pred_worst = []
    with torch.no_grad():
        for i in indices:
            sample = ds[i]
            pat = torch.from_numpy(sample["pattern"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(sample["mask"]).unsqueeze(0).to(device)
            cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
            resp = model(pat, mask, cfg)[0].cpu().numpy()
            ml = main_lo_per_idx[i]
            mh = main_hi_per_idx[i]
            main = resp[ml:mh]
            side = np.delete(resp, np.arange(ml, mh))
            pred_worst.append(float(main.min() - side.max()))
    return np.array(pred_worst)


def get_true_worst(ds, indices, main_lo_per_idx, main_hi_per_idx):
    """Get ground truth worst_supp from stored responses."""
    true_worst = []
    for i in indices:
        sample = ds[i]
        resp = sample["response"]
        ml = main_lo_per_idx[i]
        mh = main_hi_per_idx[i]
        main = resp[ml:mh]
        side = np.delete(resp, np.arange(ml, mh))
        true_worst.append(float(main.min() - side.max()))
    return np.array(true_worst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v3")
    p.add_argument("--initial_size", type=int, default=20)
    p.add_argument("--iter_size", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--epochs_per_iter", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config.device = args.device
    device = args.device

    ds = RISDataset(Path(args.dataset))
    n_total = len(ds)
    print(f"Pool size: {n_total}")

    # Pre-compute main_lo/main_hi per entry
    main_lo_per_idx = {}
    main_hi_per_idx = {}
    for i in range(n_total):
        s = ds.entries[i]
        cfg_d = s["config"]
        # Re-derive main range from config
        sample_per_deg = 2
        center = int(round((cfg_d["target_theta_c"] + 90) * sample_per_deg))
        half = int(round(cfg_d["target_width_deg"] * sample_per_deg / 2))
        main_lo_per_idx[i] = max(0, center - half)
        main_hi_per_idx[i] = min(361, center + half)

    # Get all true worst_supp
    all_indices = list(range(n_total))
    all_true_worst = get_true_worst(ds, all_indices, main_lo_per_idx, main_hi_per_idx)
    true_top_5 = np.sort(all_true_worst)[-5:].mean()
    true_max = all_true_worst.max()
    print(f"Pool stats: min={all_true_worst.min():.2f}, max={all_true_worst.max():.2f}, "
          f"top-5 avg={true_top_5:.2f}")

    # Setup
    rng = np.random.RandomState(args.seed)
    initial_idx = rng.choice(n_total, args.initial_size, replace=False).tolist()

    # === Active learning trajectory ===
    print("\n=== Active Learning (BO Greedy) ===")
    al_labeled = list(initial_idx)
    al_best_per_iter = []
    al_best_per_iter.append(get_true_worst(ds, al_labeled, main_lo_per_idx, main_hi_per_idx).max())
    print(f"Iter 0 (initial): n_labeled={len(al_labeled)}, best={al_best_per_iter[-1]:+.2f}")
    for it in range(args.n_iter):
        # Train surrogate
        model = train_surrogate_on_subset(ds, al_labeled, epochs=args.epochs_per_iter, device=device)
        # Predict on unlabeled
        unlabeled = [i for i in range(n_total) if i not in set(al_labeled)]
        pred = predict_metrics(model, ds, unlabeled, device, main_lo_per_idx, main_hi_per_idx)
        # Pick top-K by predicted worst_supp (greedy, no uncertainty)
        top_k = np.argsort(-pred)[:args.iter_size]
        new_labels = [unlabeled[i] for i in top_k]
        al_labeled.extend(new_labels)
        # Track best so far
        labeled_true = get_true_worst(ds, al_labeled, main_lo_per_idx, main_hi_per_idx)
        al_best_per_iter.append(labeled_true.max())
        # Print status
        new_true = get_true_worst(ds, new_labels, main_lo_per_idx, main_hi_per_idx)
        print(f"Iter {it+1}: n_labeled={len(al_labeled)}, "
              f"new batch avg={new_true.mean():+.2f}, max={new_true.max():+.2f}, "
              f"all-time best={al_best_per_iter[-1]:+.2f}")

    # === Random baseline ===
    print("\n=== Random Sampling Baseline ===")
    rand_labeled = list(initial_idx)
    rand_best_per_iter = []
    rand_best_per_iter.append(get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx).max())
    rng_b = np.random.RandomState(args.seed + 100)
    for it in range(args.n_iter):
        unlabeled = [i for i in range(n_total) if i not in set(rand_labeled)]
        new_labels = rng_b.choice(unlabeled, args.iter_size, replace=False).tolist()
        rand_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx)
        rand_best_per_iter.append(labeled_true.max())
        new_true = get_true_worst(ds, new_labels, main_lo_per_idx, main_hi_per_idx)
        print(f"Iter {it+1}: n_labeled={len(rand_labeled)}, "
              f"new batch avg={new_true.mean():+.2f}, max={new_true.max():+.2f}, "
              f"all-time best={rand_best_per_iter[-1]:+.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = list(range(args.n_iter + 1))
    ax.plot(iters, al_best_per_iter, "o-", label="Active learning (BO greedy)", linewidth=2)
    ax.plot(iters, rand_best_per_iter, "s-", label="Random sampling", linewidth=2)
    ax.axhline(true_max, color="green", linestyle="--", label=f"Pool max: {true_max:.2f}")
    ax.axhline(true_top_5, color="orange", linestyle=":", label=f"True top-5 avg: {true_top_5:.2f}")
    ax.set_xlabel(f"Iteration (each adds {args.iter_size} samples)")
    ax.set_ylabel("Best worst_supp found so far (dB)")
    ax.set_title(f"Active Learning vs Random — Pool {n_total}, Init {args.initial_size}, "
                 f"+{args.iter_size}/iter")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "outputs/r85_active_learning.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {out}")

    # Final comparison
    print(f"\n=== Final Comparison ===")
    print(f"Active learning final best: {al_best_per_iter[-1]:+.2f}")
    print(f"Random sampling final best: {rand_best_per_iter[-1]:+.2f}")
    print(f"True pool max:              {true_max:+.2f}")
    print(f"AL gap to pool max:         {true_max - al_best_per_iter[-1]:+.2f}")
    print(f"Random gap to pool max:     {true_max - rand_best_per_iter[-1]:+.2f}")


if __name__ == "__main__":
    main()
