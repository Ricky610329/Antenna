"""
Round 88 — MC Dropout for Active Learning Uncertainty

R86 same-arch ensemble std 太小 → UCB ≈ greedy.
試 MC Dropout: 訓 single model with dropout, inference 時 30 forward passes
keeping dropout active, mean ± std as predictive uncertainty.

對 patch BO: 比 ensemble 便宜 (1 model vs 5)，且 dropout 強制 model 學 robust。
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
from train_surrogate import RISDataset
from active_learning_demo import get_true_worst


class SurrogateCNNDropout(nn.Module):
    """CNN with dropout for MC dropout uncertainty estimation."""

    def __init__(self, max_n: int = 41, config_dim: int = 6, response_dim: int = 361,
                 channels: int = 32, depth: int = 4, dropout_p: float = 0.3):
        super().__init__()
        c_in = 2 + config_dim
        layers: list[nn.Module] = []
        c = c_in
        for i in range(depth):
            layers += [nn.Conv2d(c, channels, kernel_size=3, padding=1), nn.GELU()]
            c = channels
            layers += [nn.Dropout2d(p=dropout_p)]
            if i < depth - 1:
                layers += [nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2), nn.GELU()]
                layers += [nn.Dropout2d(p=dropout_p)]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, response_dim),
        )

    def forward(self, pattern, mask, config):
        b, h, w = pattern.shape
        cfg_map = config[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([pattern.unsqueeze(1), mask.unsqueeze(1), cfg_map], dim=1)
        return self.head(self.conv(x))


def train_dropout_surrogate(ds, indices, epochs=80, lr=1e-3, dropout_p=0.3, device="cuda:0", seed=0):
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
    model = SurrogateCNNDropout(channels=32, depth=4, dropout_p=dropout_p).to(device)
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


def mc_dropout_predict(model, ds, indices, device, main_lo_per_idx, main_hi_per_idx,
                       n_passes=30):
    """Forward N times with dropout active, return mean and std of worst_supp predictions."""
    model.train()  # keep dropout active
    all_pred = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds_this_pass = []
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
                preds_this_pass.append(float(main.min() - side.max()))
            all_pred.append(np.array(preds_this_pass))
    arr = np.stack(all_pred)
    return arr.mean(axis=0), arr.std(axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v3")
    p.add_argument("--initial_size", type=int, default=20)
    p.add_argument("--iter_size", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=5)
    p.add_argument("--epochs_per_iter", type=int, default=80)
    p.add_argument("--n_mc_passes", type=int, default=20)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config.device = args.device
    device = args.device

    ds = RISDataset(Path(args.dataset))
    n_total = len(ds)
    print(f"Pool size: {n_total}, MC passes={args.n_mc_passes}, dropout={args.dropout_p}, κ={args.kappa}")

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
    all_true = get_true_worst(ds, all_indices, main_lo_per_idx, main_hi_per_idx)
    true_max = all_true.max()
    print(f"Pool max: {true_max:.2f}")

    rng = np.random.RandomState(args.seed)
    initial_idx = rng.choice(n_total, args.initial_size, replace=False).tolist()

    # === MC Dropout UCB ===
    print("\n=== MC Dropout UCB ===")
    mc_labeled = list(initial_idx)
    mc_best_per_iter = [get_true_worst(ds, mc_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    print(f"Iter 0: n={len(mc_labeled)}, best={mc_best_per_iter[-1]:+.2f}")
    avg_stds = []
    for it in range(args.n_iter):
        model = train_dropout_surrogate(
            ds, mc_labeled, epochs=args.epochs_per_iter,
            dropout_p=args.dropout_p, device=device, seed=args.seed + it,
        )
        unlabeled = [i for i in range(n_total) if i not in set(mc_labeled)]
        means, stds = mc_dropout_predict(model, ds, unlabeled, device,
                                          main_lo_per_idx, main_hi_per_idx,
                                          n_passes=args.n_mc_passes)
        avg_stds.append(stds.mean())
        ucb = means + args.kappa * stds
        top_k = np.argsort(-ucb)[:args.iter_size]
        new_labels = [unlabeled[i] for i in top_k]
        mc_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, mc_labeled, main_lo_per_idx, main_hi_per_idx)
        mc_best_per_iter.append(labeled_true.max())
        new_true = get_true_worst(ds, new_labels, main_lo_per_idx, main_hi_per_idx)
        print(f"Iter {it+1}: n={len(mc_labeled)}, "
              f"new avg={new_true.mean():+.2f}, max={new_true.max():+.2f}, "
              f"std avg={stds[top_k].mean():.2f}, "
              f"best={mc_best_per_iter[-1]:+.2f}")

    # === Random baseline ===
    rand_labeled = list(initial_idx)
    rand_best_per_iter = [get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx).max()]
    rng_b = np.random.RandomState(args.seed + 100)
    for it in range(args.n_iter):
        unlabeled = [i for i in range(n_total) if i not in set(rand_labeled)]
        new_labels = rng_b.choice(unlabeled, args.iter_size, replace=False).tolist()
        rand_labeled.extend(new_labels)
        labeled_true = get_true_worst(ds, rand_labeled, main_lo_per_idx, main_hi_per_idx)
        rand_best_per_iter.append(labeled_true.max())

    print(f"\n=== Final (gap to {true_max:+.2f}) ===")
    print(f"  MC Dropout UCB: {mc_best_per_iter[-1]:+.2f}, gap={true_max - mc_best_per_iter[-1]:.2f}")
    print(f"  Random:         {rand_best_per_iter[-1]:+.2f}, gap={true_max - rand_best_per_iter[-1]:.2f}")
    print(f"  R86 ensemble κ=2: +4.42 (gap 1.16) reference")
    print(f"  Avg MC std per iter: {avg_stds}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = list(range(args.n_iter + 1))
    ax.plot(iters, mc_best_per_iter, "o-", label=f"MC Dropout UCB (κ={args.kappa})", linewidth=2.5, color="purple")
    ax.plot(iters, rand_best_per_iter, "s-", label="Random", linewidth=2, color="steelblue")
    ax.axhline(true_max, color="black", linestyle="--", label=f"Pool max: {true_max:.2f}")
    ax.set_xlabel(f"Iteration (each adds {args.iter_size} samples)")
    ax.set_ylabel("Best worst_supp found so far (dB)")
    ax.set_title(f"MC Dropout UCB vs Random — Pool {n_total}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/r88_mc_dropout.png", dpi=110, bbox_inches="tight")
    print("saved: outputs/r88_mc_dropout.png")


if __name__ == "__main__":
    main()
