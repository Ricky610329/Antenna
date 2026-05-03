"""
Round 90 — Het Ensemble Gradient Quality

R79 single CNN gradient cosine = 0.001 (random).
Test if R89 het ensemble (different architectures) gives better gradient signal.
- Forward: each model predicts response
- Gradient: average gradients across ensemble (or single mean prediction's gradient)

如果 cos > 0.5, GD-through-ensemble 變 viable, 大幅 widen patch BO toolkit.
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

from antenna.ris import RISSimulator
from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import RISDataset, SurrogateCNN
from active_learning_het_ensemble import train_with_arch


def soft_max(x, beta=20.0):
    return (1/beta) * torch.logsumexp(beta * x, dim=-1)


def soft_min(x, beta=20.0):
    return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0):
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    main_min = soft_min(main, beta)
    side_max = soft_max(side, beta)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max(main, beta)
        loss = loss + ripple_weight * (main_max - main_min)
    return loss


def gradient_through_real_sim(sim, soft_bin, main_lo, main_hi, rw):
    soft_bin = soft_bin.detach().clone().requires_grad_(True)
    resp = sim(soft_bin)["response"]
    loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
    grad = torch.autograd.grad(loss, soft_bin)[0]
    return grad.detach()


def gradient_through_ensemble(ensemble, soft_bin, max_n, mask, cfg_vec, main_lo, main_hi, rw, mode="mean"):
    """mode: 'mean' = mean of predictions then loss; 'avg_grads' = avg gradient across models."""
    n = soft_bin.shape[0]
    offset = (max_n - n) // 2
    if mode == "mean":
        # Compute mean prediction, then loss, then gradient
        soft_bin = soft_bin.detach().clone().requires_grad_(True)
        full = torch.zeros(max_n, max_n, device=soft_bin.device)
        full[offset:offset+n, offset:offset+n] = soft_bin
        pat_padded = full.unsqueeze(0)
        preds = []
        for model in ensemble:
            preds.append(model(pat_padded, mask.unsqueeze(0), cfg_vec)[0])
        mean_resp = torch.stack(preds).mean(0)
        loss = worst_case_loss(mean_resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
        grad = torch.autograd.grad(loss, soft_bin)[0]
        return grad.detach()
    else:  # avg_grads
        all_grads = []
        for model in ensemble:
            sb = soft_bin.detach().clone().requires_grad_(True)
            full = torch.zeros(max_n, max_n, device=sb.device)
            full[offset:offset+n, offset:offset+n] = sb
            pat_padded = full.unsqueeze(0)
            resp = model(pat_padded, mask.unsqueeze(0), cfg_vec)[0]
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
            g = torch.autograd.grad(loss, sb)[0]
            all_grads.append(g.detach())
        return torch.stack(all_grads).mean(0)


def build_target_idx(theta_c, width):
    sample_per_deg = 2
    center = int(round((theta_c + 90) * sample_per_deg))
    half = int(round(width * sample_per_deg / 2))
    return max(0, center - half), min(361, center + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v3")
    p.add_argument("--n_samples_per_config", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device
    device = args.device

    # Train het ensemble on FULL dataset_v3
    ds = RISDataset(Path(args.dataset))
    print(f"Training het ensemble on full dataset_v3 ({len(ds)} entries)...")
    indices = list(range(len(ds)))
    archs = [(16, 3), (32, 4), (64, 5)]
    ensemble = []
    for k, (ch, dp) in enumerate(archs):
        print(f"  Training arch (c={ch}, d={dp})...")
        m = train_with_arch(ds, indices, channels=ch, depth=dp, epochs=100, device=device, seed=k)
        ensemble.append(m)
    for m in ensemble:
        m.eval()

    test_configs = [
        (38e9, 31, 0.0, 20.0, 51.0, 2.0),
        (38e9, 31, -30.0, 10.0, 51.0, 2.0),
        (28e9, 31, 0.0, 20.0, 51.0, 2.0),
        (38e9, 41, 0.0, 10.0, 51.0, 2.0),
        (38e9, 31, 30.0, 30.0, 51.0, 0.0),
    ]

    cos_sims_mean = []
    cos_sims_avg = []
    rel_errs = []

    print(f"\n{'config':<35} | {'cos(mean)':>9} | {'cos(avg)':>9} | {'rel err':>9}")
    print("-" * 75)
    for freq, n, tc, tw, inc, rw in test_configs:
        sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
        main_lo, main_hi = build_target_idx(tc, tw)
        max_n = 41
        offset = (max_n - n) // 2
        mask = torch.zeros(max_n, max_n, device=device)
        mask[offset:offset+n, offset:offset+n] = 1.0
        cfg_vec = torch.tensor([
            freq / 100e9, n / 41.0, tc / 90.0, tw / 90.0, inc / 90.0, rw / 5.0,
        ], device=device).unsqueeze(0)

        for s in range(args.n_samples_per_config):
            torch.manual_seed(s)
            params = torch.rand(n, n, device=device) * 2.0
            params.requires_grad_(True)
            opt = torch.optim.Adam([params], lr=0.05)
            for _ in range(200):
                opt.zero_grad()
                resp = sim(params)["response"]
                ll = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
                ll.backward()
                opt.step()
            with torch.no_grad():
                phase = (params * torch.pi) % (2 * torch.pi)
                soft_bin = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()

            g_real = gradient_through_real_sim(sim, soft_bin, main_lo, main_hi, rw)
            g_pred_mean = gradient_through_ensemble(
                ensemble, soft_bin, max_n, mask, cfg_vec, main_lo, main_hi, rw, mode="mean")
            g_pred_avg = gradient_through_ensemble(
                ensemble, soft_bin, max_n, mask, cfg_vec, main_lo, main_hi, rw, mode="avg_grads")

            cos_mean = F.cosine_similarity(
                g_real.flatten().unsqueeze(0), g_pred_mean.flatten().unsqueeze(0), dim=1).item()
            cos_avg = F.cosine_similarity(
                g_real.flatten().unsqueeze(0), g_pred_avg.flatten().unsqueeze(0), dim=1).item()
            rel_err = (g_real - g_pred_mean).norm().item() / (g_real.norm().item() + 1e-8)

            cos_sims_mean.append(cos_mean)
            cos_sims_avg.append(cos_avg)
            rel_errs.append(rel_err)

        print(f"f={freq/1e9:.0f}G n={n} θc={tc:+.0f} w={tw:.0f}      | "
              f"{np.mean(cos_sims_mean[-args.n_samples_per_config:]):+9.3f} | "
              f"{np.mean(cos_sims_avg[-args.n_samples_per_config:]):+9.3f} | "
              f"{np.mean(rel_errs[-args.n_samples_per_config:]):9.3f}")

    print(f"\n=== Aggregate ({len(cos_sims_mean)} samples) ===")
    print(f"  Mean cos similarity (predict-then-loss):  {np.mean(cos_sims_mean):+.3f}")
    print(f"  Mean cos similarity (avg gradients):       {np.mean(cos_sims_avg):+.3f}")
    print(f"  Mean rel error:                            {np.mean(rel_errs):.3f}")
    print(f"\n  R79 single CNN baseline: cos sim 0.001, rel err 1.0")
    print(f"  Pass threshold for GD-through-surrogate: cos > 0.7, rel < 0.5")


if __name__ == "__main__":
    main()
