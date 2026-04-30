"""
Round 79 — 量測 surrogate gradient quality vs real-sim

對 N 個 random sampled points，比較:
- ∂loss/∂params via real RIS sim (autograd ground truth)
- ∂loss/∂params via R72 surrogate (autograd)

Metrics:
- Cosine similarity (gradient direction agreement)
- Relative error (gradient magnitude)
- Per-pixel gradient correlation

實證 R78 的 hypothesis: surrogate function MAE 好但 gradient quality 不夠。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antenna.ris import RISSimulator
from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import SurrogateCNN


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


def compute_gradient(model_or_sim, soft_bin: torch.Tensor, main_lo: int, main_hi: int,
                      surrogate_inputs=None, ripple_w=2.0):
    """Return ∂loss/∂soft_bin for given model_or_sim."""
    soft_bin = soft_bin.detach().clone().requires_grad_(True)
    if surrogate_inputs is not None:  # surrogate path
        max_n, mask, cfg_vec = surrogate_inputs
        n = soft_bin.shape[0]
        offset = (max_n - n) // 2
        pat_padded = torch.zeros(1, max_n, max_n, device=soft_bin.device)
        pat_padded[0, offset:offset+n, offset:offset+n] = soft_bin
        # Re-attach grad through padded form
        pat_padded = pat_padded.clone()
        pat_padded[0, offset:offset+n, offset:offset+n] = soft_bin  # ensures gradient connects
        resp = model_or_sim(pat_padded, mask.unsqueeze(0), cfg_vec)[0]
    else:  # real-sim path
        resp = model_or_sim(soft_bin)["response"]
    loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_w)
    grad = torch.autograd.grad(loss, soft_bin, create_graph=False)[0]
    return grad.detach(), loss.detach()


def compute_gradient_padded(surrogate, soft_bin, max_n, mask, cfg_vec, main_lo, main_hi, ripple_w):
    """Surrogate path: gradient through padded surrogate input."""
    n = soft_bin.shape[0]
    offset = (max_n - n) // 2
    soft_bin = soft_bin.detach().clone().requires_grad_(True)
    pat_padded = torch.zeros(1, max_n, max_n, device=soft_bin.device)
    # Use addition to maintain grad
    template = torch.zeros_like(pat_padded)
    template[0, offset:offset+n, offset:offset+n] = 1.0
    pat_padded = template.clone()
    full = torch.zeros(max_n, max_n, device=soft_bin.device)
    full[offset:offset+n, offset:offset+n] = soft_bin
    pat_padded = full.unsqueeze(0)
    resp = surrogate(pat_padded, mask.unsqueeze(0), cfg_vec)[0]
    loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_w)
    grad = torch.autograd.grad(loss, soft_bin, create_graph=False)[0]
    return grad.detach(), loss.detach()


def compute_gradient_real(sim, soft_bin, main_lo, main_hi, ripple_w):
    soft_bin = soft_bin.detach().clone().requires_grad_(True)
    resp = sim(soft_bin)["response"]
    loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_w)
    grad = torch.autograd.grad(loss, soft_bin, create_graph=False)[0]
    return grad.detach(), loss.detach()


def build_target_idx(theta_c, width):
    sample_per_deg = 2
    center = int(round((theta_c + 90) * sample_per_deg))
    half = int(round(width * sample_per_deg / 2))
    return max(0, center - half), min(361, center + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate_pt", type=str, default="outputs/r72_cnn_v2/surrogate.pt")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device
    device = args.device
    torch.manual_seed(0)

    surrogate = SurrogateCNN(channels=32, depth=4).to(device)
    surrogate.load_state_dict(torch.load(args.surrogate_pt, map_location=device))
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    # Sample diverse points
    test_configs = [
        (38e9, 31, 0.0, 20.0, 51.0, 2.0),
        (38e9, 31, -30.0, 10.0, 51.0, 2.0),
        (28e9, 31, 0.0, 20.0, 51.0, 2.0),
        (38e9, 41, 0.0, 10.0, 51.0, 2.0),
        (38e9, 31, 30.0, 30.0, 51.0, 0.0),
        (28e9, 21, 0.0, 20.0, 51.0, 2.0),
    ]

    cos_sims = []
    rel_errs = []
    main_acc_pcts = []
    side_acc_pcts = []
    func_errs = []

    print(f"\n{'config':<40} | {'cos sim':>9} | {'rel err':>9} | {'func err':>9}")
    print("-" * 75)
    for freq, n, tc, tw, inc, rw in test_configs:
        sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
        main_lo, main_hi = build_target_idx(tc, tw)

        # Build surrogate inputs
        max_n = 41
        offset = (max_n - n) // 2
        mask = torch.zeros(max_n, max_n, device=device)
        mask[offset:offset+n, offset:offset+n] = 1.0
        cfg_vec = torch.tensor([
            freq / 100e9, n / 41.0, tc / 90.0, tw / 90.0, inc / 90.0, rw / 5.0,
        ], device=device).unsqueeze(0)

        # Average over n_samples optimized patterns (in dataset distribution)
        sims_list, errs_list = [], []
        for s in range(args.n_samples):
            torch.manual_seed(s)
            # Generate an optimized binary pattern via short GD on real sim (closer to dataset distribution)
            params_init = torch.rand(n, n, device=device) * 2.0
            params_init.requires_grad_(True)
            opt_init = torch.optim.Adam([params_init], lr=0.05)
            for _ in range(200):
                opt_init.zero_grad()
                resp = sim(params_init)["response"]
                ll = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
                ll.backward()
                opt_init.step()
            with torch.no_grad():
                phase = (params_init * torch.pi) % (2 * torch.pi)
                soft_bin = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()

            g_real, loss_real = compute_gradient_real(sim, soft_bin, main_lo, main_hi, rw)
            g_pred, loss_pred = compute_gradient_padded(surrogate, soft_bin, max_n, mask, cfg_vec,
                                                        main_lo, main_hi, rw)

            cos = F.cosine_similarity(g_real.flatten().unsqueeze(0),
                                       g_pred.flatten().unsqueeze(0), dim=1).item()
            rel = (g_real - g_pred).norm().item() / (g_real.norm().item() + 1e-8)
            func_err = abs(loss_real.item() - loss_pred.item())

            sims_list.append(cos)
            errs_list.append(rel)
            cos_sims.append(cos)
            rel_errs.append(rel)
            func_errs.append(func_err)

        cfg_str = f"f={freq/1e9:.0f}G n={n} θc={tc:+.0f} w={tw:.0f}"
        print(f"{cfg_str:<40} | {np.mean(sims_list):+9.3f} | {np.mean(errs_list):9.3f} | "
              f"{np.mean(func_errs):9.3f}")

    print()
    print(f"=== Aggregate over {args.n_samples} × {len(test_configs)} = "
          f"{args.n_samples * len(test_configs)} samples ===")
    print(f"Mean cosine similarity: {np.mean(cos_sims):+.3f} (1.0 perfect, 0 random)")
    print(f"Median cosine similarity: {np.median(cos_sims):+.3f}")
    print(f"Mean relative gradient error: {np.mean(rel_errs):.3f}")
    print(f"Median relative gradient error: {np.median(rel_errs):.3f}")
    print(f"Mean |loss_real - loss_pred|: {np.mean(func_errs):.3f} dB")
    print()
    print(f"Pass criteria for surrogate-in-loop deployment:")
    print(f"  cosine similarity > 0.7  (R76 protocol)")
    print(f"  relative error < 0.5     (R76 protocol)")
    pass_cos = np.mean(cos_sims) > 0.7
    pass_rel = np.mean(rel_errs) < 0.5
    print(f"  → cos {'PASS' if pass_cos else 'FAIL'}, rel {'PASS' if pass_rel else 'FAIL'}")


if __name__ == "__main__":
    main()
