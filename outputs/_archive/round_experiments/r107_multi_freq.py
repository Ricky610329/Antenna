"""Round 107 — Multi-frequency test (single pattern, dual-band)."""

import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import worst_case_loss, evaluate_metrics

config.device = "cuda:0"

n = 51
inc = 51.0
main_lo, main_hi = 162, 192  # 15° broadside
gd_steps = 1500
n_restarts = 5

def run_pipeline(freqs, name):
    """Single pattern optimized for given list of freqs."""
    sims = [RISSimulator(element_num=n, freq_hz=f, inc_theta_deg=inc) for f in freqs]
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            losses = []
            for sim in sims:
                resp = sim(params)["response"]
                losses.append(worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0))
            loss = sum(losses)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            metrics_per_freq = []
            for sim in sims:
                resp_bin = sim(binary)["response"].cpu().numpy()
                m = evaluate_metrics(resp_bin, main_lo, main_hi)
                metrics_per_freq.append(m)
        score = sum(m["worst_supp"] for m in metrics_per_freq)
        if best is None or score > best["score"]:
            best = {"score": score, "metrics": metrics_per_freq, "binary": binary.cpu().numpy()}
    return best


print("=" * 75)
print("R107 — Multi-frequency: single pattern at 28+38 GHz")
print("=" * 75)

# Baselines
print("\n--- Baseline 1: 28 GHz only ---")
b1 = run_pipeline([28e9], "28GHz_only")
m_28_alone = b1["metrics"][0]
print(f"  worst={m_28_alone['worst_supp']:+.2f}, ripple={m_28_alone['main_ripple']:.2f}, "
      f"flat-top={'yes' if m_28_alone['flat_top_compliant'] else 'no'}")

print("\n--- Baseline 2: 38 GHz only ---")
b2 = run_pipeline([38e9], "38GHz_only")
m_38_alone = b2["metrics"][0]
print(f"  worst={m_38_alone['worst_supp']:+.2f}, ripple={m_38_alone['main_ripple']:.2f}, "
      f"flat-top={'yes' if m_38_alone['flat_top_compliant'] else 'no'}")

print("\n--- Multi-freq: 28+38 GHz simultaneously ---")
b_multi = run_pipeline([28e9, 38e9], "multi_freq")
m_28_mt = b_multi["metrics"][0]
m_38_mt = b_multi["metrics"][1]
print(f"  At 28 GHz: worst={m_28_mt['worst_supp']:+.2f}, ripple={m_28_mt['main_ripple']:.2f}, "
      f"flat-top={'yes' if m_28_mt['flat_top_compliant'] else 'no'}")
print(f"  At 38 GHz: worst={m_38_mt['worst_supp']:+.2f}, ripple={m_38_mt['main_ripple']:.2f}, "
      f"flat-top={'yes' if m_38_mt['flat_top_compliant'] else 'no'}")

print("\n" + "=" * 75)
print(f"{'mode':<25} | {'@28 worst':>10} | {'@38 worst':>10} | {'@28 ft':>7} | {'@38 ft':>7}")
print("-" * 75)
print(f"{'baseline 28GHz only':<25} | {m_28_alone['worst_supp']:>+10.2f} | {'N/A':>10} | "
      f"{'yes' if m_28_alone['flat_top_compliant'] else 'no':>7} | {'-':>7}")
print(f"{'baseline 38GHz only':<25} | {'N/A':>10} | {m_38_alone['worst_supp']:>+10.2f} | "
      f"{'-':>7} | {'yes' if m_38_alone['flat_top_compliant'] else 'no':>7}")
print(f"{'multi-freq 28+38':<25} | {m_28_mt['worst_supp']:>+10.2f} | {m_38_mt['worst_supp']:>+10.2f} | "
      f"{'yes' if m_28_mt['flat_top_compliant'] else 'no':>7} | "
      f"{'yes' if m_38_mt['flat_top_compliant'] else 'no':>7}")
deg_28 = m_28_alone['worst_supp'] - m_28_mt['worst_supp']
deg_38 = m_38_alone['worst_supp'] - m_38_mt['worst_supp']
print(f"\nDegradation: 28 GHz {deg_28:+.2f} dB, 38 GHz {deg_38:+.2f} dB")
print(f"Compare R102 multi-target: ~5 dB per-target")
