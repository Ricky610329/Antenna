"""Round 102 — Multi-target: single pattern serving 2 targets simultaneously."""

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
freq = 38e9
inc = 51.0
gd_steps = 1500
n_restarts = 5

# Two targets
target_1 = {"name": "broadside_15deg", "main_lo": 162, "main_hi": 192}
target_2 = {"name": "+30deg_15deg",   "main_lo": 218, "main_hi": 248}  # θ_c=+29° to +44°

def deploy_multi_target(targets, rw=2.0):
    """Single pattern, sum of worst_case losses across targets."""
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            # Sum worst-case loss across all targets
            losses = [worst_case_loss(resp, t["main_lo"], t["main_hi"], beta=20.0, ripple_weight=rw)
                      for t in targets]
            loss = sum(losses)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()
        # Eval on each target
        metrics_per_target = [evaluate_metrics(resp_bin, t["main_lo"], t["main_hi"]) for t in targets]
        # Score: sum of worst supps
        score = sum(m["worst_supp"] for m in metrics_per_target)
        if best is None or score > best["score"]:
            best = {"score": score, "metrics": metrics_per_target, "binary": binary.cpu().numpy()}
    return best


print("=" * 75)
print("R102 — Multi-target: single pattern at 2 targets")
print("  Targets: broadside (15°) + +30° (15°), n=51, rw=2, 5 restarts")
print("=" * 75)

# Baseline: single-target (each separately)
print("\n--- Baseline 1: broadside only ---")
b1 = deploy_multi_target([target_1])
m1_base = b1["metrics"][0]
print(f"  worst={m1_base['worst_supp']:+.2f}, ripple={m1_base['main_ripple']:.2f}, "
      f"flat-top={'yes' if m1_base['flat_top_compliant'] else 'no'}")

print("\n--- Baseline 2: +30° only ---")
b2 = deploy_multi_target([target_2])
m2_base = b2["metrics"][0]
print(f"  worst={m2_base['worst_supp']:+.2f}, ripple={m2_base['main_ripple']:.2f}, "
      f"flat-top={'yes' if m2_base['flat_top_compliant'] else 'no'}")

print("\n--- Multi-target (both simultaneously) ---")
b_multi = deploy_multi_target([target_1, target_2])
m1_mt = b_multi["metrics"][0]
m2_mt = b_multi["metrics"][1]
print(f"  Target 1 (broadside): worst={m1_mt['worst_supp']:+.2f}, ripple={m1_mt['main_ripple']:.2f}, "
      f"flat-top={'yes' if m1_mt['flat_top_compliant'] else 'no'}")
print(f"  Target 2 (+30°):      worst={m2_mt['worst_supp']:+.2f}, ripple={m2_mt['main_ripple']:.2f}, "
      f"flat-top={'yes' if m2_mt['flat_top_compliant'] else 'no'}")

print("\n" + "=" * 75)
print(f"{'mode':<25} | {'T1 worst':>10} | {'T2 worst':>10} | {'T1 flat':>9} | {'T2 flat':>9}")
print("-" * 75)
print(f"{'baseline (T1 only)':<25} | {m1_base['worst_supp']:>+10.2f} | {'N/A':>10} | "
      f"{'yes' if m1_base['flat_top_compliant'] else 'no':>9} | {'-':>9}")
print(f"{'baseline (T2 only)':<25} | {'N/A':>10} | {m2_base['worst_supp']:>+10.2f} | "
      f"{'-':>9} | {'yes' if m2_base['flat_top_compliant'] else 'no':>9}")
print(f"{'multi-target':<25} | {m1_mt['worst_supp']:>+10.2f} | {m2_mt['worst_supp']:>+10.2f} | "
      f"{'yes' if m1_mt['flat_top_compliant'] else 'no':>9} | "
      f"{'yes' if m2_mt['flat_top_compliant'] else 'no':>9}")

# Single-target degradation 計算
deg1 = m1_base['worst_supp'] - m1_mt['worst_supp']
deg2 = m2_base['worst_supp'] - m2_mt['worst_supp']
print(f"\nDegradation when pursuing both:")
print(f"  T1 degradation: {deg1:+.2f} dB")
print(f"  T2 degradation: {deg2:+.2f} dB")
