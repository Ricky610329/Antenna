"""Round 113 — rw=5 + multi-target combined."""

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

target_1 = {"name": "broadside", "main_lo": 162, "main_hi": 192}
target_2 = {"name": "+30deg",   "main_lo": 218, "main_hi": 248}

def deploy_multi(targets, rw):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            losses = [worst_case_loss(resp, t["main_lo"], t["main_hi"], beta=20.0, ripple_weight=rw)
                      for t in targets]
            loss = sum(losses)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()
        metrics = [evaluate_metrics(resp_bin, t["main_lo"], t["main_hi"]) for t in targets]
        score = sum(m["worst_supp"] for m in metrics)
        if best is None or score > best["score"]:
            best = {"score": score, "metrics": metrics}
    return best

print("=" * 70)
print("R113 — rw=5 helps multi-target?")
print("=" * 70)

# Single targets
print("\n--- Baseline single-target rw=2 ---")
b1 = deploy_multi([target_1], rw=2.0)
print(f"  T1 alone: worst={b1['metrics'][0]['worst_supp']:+.2f}, "
      f"flat-top={'yes' if b1['metrics'][0]['flat_top_compliant'] else 'no'}")

# Multi-target rw=2 (R102 baseline)
print("\n--- R102 multi-target rw=2 ---")
b_mt2 = deploy_multi([target_1, target_2], rw=2.0)
m1, m2 = b_mt2["metrics"]
print(f"  T1: worst={m1['worst_supp']:+.2f}, flat-top={'yes' if m1['flat_top_compliant'] else 'no'}")
print(f"  T2: worst={m2['worst_supp']:+.2f}, flat-top={'yes' if m2['flat_top_compliant'] else 'no'}")

# Multi-target rw=5 (R113 NEW)
print("\n--- R113 multi-target rw=5 ---")
b_mt5 = deploy_multi([target_1, target_2], rw=5.0)
m1_5, m2_5 = b_mt5["metrics"]
print(f"  T1: worst={m1_5['worst_supp']:+.2f}, flat-top={'yes' if m1_5['flat_top_compliant'] else 'no'}")
print(f"  T2: worst={m2_5['worst_supp']:+.2f}, flat-top={'yes' if m2_5['flat_top_compliant'] else 'no'}")

print("\n" + "=" * 70)
print(f"{'mode':<25} | {'T1 worst':>9} | {'T2 worst':>9} | {'T1 flat':>8} | {'T2 flat':>8}")
print("-" * 70)
print(f"{'Single T1 baseline':<25} | {b1['metrics'][0]['worst_supp']:>+9.2f} | {'N/A':>9} | "
      f"{'yes' if b1['metrics'][0]['flat_top_compliant'] else 'no':>8} | {'-':>8}")
print(f"{'Multi-target rw=2 (R102)':<25} | {m1['worst_supp']:>+9.2f} | {m2['worst_supp']:>+9.2f} | "
      f"{'yes' if m1['flat_top_compliant'] else 'no':>8} | {'yes' if m2['flat_top_compliant'] else 'no':>8}")
print(f"{'Multi-target rw=5 (R113)':<25} | {m1_5['worst_supp']:>+9.2f} | {m2_5['worst_supp']:>+9.2f} | "
      f"{'yes' if m1_5['flat_top_compliant'] else 'no':>8} | {'yes' if m2_5['flat_top_compliant'] else 'no':>8}")
