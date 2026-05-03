"""Round 117 — 3-bit phase + multi-target test."""

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

def quantize(params, n_levels):
    phase = (params * torch.pi) % (2 * torch.pi)
    if n_levels == 0:
        return phase / torch.pi
    levels = torch.linspace(0, 2*torch.pi*(n_levels-1)/n_levels, n_levels, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    idx = dist.argmin(0)
    return levels[idx] / torch.pi

def deploy(targets, n_levels, rw=2.0):
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
            quantized = quantize(params, n_levels)
            resp_q = sim(quantized)["response"].cpu().numpy()
        metrics = [evaluate_metrics(resp_q, t["main_lo"], t["main_hi"]) for t in targets]
        score = sum(m["worst_supp"] for m in metrics)
        if best is None or score > best["score"]:
            best = {"score": score, "metrics": metrics}
    return best

print("=" * 70)
print("R117 — Does multi-bit phase reduce multi-target penalty?")
print("=" * 70)

trials = [
    {"name": "1bit single T1", "n_levels": 2, "targets": [target_1]},
    {"name": "1bit multi (R102)", "n_levels": 2, "targets": [target_1, target_2]},
    {"name": "2bit multi", "n_levels": 4, "targets": [target_1, target_2]},
    {"name": "3bit multi (NEW)", "n_levels": 8, "targets": [target_1, target_2]},
    {"name": "3bit single T1 (ref)", "n_levels": 8, "targets": [target_1]},
]

print(f"\n{'config':<25} | {'T1 worst':>9} | {'T2 worst':>9} | {'T1 flat':>8} | {'T2 flat':>8}")
print("-" * 75)
for t in trials:
    b = deploy(t["targets"], t["n_levels"])
    if len(t["targets"]) == 1:
        m1 = b["metrics"][0]
        print(f"{t['name']:<25} | {m1['worst_supp']:>+9.2f} | {'N/A':>9} | "
              f"{'yes' if m1['flat_top_compliant'] else 'no':>8} | {'-':>8}")
    else:
        m1, m2 = b["metrics"]
        print(f"{t['name']:<25} | {m1['worst_supp']:>+9.2f} | {m2['worst_supp']:>+9.2f} | "
              f"{'yes' if m1['flat_top_compliant'] else 'no':>8} | "
              f"{'yes' if m2['flat_top_compliant'] else 'no':>8}")
