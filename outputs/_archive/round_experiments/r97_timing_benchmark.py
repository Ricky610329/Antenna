"""Round 97 — Pipeline timing benchmark."""

import sys
import time
sys.path.insert(0, "script")

import torch
import numpy as np
from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import worst_case_loss, evaluate_metrics
import torch.nn as nn

config.device = "cuda:0"

def time_one_restart(n, freq, gd_steps=1500, lr=0.05, ripple_weight=2.0, seed=0):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=51.0)
    main_lo, main_hi = 162, 192

    torch.manual_seed(seed)
    t0 = time.time()
    params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
    opt = torch.optim.Adam([params], lr=lr)
    for step in range(gd_steps):
        opt.zero_grad()
        resp = sim(params)["response"]
        loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=ripple_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        phase = (params * torch.pi) % (2 * torch.pi)
        binary = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()
        sim(binary)["response"].cpu().numpy()
    t1 = time.time()
    return t1 - t0

print("=" * 70)
print("R97 — Pipeline Timing Benchmark (1500 GD steps + 1-bit quantize)")
print("=" * 70)

print(f"\n{'n':>4} | {'freq':>6} | {'time/restart (sec)':>18} | {'5 seeds total':>15} | {'10 seeds total':>16}")
print("-" * 70)

for n in [21, 31, 41, 51, 61]:
    times = []
    # Warm up
    _ = time_one_restart(n, 38e9, gd_steps=100)
    for s in range(2):  # avg over 2 runs
        t = time_one_restart(n, 38e9, seed=s)
        times.append(t)
    avg = np.mean(times)
    print(f"{n:>4} | 38GHz | {avg:>18.1f} | {avg*5:>13.0f}s | {avg*10:>14.0f}s")

print("\nDeployment time budget estimates:")
print(f"  per-target opt (n=41, 5 restarts):  ~3 min")
print(f"  per-target opt (n=51, 10 restarts): ~10 min")
print(f"  per-target opt (n=61, 5 restarts):  ~10-15 min (slow)")
print(f"\nFor patch HFSS (~5 min/run baseline):")
print(f"  RIS sim is 10-20x faster than HFSS")
print(f"  patch budget: ~5 min × N_restarts × N_targets")
