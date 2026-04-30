"""Round 101 — Methodology ablation study."""

import sys, time
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
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5

def run_full_pipeline(use_worst_case=True, use_multi_restart=True,
                     use_free_phase=True, use_optimal_quantize=True,
                     use_ripple_penalty=True):
    """Run pipeline with ablations."""
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    target = torch.full((361,), -25.0, device="cuda:0")
    target[main_lo:main_hi] = 0.0
    
    n_seeds = n_restarts if use_multi_restart else 1
    best = None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        if use_free_phase:
            params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        else:
            # Sigmoid path (R57 old, half-circle [0, π])
            params = nn.Parameter(torch.randn(n, n, device="cuda:0"))
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            if use_free_phase:
                pat_in = params
            else:
                pat_in = torch.sigmoid(params)  # in [0,1] → phase [0, π]
            resp = sim(pat_in)["response"]
            
            if use_worst_case:
                rw = 2.0 if use_ripple_penalty else 0.0
                loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
            else:
                # Max-max loss (R57-R63 style)
                from antenna.ris import custom_loss_tolerance
                loss = custom_loss_tolerance(resp, target, sidelobe_threshold=-25.0,
                                             main_target=0.0, main_weight=5.0)
            loss.backward()
            opt.step()
        
        # Quantize
        with torch.no_grad():
            if use_free_phase:
                phase = (params * torch.pi) % (2 * torch.pi)
                if use_optimal_quantize:
                    binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
                else:
                    # Naive: phase > π
                    binary = (phase > torch.pi).float()
            else:
                soft = torch.sigmoid(params)
                binary = (soft > 0.5).float()  # only choice for sigmoid
            resp_bin = sim(binary)["response"].cpu().numpy()
        
        m = evaluate_metrics(resp_bin, main_lo, main_hi)
        if best is None or m["worst_supp"] > best["worst_supp"]:
            best = m
    return best


print("=" * 75)
print("R101 — Ablation Study (n=51, 38 GHz, 15° broadside flat-top)")
print("=" * 75)

ablations = [
    ("FULL recommended pipeline", {}),
    ("✗ worst-case loss (use max-max)", {"use_worst_case": False}),
    ("✗ multi-restart (1 seed only)", {"use_multi_restart": False}),
    ("✗ free-phase (use sigmoid)", {"use_free_phase": False}),
    ("✗ optimal quantize (use naive >π)", {"use_optimal_quantize": False}),
    ("✗ ripple penalty (rw=0)", {"use_ripple_penalty": False}),
]

results = []
for name, kwargs in ablations:
    t0 = time.time()
    m = run_full_pipeline(**kwargs)
    t = time.time() - t0
    results.append((name, m, t))
    print(f"\n{name}: worst={m['worst_supp']:+.2f}, ripple={m['main_ripple']:.2f}, "
          f"flat-top={'yes' if m['flat_top_compliant'] else 'no'}, time={t:.0f}s")

print("\n" + "=" * 90)
print(f"{'config':<40} | {'worst':>8} | {'ripple':>8} | {'flat-top':>9} | {'Δ vs FULL':>10}")
print("-" * 90)
full_worst = results[0][1]["worst_supp"]
for name, m, _ in results:
    flat = "yes" if m['flat_top_compliant'] else "no"
    delta = m["worst_supp"] - full_worst
    print(f"{name:<40} | {m['worst_supp']:>+8.2f} | {m['main_ripple']:>8.2f} | "
          f"{flat:>9} | {delta:>+10.2f}")
