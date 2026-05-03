"""Round 135 — probe width transition zone (12/15/18 deg) for 1-bit selector.

R134 selector failed at n=51, inc=51, 38GHz, width=15deg with R119 (rw=2 lam=1):
  -> worst +3.40, flat 3/5 (just under threshold)

Hypothesis: width=15deg sits in transition between narrow recipe (R119, rw=2)
and wide-cap recipe (R129, rw=3). Need finer width branches in selector.

Probe both recipes at widths 12, 15, 18 deg to find clean cutover.
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 51; freq = 38e9; inc = 51.0
gd_steps = 1500; n_restarts = 5

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

def deploy_1bit(width_deg, mean_w, ripple_w):
    main_lo, main_hi = steer_to_indices(0, width_deg)
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None; seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_b = sim(binary)["response"].cpu().numpy()
        main_arr = resp_b[main_lo:main_hi]; side_arr = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main_arr.min()-side_arr.max()),
             "side_mean": float(side_arr.mean()),
             "ripple": float(main_arr.max()-main_arr.min()),
             "flat_top": int(np.sum(main_arr<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 100, flush=True)
print(f"R135 -- width transition zone probe (n=51, inc=51, 38GHz, 1-bit)", flush=True)
print(f"  Goal: find clean cutover between R119 (rw=2, lam=1) and R129 (rw=3, lam=1)", flush=True)
print("=" * 100, flush=True)

print(f"\n{'width':>6} | {'recipe':<25} | {'worst':>7} | {'side_mean':>10} | {'ripple':>7} | {'flat':>5}", flush=True)
print("-" * 75, flush=True)
for w in [12, 15, 18]:
    print(f"[runner] width={w}deg ...", flush=True)
    for name, rw, lam in [("R119 (rw=2 lam=1)", 2.0, 1.0),
                          ("R129 (rw=3 lam=1)", 3.0, 1.0)]:
        b, f = deploy_1bit(w, lam, rw)
        flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
        passed = (f >= 4) and (b["worst"] > 0)
        marker = " PASS" if passed else " fail"
        print(f"{w:>5}d | {name:<25} | {b['worst']:>+7.2f} | {b['side_mean']:>+10.2f} | "
              f"{b['ripple']:>+7.2f} | {flat_str:>5}{marker}", flush=True)
    print(flush=True)
