"""Round 133 — find n=71 rescue recipe at inc=0+60GHz (1-bit narrow cap).

R132 showed n=71 expands worst-case headroom (+0.34 -> +4.86) but rw=2
loses flat-top. Need higher rw to recover, like R129 did for wide cap.

Grid (rw, lambda) at n=71, inc=0deg, 60GHz, width=10deg, 1-bit.
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 71
freq = 60e9
inc = 0.0
gd_steps = 1500
n_restarts = 3  # VRAM constraint at n=71

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

main_lo, main_hi = steer_to_indices(0, 10)

def deploy_1bit(mean_w, ripple_w):
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
             "main_below_3": int(np.sum(main_arr<-3)),
             "flat_top": int(np.sum(main_arr<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 95, flush=True)
print(f"R133 -- n=71 rescue grid at inc=0deg + 60GHz + width=10deg + 1-bit", flush=True)
print(f"  Goal: find (rw, lam) that gives worst > 0 AND flat-top OK", flush=True)
print("=" * 95, flush=True)

# Need higher rw to compensate for n=71 headroom. Probe up to rw=10.
rws = [3.0, 5.0, 8.0, 10.0]
lams = [0.3, 0.5, 1.0]

print(f"\n{'rw':>4} | {'lam':>4} | {'worst':>7} | {'side_mean':>10} | {'ripple':>7} | {'main<-3':>8} | {'flat-top':>9}", flush=True)
print("-" * 75, flush=True)
for rw in rws:
    for lam in lams:
        b, f = deploy_1bit(lam, rw)
        flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
        passed = (f >= n_restarts - 1) and (b["worst"] > 0)
        marker = " PASS" if passed else " fail"
        print(f"{rw:>4.1f} | {lam:>4.1f} | {b['worst']:>+7.2f} | {b['side_mean']:>+10.2f} | "
              f"{b['ripple']:>+7.2f} | {b['main_below_3']:>8d} | {flat_str:>9}{marker}", flush=True)
    print(flush=True)
