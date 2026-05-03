"""Round 131 — rescue inc=0deg + mmWave failure for 1-bit narrow-cap.

R130 found inc=0deg + 28/38/60 GHz fails R119 recipe at narrow cap (10deg).
R131 grid-searches (rw, lambda) at the hardest case (inc=0deg, 38GHz)
to see if a recipe variant rescues normal-incidence mmWave.

Constraint: 1-bit ONLY.
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 51
inc = 0.0  # normal incidence (the hard case)
gd_steps = 1500
n_restarts = 3

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

main_lo, main_hi = steer_to_indices(0, 10)  # broadside, width=10deg

def deploy_1bit(freq_hz, mean_w, ripple_w):
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc)
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

rws = [2.0, 3.0, 5.0, 8.0]
lams = [0.3, 0.5, 1.0]

for freq_hz, freq_label in [(28e9, "28GHz"), (38e9, "38GHz"), (60e9, "60GHz")]:
    print("=" * 95, flush=True)
    print(f"R131 -- inc=0deg, freq={freq_label}, width=10deg, 1-bit, n=51 (rescue grid)", flush=True)
    print("=" * 95, flush=True)
    print(f"\n{'rw':>4} | {'lam':>4} | {'worst':>7} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}", flush=True)
    print("-" * 60, flush=True)
    for rw in rws:
        for lam in lams:
            b, f = deploy_1bit(freq_hz, lam, rw)
            flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
            print(f"{rw:>4.1f} | {lam:>4.1f} | {b['worst']:>+7.2f} | {b['side_mean']:>+10.2f} | "
                  f"{b['ripple']:>+7.2f} | {flat_str:>9}", flush=True)
        print(flush=True)
    print(flush=True)
