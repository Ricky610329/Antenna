"""Round 125 -- R121 CHAMPION cross-steering-angle validation.

Patch arrays 通常需要 beam steering (主波 off broadside)。
R123/R124 測過 inc + freq 軸都 universal pass。
R125 驗證 main beam 不在 broadside 時 R121 是否還 hold。

Steering angles tested (main beam center):
  -30 deg, -15 deg, 0 deg (broadside, R121 baseline), +15 deg, +30 deg, +45 deg

Width 固定 15 deg (broadside spec)。
THETA_DEG = arange(-90, 90.1, 0.5) -> 361 grid。
broadside 0 deg -> index 180; main 15 deg width -> [172, 188]。

Steering by index shift:
  -30 deg -> [112, 128]
  -15 deg -> [142, 158]
    0 deg -> [172, 188]   (broadside, R121 baseline equivalent)
  +15 deg -> [202, 218]
  +30 deg -> [232, 248]
  +45 deg -> [262, 278]
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

def quantize_2bit(params):
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi*3/4, 4, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(main_lo, main_hi, mean_w, n_levels):
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
            loss = -(mm - sx) + 2.0 * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            if n_levels == 4:
                quantized = quantize_2bit(params)
            else:
                phase = (params * torch.pi) % (2 * torch.pi)
                quantized = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_b = sim(quantized)["response"].cpu().numpy()
        main = resp_b[main_lo:main_hi]; side = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main.min()-side.max()), "side_max": float(side.max()),
             "side_mean": float(side.mean()), "ripple": float(main.max()-main.min()),
             "main_min": float(main.min()), "flat_top": int(np.sum(main<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 100)
print("R125 -- R121 CHAMPION (2-bit + lambda=1) cross-steering-angle validation")
print(f"  n=51, freq=38GHz, inc=51deg, main width 15deg, 5 restarts x 1500 steps")
print("=" * 100)

# Each (lo, hi) corresponds to main beam center at angle:
#   center_deg = -90 + (lo+hi)/2 * 0.5
configs = [
    (-30, 112, 128),
    (-15, 142, 158),
    (  0, 172, 188),
    (+15, 202, 218),
    (+30, 232, 248),
    (+45, 262, 278),
]

print(f"\n{'steer':>6} | {'recipe':<25} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'main_min':>9} | {'flat-top':>9}")
print("-" * 105)
for steer_deg, lo, hi in configs:
    b1, f1 = deploy(lo, hi, 0.0, 2)
    print(f"{steer_deg:>+5}d | {'1-bit (R94 baseline)':<25} | {b1['worst']:>+7.2f} | {b1['side_max']:>+9.2f} | "
          f"{b1['side_mean']:>+10.2f} | {b1['main_min']:>+9.2f} | {'OK' if f1==5 else f'{f1}/5':>9}")
    b2, f2 = deploy(lo, hi, 1.0, 4)
    print(f"{steer_deg:>+5}d | {'2-bit+lam=1 (R121)':<25} | {b2['worst']:>+7.2f} | {b2['side_max']:>+9.2f} | "
          f"{b2['side_mean']:>+10.2f} | {b2['main_min']:>+9.2f} | {'OK' if f2==5 else f'{f2}/5':>9}")
    print()
