"""Round 124 — R121 champion cross-frequency validation.

R123 顯示 2-bit + λ=1 universally 解 inc 問題。
本輪驗證 frequency 軸：
  - 5.8 GHz (sub-6 patch territory)
  - 28 GHz (mmWave low)
  - 38 GHz (mmWave standard, R121 baseline)
  - 60 GHz (mmWave high)

如果 R121 recipe 在 freq 軸也 universal，可宣告 patch transition recipe 完整。
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
inc = 51.0
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def quantize_2bit(params):
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi*3/4, 4, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(freq_hz, recipe_name, mean_w, n_levels):
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
             "flat_top": int(np.sum(main<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 95)
print("R124 -- R121 CHAMPION (2-bit + lambda=1) cross-frequency validation")
print(f"  n=51, inc=51deg, broadside 15deg width, 5 restarts x 1500 steps")
print("=" * 95)

print(f"\n{'freq':>8} | {'recipe':<25} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'flat-top':>9}")
print("-" * 88)
for freq_label, freq_hz in [("5.8GHz", 5.8e9), ("28GHz", 28e9), ("38GHz", 38e9), ("60GHz", 60e9)]:
    b1, f1 = deploy(freq_hz, "1-bit baseline", 0.0, 2)
    print(f"{freq_label:>8} | {'1-bit (R94 baseline)':<25} | {b1['worst']:>+7.2f} | {b1['side_max']:>+9.2f} | "
          f"{b1['side_mean']:>+10.2f} | {'OK' if f1==5 else f'{f1}/5':>9}")
    b2, f2 = deploy(freq_hz, "R121", 1.0, 4)
    print(f"{freq_label:>8} | {'2-bit+lam=1 (R121)':<25} | {b2['worst']:>+7.2f} | {b2['side_max']:>+9.2f} | "
          f"{b2['side_mean']:>+10.2f} | {'OK' if f2==5 else f'{f2}/5':>9}")
    print()
