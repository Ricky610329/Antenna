"""Round 127 -- Does bigger aperture break the +45 deg steering boundary?

R126 hypothesized: at n=51, +45deg, even continuous phase saturates at +1.32 dB.
This is a physical aperture limit, not hardware.

R127 directly tests by sweeping n at +45 deg steering with R121 recipe:
  n = 31, 41, 51, 71, 91

If the hypothesis is correct, bigger n should break the +1.32 ceiling.
Otherwise the boundary is something else (loss design, freq, inc).

Also test broadside (steer=0) at same n values as control.

Theta grid: arange(-90, 90.1, 0.5) -> 361 entries.
broadside  -> [172, 188]  (15deg width centered on 0deg)
+45deg     -> [262, 278]
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
freq = 38e9; inc = 51.0
gd_steps = 1500; n_restarts = 5  # n=71 needs reduce to 3 due to 8GB VRAM

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def quantize_2bit(params):
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi*3/4, 4, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(n, main_lo, main_hi):
    """R121 CHAMPION recipe: 2-bit + lambda_mean=1 + ripple_w=2."""
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
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            quantized = quantize_2bit(params)
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
print("R127 -- Aperture sweep at broadside vs +45deg with R121 CHAMPION recipe")
print(f"  freq=38GHz, inc=51deg, width=15deg, 5 restarts x 1500 steps")
print(f"  Question: does bigger n break the +45deg boundary discovered in R126?")
print("=" * 100)

n_values = [31, 51, 71]  # n=71 ran separately with n_restarts=3 due to VRAM

print(f"\n{'n':>4} | {'steer':>6} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat':>5}", flush=True)
print("-" * 80, flush=True)
for n in n_values:
    print(f"[runner] starting n={n} broadside ...", flush=True)
    b, f = deploy(n, 172, 188)
    print(f"{n:>4} | {' 0d':>6} | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | {b['side_mean']:>+10.2f} | "
          f"{b['ripple']:>+7.2f} | {'OK' if f==5 else f'{f}/5':>5}", flush=True)
    print(f"[runner] starting n={n} +45deg ...", flush=True)
    b, f = deploy(n, 262, 278)
    print(f"{n:>4} | {'+45d':>6} | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | {b['side_mean']:>+10.2f} | "
          f"{b['ripple']:>+7.2f} | {'OK' if f==5 else f'{f}/5':>5}", flush=True)
    print(flush=True)
