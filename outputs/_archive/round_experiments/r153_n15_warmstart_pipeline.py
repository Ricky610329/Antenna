"""Round 153 — Run R141 pipeline on n=15 RIS via R146 warm-start surrogate.

R152 found the existing pre-trained HFSSNet checkpoint is too inaccurate
for surrogate-loop optimization. Instead, use R146's proven warm-start trick:
extract pre_calAF coefficients from the analytical sim and initialize a
WarmStartSurrogate -> exact match (R^2 = 1.0) without any training.

This validates that the R141 unified pipeline works at a NEW aperture (n=15)
that was outside R141's selector envelope (which covered n=31/51/71). The
selector needs a small extension for n=15.
"""
import sys
sys.path.insert(0, "script")
import time
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 15  # NEW aperture, outside R141 selector
freq = 38e9; inc = 51.0
gd_steps = 1500
n_restarts = 5


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


class ContinuousWarmStartSurrogate(nn.Module):
    def __init__(self, n_elem, n_angles=361):
        super().__init__()
        self.n = n_elem
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)

    def forward(self, x):
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)
        phase = x * np.pi
        cos_p = torch.cos(phase).transpose(1, 2).contiguous().flatten(1)
        sin_p = torch.sin(phase).transpose(1, 2).contiguous().flatten(1)
        F_real = self.real_lin(cos_p) - self.imag_lin(sin_p)
        F_imag = self.real_lin(sin_p) + self.imag_lin(cos_p)
        amp = torch.sqrt(F_real * F_real + F_imag * F_imag + 1e-12)
        peak = amp.max(dim=1, keepdim=True).values
        out = 20.0 * torch.log10(torch.clamp(amp, min=1e-8) / torch.clamp(peak, min=1e-8))
        return {"response": out.squeeze(0) if single else out}


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def eval_binary_on(eval_fn, params, main_lo, main_hi):
    with torch.no_grad():
        phase = (params * torch.pi) % (2 * torch.pi)
        binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
        resp = eval_fn(binary)["response"].cpu().numpy()
    main_arr = resp[main_lo:main_hi]; side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    return {
        "worst": float(main_arr.min() - side_arr.max()),
        "side_mean": float(side_arr.mean()),
        "ripple": float(main_arr.max() - main_arr.min()),
        "flat_top": int(np.sum(main_arr < -3)) == 0,
    }


def run_optimization(forward_fn, sim_for_eval, main_lo, main_hi, mean_w, ripple_w):
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        best_joint_worst = -1e9; best_joint_state = None
        for step in range(gd_steps):
            opt.zero_grad()
            resp = forward_fn(params)["response"]
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
            if (step + 1) % 50 == 0:
                m = eval_binary_on(sim_for_eval, params, main_lo, main_hi)
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()
        if best_joint_state is not None:
            metrics = eval_binary_on(sim_for_eval, best_joint_state, main_lo, main_hi)
        else:
            metrics = eval_binary_on(sim_for_eval, params, main_lo, main_hi)
        metrics["seed"] = seed
        seed_results.append(metrics)
    return seed_results


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
main_lo, main_hi = steer_to_indices(0, 10)

print("=" * 100, flush=True)
print(f"R153 -- R141 pipeline at NEW aperture n={n} via R146 warm-start surrogate", flush=True)
print(f"  freq={freq/1e9}GHz, inc={inc}, w=10, R119 recipe (rw=2, lam=1)", flush=True)
print("=" * 100, flush=True)

# Build warm-start surrogate
print("\n[1/3] Building warm-start surrogate from analytical pre_calAF...", flush=True)
sur = ContinuousWarmStartSurrogate(n).to("cuda:0")
W_re = sim.pre_calAF[0].real.to(torch.float32).to("cuda:0")
W_im = sim.pre_calAF[0].imag.to(torch.float32).to("cuda:0")
with torch.no_grad():
    sur.real_lin.weight.copy_(W_re)
    sur.imag_lin.weight.copy_(W_im)

# Verify match
torch.manual_seed(42)
b_test = torch.bernoulli(torch.full((50, n, n), 0.5)).to("cuda:0")
with torch.no_grad():
    sp = sur(b_test)["response"].cpu().numpy()
    st = torch.stack([sim(b_test[i])["response"] for i in range(50)]).cpu().numpy()
err = float(np.mean(np.abs(sp - st)))
print(f"  Warm-start fit (50 random binary patches): mean abs err = {err:.6f} dB", flush=True)

# Run both modes
print("\n[2/3] Running analytical-truth optimization (5 seeds)...", flush=True)
t0 = time.time()
truth_r = run_optimization(sim, sim, main_lo, main_hi, 1.0, 2.0)
worsts_t = [r["worst"] for r in truth_r]
flats_t = sum(1 for r in truth_r if r["flat_top"])
print(f"  done in {time.time()-t0:.1f}s -> best {max(worsts_t):+.2f}, mean {np.mean(worsts_t):+.2f}, "
      f"flat {flats_t}/{n_restarts}", flush=True)

print("\n[3/3] Running surrogate-loop optimization (5 seeds)...", flush=True)
t0 = time.time()
sur_r = run_optimization(sur, sim, main_lo, main_hi, 1.0, 2.0)
worsts_s = [r["worst"] for r in sur_r]
flats_s = sum(1 for r in sur_r if r["flat_top"])
print(f"  done in {time.time()-t0:.1f}s -> best {max(worsts_s):+.2f}, mean {np.mean(worsts_s):+.2f}, "
      f"flat {flats_s}/{n_restarts}", flush=True)

# Comparison
print("\n" + "=" * 70, flush=True)
print(f"  Analytical: best {max(worsts_t):+.2f}, mean {np.mean(worsts_t):+.2f}, "
      f"min {min(worsts_t):+.2f}, flat {flats_t}/{n_restarts}", flush=True)
print(f"  Surrogate:  best {max(worsts_s):+.2f}, mean {np.mean(worsts_s):+.2f}, "
      f"min {min(worsts_s):+.2f}, flat {flats_s}/{n_restarts}", flush=True)
delta_mean = np.mean(worsts_s) - np.mean(worsts_t)
print(f"  delta mean: {delta_mean:+.2f}", flush=True)
print(f"\n  Per-seed truth worsts:     {[f'{w:+.2f}' for w in worsts_t]}", flush=True)
print(f"  Per-seed surrogate worsts: {[f'{w:+.2f}' for w in worsts_s]}", flush=True)

# Selector envelope note
print("\n" + "=" * 70, flush=True)
print(f"NOTE: n={n} is OUTSIDE R141 selector envelope (validated n in {{31, 51, 71}}).", flush=True)
print(f"For n={n} the worst-case headroom is much smaller (analytical baseline {max(worsts_t):+.2f} dB).", flush=True)
print(f"Selector should be EXTENDED to support smaller apertures, OR documented as min n=31.", flush=True)
print("=" * 70, flush=True)
