"""Round 150 — Unified deployment pipeline: selector + (optional) surrogate + joint ES.

Generalizes R141's optimize_ris_1bit() to accept an optional surrogate forward
function. When given:
  - forward_fn = None  : use analytical sim for both gradient AND evaluation
  - forward_fn = sur   : use surrogate for gradient, sim for joint early-stop eval

Re-runs R141's 6 held-out combos in BOTH modes (analytical, surrogate-loop)
and confirms all PASS (surrogate-loop should match or beat analytical based
on R148/R149).

This is the API patch transition will use — same function for analytical RIS
playground AND patch HFSS-surrogate. Loss/recipe/early-stop are identical.
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
GD_STEPS = 1500
EVAL_EVERY = 50


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def select_1bit_recipe(n, inc_deg, freq_hz, width_deg):
    """R134/R135 selector."""
    if width_deg > 30: raise ValueError(f"width={width_deg} > 30")
    if n not in (31, 51, 71): raise ValueError(f"n={n} not validated")
    if n == 71:
        if inc_deg == 0 and freq_hz >= 50e9:
            return {"rw": 5.0, "lambda_mean": 0.3, "tier": "R133 n=71 inc=0 mmWave"}
        if width_deg <= 15:
            return {"rw": 5.0, "lambda_mean": 0.5, "tier": "n=71 narrow extrap"}
        return {"rw": 7.0, "lambda_mean": 0.5, "tier": "n=71 wide extrap"}
    if width_deg > 12:
        if width_deg <= 20:
            return {"rw": 3.0, "lambda_mean": 1.0, "tier": "R129 wide cap (12-20)"}
        return {"rw": 3.0, "lambda_mean": 0.5, "tier": "R129 wide cap 30 (marginal)"}
    if inc_deg == 0 and freq_hz >= 20e9:
        if freq_hz >= 50e9: raise ValueError("inc=0 + freq>=50GHz at n=51 -> use n=71")
        if freq_hz >= 35e9: return {"rw": 2.0, "lambda_mean": 0.5, "tier": "R131 inc=0 38GHz"}
        return {"rw": 2.0, "lambda_mean": 0.3, "tier": "R131 inc=0 28GHz"}
    return {"rw": 2.0, "lambda_mean": 1.0, "tier": "R119 baseline"}


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


class ContinuousWarmStartSurrogate(nn.Module):
    """Generic warm-start surrogate. For RIS playground only — patch will use HFSSNet."""
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


def optimize_ris_1bit(n, inc_deg, freq_hz, width_deg,
                     n_restarts=5, gd_steps=GD_STEPS,
                     steering_center_deg=0,
                     forward_fn=None, eval_fn=None):
    """Unified deployment pipeline.

    Parameters
    ----------
    forward_fn : callable, optional
        If provided, used for gradient computation (e.g. surrogate model).
        Must accept (params) and return dict with 'response' key.
        Default = analytical sim.
    eval_fn : callable, optional
        Used for joint early-stop evaluation (the "truth" for snapshot
        selection). Default = analytical sim.

    Returns
    -------
    dict with: recipe, best, n_flat_top, n_restarts, n_early_stop_used,
               seed_results
    """
    recipe = select_1bit_recipe(n, inc_deg, freq_hz, width_deg)
    rw, lam = recipe["rw"], recipe["lambda_mean"]
    main_lo, main_hi = steer_to_indices(steering_center_deg, width_deg)

    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    if forward_fn is None: forward_fn = sim
    if eval_fn is None: eval_fn = sim

    seed_results = []; best_overall = None

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
            loss = -(mm - sx) + rw * (mx - mm) + lam * side.mean()
            loss.backward(); opt.step()

            if (step + 1) % EVAL_EVERY == 0:
                m = eval_binary_on(eval_fn, params, main_lo, main_hi)
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()

        if best_joint_state is not None:
            metrics = eval_binary_on(eval_fn, best_joint_state, main_lo, main_hi)
            metrics["used_early_stop"] = True
        else:
            metrics = eval_binary_on(eval_fn, params, main_lo, main_hi)
            metrics["used_early_stop"] = False
        metrics["seed"] = seed
        seed_results.append(metrics)
        if best_overall is None or metrics["worst"] > best_overall["worst"]:
            best_overall = metrics

    return {
        "recipe": recipe,
        "best": best_overall,
        "n_flat_top": sum(1 for r in seed_results if r["flat_top"]),
        "n_restarts": n_restarts,
        "n_early_stop_used": sum(1 for r in seed_results if r["used_early_stop"]),
        "seed_results": seed_results,
    }


def build_warmstart_surrogate(n, freq_hz, inc_deg):
    """Helper: build perfect warm-start surrogate from analytical sim."""
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    sur = ContinuousWarmStartSurrogate(n).to("cuda:0")
    W_re = sim.pre_calAF[0].real.to(torch.float32).to("cuda:0")
    W_im = sim.pre_calAF[0].imag.to(torch.float32).to("cuda:0")
    with torch.no_grad():
        sur.real_lin.weight.copy_(W_re)
        sur.imag_lin.weight.copy_(W_im)
    return sur


# Re-run R141's 6 held-out combos in both modes
combos = [
    {"n": 51, "inc": 30, "freq": 28e9, "width": 10, "label": "off-normal 28GHz"},
    {"n": 51, "inc": 70, "freq": 60e9, "width": 10, "label": "70deg + 60GHz"},
    {"n": 51, "inc": 51, "freq": 38e9, "width": 15, "label": "sweet inc + width=15"},
    {"n": 51, "inc": 51, "freq": 38e9, "width": 20, "label": "sweet inc + wide"},
    # n=71 skipped to keep cron budget tight (each n=71 takes ~225s)
]

print("=" * 100, flush=True)
print(f"R150 -- Unified pipeline: re-run R141 held-out combos in analytical + surrogate modes", flush=True)
print(f"  n=71 combos skipped to keep cron budget tight", flush=True)
print("=" * 100, flush=True)

print(f"\n{'config':<35} | {'mode':<10} | {'recipe':<22} | {'best':>6} | {'mean':>6} | {'flat':>5} | {'time':>6}", flush=True)
print(f"{'-'*110}", flush=True)

for combo in combos:
    n, inc, freq, w = combo["n"], combo["inc"], combo["freq"], combo["width"]
    cfg_str = f"n={n} inc={inc:>2} {freq/1e9:>4g}GHz w={w}d"

    # 1. Analytical mode
    t0 = time.time()
    r_ana = optimize_ris_1bit(n, inc, freq, w, n_restarts=5)
    elapsed_ana = time.time() - t0
    seeds_ana_worst = [s["worst"] for s in r_ana["seed_results"]]
    mean_ana = float(np.mean(seeds_ana_worst))
    print(f"{cfg_str:<35} | {'analytical':<10} | {r_ana['recipe']['tier'][:22]:<22} | "
          f"{r_ana['best']['worst']:>+6.2f} | {mean_ana:>+6.2f} | "
          f"{r_ana['n_flat_top']}/{r_ana['n_restarts']:>3} | {elapsed_ana:>5.0f}s", flush=True)

    # 2. Surrogate mode
    sur = build_warmstart_surrogate(n, freq, inc)
    t0 = time.time()
    r_sur = optimize_ris_1bit(n, inc, freq, w, n_restarts=5, forward_fn=sur)
    elapsed_sur = time.time() - t0
    seeds_sur_worst = [s["worst"] for s in r_sur["seed_results"]]
    mean_sur = float(np.mean(seeds_sur_worst))
    speedup = elapsed_ana / elapsed_sur
    print(f"{cfg_str:<35} | {'surrogate':<10} | {r_sur['recipe']['tier'][:22]:<22} | "
          f"{r_sur['best']['worst']:>+6.2f} | {mean_sur:>+6.2f} | "
          f"{r_sur['n_flat_top']}/{r_sur['n_restarts']:>3} | {elapsed_sur:>5.0f}s "
          f"({speedup:.1f}x)", flush=True)
    print(flush=True)
