"""Round 148 — Surrogate perturbation robustness for surrogate-in-the-loop.

R147 showed perfect warm-start surrogate produces patterns equivalent to
analytical baseline. But real HFSS surrogates won't be perfect. R148 tests
how much surrogate fit error breaks the methodology.

Approach:
  1. Start from R146 warm-start surrogate (R^2 = 1.0).
  2. Inject Gaussian noise into linear weights at sigma in {5%, 10%, 20%}
     of weight std. This simulates real surrogate fit error.
  3. Measure perturbed surrogate's R^2 vs analytical (fit quality).
  4. Run surrogate-loop optimization with perturbed surrogate.
  5. Evaluate resulting patterns on analytical TRUTH (the real measurement).

Critical question: at what noise level does the methodology break?
- Acceptance criterion: mean worst stays >= 0, flat-top stays >= 4/5.
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
n = 31; freq = 38e9; inc = 51.0
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


def eval_binary_on_sim(params, sim, main_lo, main_hi):
    with torch.no_grad():
        phase = (params * torch.pi) % (2 * torch.pi)
        binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
        resp = sim(binary)["response"].cpu().numpy()
    main_arr = resp[main_lo:main_hi]; side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    return {
        "worst": float(main_arr.min() - side_arr.max()),
        "side_mean": float(side_arr.mean()),
        "ripple": float(main_arr.max() - main_arr.min()),
        "flat_top": int(np.sum(main_arr < -3)) == 0,
    }


def run_optimization(forward_fn, sim_for_eval, main_lo, main_hi):
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
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
            loss.backward(); opt.step()
            if (step + 1) % 50 == 0:
                m = eval_binary_on_sim(params, sim_for_eval, main_lo, main_hi)
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()
        if best_joint_state is not None:
            metrics = eval_binary_on_sim(best_joint_state, sim_for_eval, main_lo, main_hi)
        else:
            metrics = eval_binary_on_sim(params, sim_for_eval, main_lo, main_hi)
        metrics["seed"] = seed
        seed_results.append(metrics)
    return seed_results


def summarize(results):
    worsts = [r["worst"] for r in results]
    flats = sum(1 for r in results if r["flat_top"])
    return {
        "best": max(worsts),
        "mean": float(np.mean(worsts)),
        "min": min(worsts),
        "flat": flats,
        "n": len(results),
    }


def make_perturbed_surrogate(W_re_clean, W_im_clean, noise_pct, seed):
    """Clone surrogate, add Gaussian noise to weights."""
    torch.manual_seed(seed)
    sur = ContinuousWarmStartSurrogate(n).to("cuda:0")
    sigma_re = float(W_re_clean.std()) * (noise_pct / 100.0)
    sigma_im = float(W_im_clean.std()) * (noise_pct / 100.0)
    with torch.no_grad():
        sur.real_lin.weight.copy_(W_re_clean + torch.randn_like(W_re_clean) * sigma_re)
        sur.imag_lin.weight.copy_(W_im_clean + torch.randn_like(W_im_clean) * sigma_im)
    return sur


def measure_surrogate_fit(sur, sim, n_test=200):
    torch.manual_seed(123)
    test_b = torch.bernoulli(torch.full((n_test, n, n), 0.5)).to("cuda:0")
    with torch.no_grad():
        pred = sur(test_b)["response"].cpu().numpy()
        true = torch.stack([sim(test_b[i])["response"] for i in range(n_test)]).cpu().numpy()
    abs_err = np.abs(pred - true)
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "mean_abs_err": float(np.mean(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
    }


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
main_lo, main_hi = steer_to_indices(0, 10)

# Clean weights from R146
W_re_clean = sim.pre_calAF[0].real.to(torch.float32).to("cuda:0")
W_im_clean = sim.pre_calAF[0].imag.to(torch.float32).to("cuda:0")

print("=" * 100, flush=True)
print(f"R148 -- Surrogate perturbation robustness test", flush=True)
print(f"  n={n}, inc={inc}, 38GHz, w=10, R119 recipe, 5 seeds joint early-stop", flush=True)
print(f"  Noise levels (% of weight std): 0, 5, 10, 20", flush=True)
print("=" * 100, flush=True)

# First: analytical baseline
print("\n[Baseline] Running analytical-truth optimization (5 seeds)...", flush=True)
t0 = time.time()
truth_results = run_optimization(sim, sim, main_lo, main_hi)
ts = summarize(truth_results)
print(f"  done in {time.time()-t0:.1f}s -> best {ts['best']:+.2f}, mean {ts['mean']:+.2f}, "
      f"min {ts['min']:+.2f}, flat {ts['flat']}/{ts['n']}", flush=True)

# Then: perturbed surrogate at each noise level
results_by_noise = {}
for noise_pct in [0, 5, 10, 20]:
    print(f"\n[Noise {noise_pct}%]", flush=True)
    sur = make_perturbed_surrogate(W_re_clean, W_im_clean, noise_pct, seed=2024)
    fit = measure_surrogate_fit(sur, sim)
    print(f"  Surrogate fit: R^2 = {fit['r2']:.4f}, mean abs err = {fit['mean_abs_err']:.3f} dB, "
          f"max = {fit['max_abs_err']:.3f} dB", flush=True)

    print(f"  Running surrogate-loop optimization ({n_restarts} seeds)...", flush=True)
    t0 = time.time()
    sur_results = run_optimization(sur, sim, main_lo, main_hi)
    ss = summarize(sur_results)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s -> best {ss['best']:+.2f}, mean {ss['mean']:+.2f}, "
          f"min {ss['min']:+.2f}, flat {ss['flat']}/{ss['n']}", flush=True)
    results_by_noise[noise_pct] = {"fit": fit, "summary": ss, "details": sur_results}

# Comparison table
print("\n" + "=" * 80, flush=True)
print(f"  {'Noise':>6} | {'R^2':>8} | {'fit err':>8} | {'best':>7} | {'mean':>7} | {'min':>7} | {'flat':>5}", flush=True)
print(f"  {'-'*70}", flush=True)
print(f"  {'truth':>6} |        - |        - | {ts['best']:>+7.2f} | {ts['mean']:>+7.2f} | "
      f"{ts['min']:>+7.2f} | {ts['flat']}/{ts['n']:>3}", flush=True)
for noise_pct, r in results_by_noise.items():
    f, s = r["fit"], r["summary"]
    delta_mean = s["mean"] - ts["mean"]
    print(f"  {noise_pct:>5}%  | {f['r2']:>8.4f} | {f['mean_abs_err']:>5.3f}dB | {s['best']:>+7.2f} | "
          f"{s['mean']:>+7.2f} | {s['min']:>+7.2f} | {s['flat']}/{s['n']:>3}  (delta_mean {delta_mean:+.2f})",
          flush=True)

# Verdict
print("\n" + "=" * 70, flush=True)
print("Acceptance: mean >= 0 AND flat >= 4 (since 5 seeds)", flush=True)
print("=" * 70, flush=True)
breaking_noise = None
for noise_pct in [0, 5, 10, 20]:
    s = results_by_noise[noise_pct]["summary"]
    if s["mean"] >= 0 and s["flat"] >= 4:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        if breaking_noise is None:
            breaking_noise = noise_pct
    print(f"  noise={noise_pct:>2}%: {verdict}", flush=True)

if breaking_noise is None:
    print(f"\n  All noise levels PASS up to 20%. Methodology is highly robust.", flush=True)
else:
    print(f"\n  Methodology breaks at noise = {breaking_noise}%. Patch HFSS surrogate must fit better than this.", flush=True)
