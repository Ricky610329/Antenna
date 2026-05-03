"""Round 149 — Surrogate-loop cross-config validation.

R148 showed surrogate-loop robust to 20% weight noise at one config (n=31,
inc=51, 38GHz, w=10). R149 verifies the methodology generalizes across the
selector's recipe space.

Test 4 representative configs from R141 selector with 10% perturbed surrogate:
  A: n=31 inc=51 38GHz w=10 (R119)        — R148 reference
  B: n=51 inc=51 38GHz w=10 (R119)        — different aperture
  C: n=31 inc=51 38GHz w=18 (R129)        — wider cap
  D: n=31 inc=0  28GHz w=10 (R131 rescue) — normal incidence

For each: build perfect warm-start surrogate, perturb 10%, run surrogate-loop
optimization, compare to analytical baseline.

PASS = surrogate-loop mean worst >= analytical mean - 0.5 dB AND flat >= analytical - 1.
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


def build_perturbed_surrogate(sim, n_elem, noise_pct, seed):
    sur = ContinuousWarmStartSurrogate(n_elem).to("cuda:0")
    W_re = sim.pre_calAF[0].real.to(torch.float32).to("cuda:0")
    W_im = sim.pre_calAF[0].imag.to(torch.float32).to("cuda:0")
    torch.manual_seed(seed)
    sigma_re = float(W_re.std()) * (noise_pct / 100.0)
    sigma_im = float(W_im.std()) * (noise_pct / 100.0)
    with torch.no_grad():
        sur.real_lin.weight.copy_(W_re + torch.randn_like(W_re) * sigma_re)
        sur.imag_lin.weight.copy_(W_im + torch.randn_like(W_im) * sigma_im)
    return sur


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


def run_optimization(forward_fn, sim_for_eval, n_elem, main_lo, main_hi, mean_w, ripple_w):
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n_elem, n_elem, device="cuda:0") * 2.0)
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


configs = [
    {"label": "A: n=31 inc=51 38GHz w=10 (R119)", "n": 31, "inc": 51, "freq": 38e9, "w": 10, "rw": 2.0, "lam": 1.0},
    {"label": "B: n=51 inc=51 38GHz w=10 (R119)", "n": 51, "inc": 51, "freq": 38e9, "w": 10, "rw": 2.0, "lam": 1.0},
    {"label": "C: n=31 inc=51 38GHz w=18 (R129)", "n": 31, "inc": 51, "freq": 38e9, "w": 18, "rw": 3.0, "lam": 1.0},
    {"label": "D: n=31 inc=0  28GHz w=10 (R131)", "n": 31, "inc": 0,  "freq": 28e9, "w": 10, "rw": 2.0, "lam": 0.3},
]

NOISE_PCT = 10  # representative real-world noise level

print("=" * 100, flush=True)
print(f"R149 -- Surrogate-loop cross-config validation at noise={NOISE_PCT}%", flush=True)
print(f"  4 selector configs, 5 seeds each, joint early-stop", flush=True)
print("=" * 100, flush=True)

all_results = {}
for cfg in configs:
    print(f"\n{'='*70}", flush=True)
    print(f"  {cfg['label']}", flush=True)
    print(f"{'='*70}", flush=True)

    sim = RISSimulator(element_num=cfg["n"], freq_hz=cfg["freq"], inc_theta_deg=cfg["inc"])
    main_lo, main_hi = steer_to_indices(0, cfg["w"])

    print(f"  [analytical baseline]", flush=True)
    t0 = time.time()
    truth_r = run_optimization(sim, sim, cfg["n"], main_lo, main_hi, cfg["lam"], cfg["rw"])
    ts = summarize(truth_r)
    print(f"    {time.time()-t0:.1f}s -> best {ts['best']:+.2f}, mean {ts['mean']:+.2f}, "
          f"flat {ts['flat']}/{ts['n']}", flush=True)

    print(f"  [perturbed surrogate {NOISE_PCT}%]", flush=True)
    sur = build_perturbed_surrogate(sim, cfg["n"], NOISE_PCT, seed=2024)
    t0 = time.time()
    sur_r = run_optimization(sur, sim, cfg["n"], main_lo, main_hi, cfg["lam"], cfg["rw"])
    ss = summarize(sur_r)
    print(f"    {time.time()-t0:.1f}s -> best {ss['best']:+.2f}, mean {ss['mean']:+.2f}, "
          f"flat {ss['flat']}/{ss['n']}", flush=True)

    delta_mean = ss["mean"] - ts["mean"]
    delta_flat = ss["flat"] - ts["flat"]
    passed = (delta_mean >= -0.5) and (delta_flat >= -1)
    verdict = "PASS" if passed else "FAIL"
    print(f"  Delta mean = {delta_mean:+.2f}, delta flat = {delta_flat:+d}  -> {verdict}",
          flush=True)
    all_results[cfg["label"]] = {"truth": ts, "surrogate": ss, "passed": passed,
                                 "delta_mean": delta_mean, "delta_flat": delta_flat}

# Summary table
print("\n" + "=" * 90, flush=True)
print("SUMMARY:", flush=True)
print("=" * 90, flush=True)
print(f"  {'config':<35} | {'mean truth':>10} | {'mean surr':>9} | {'delta':>6} | {'flat T/S':>9} | {'verdict':>7}",
      flush=True)
print(f"  {'-'*88}", flush=True)
all_pass = True
for label, r in all_results.items():
    short = label[:33]
    ts, ss = r["truth"], r["surrogate"]
    v = "PASS" if r["passed"] else "FAIL"
    if not r["passed"]: all_pass = False
    print(f"  {short:<35} | {ts['mean']:>+10.2f} | {ss['mean']:>+9.2f} | "
          f"{r['delta_mean']:>+6.2f} | {ts['flat']}/{ss['flat']:>7} | {v:>7}", flush=True)

print("\n" + "=" * 70, flush=True)
print(f"OVERALL: {'ALL CONFIGS PASS' if all_pass else 'SOME CONFIGS FAIL'}", flush=True)
print("=" * 70, flush=True)
