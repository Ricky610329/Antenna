"""Round 147 — Surrogate-in-the-loop optimization with warm-started surrogate.

R146 verified warm-start matches analytical sim exactly (R^2=1.0).
But R146 used (1-2x) transform which only handles BINARY input. GD operates
on continuous params, so the surrogate needs continuous-aware forward.

Generalized: for continuous x in [0, 1], phase = x*pi, so amp = exp(j*phase) =
cos(phase) + i*sin(phase). Then complex multiply with pre_calAF:
  af = pre_calAF * exp(j*phase)
     = (W_re + i*W_im) * (cos(phase) + i*sin(phase))
     = (W_re*cos - W_im*sin) + i*(W_re*sin + W_im*cos)

In linear-layer terms (per output angle j, summing over element k):
  F_real(j) = sum_k W_re[j,k]*cos(phase[k]) - W_im[j,k]*sin(phase[k])
  F_imag(j) = sum_k W_re[j,k]*sin(phase[k]) + W_im[j,k]*cos(phase[k])

Then |F|, log10, normalize.

R147 flow:
  1. Build continuous-aware surrogate, warm-start from pre_calAF[0].
  2. Verify it matches analytical sim on BOTH binary AND continuous inputs.
  3. Run R141 R119 recipe with surrogate as forward pass (5 seeds, 1500 GD).
  4. Quantize result to binary, evaluate on analytical truth.
  5. Compare to R141 analytical-baseline numbers.

Acceptance: surrogate-loop produces patterns whose analytical-truth metrics
are equivalent to analytical-baseline (worst > +2.5 dB, flat-top OK).
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
    """Generalized warm-start surrogate. Accepts continuous x in [0, 2] (phase = x*pi)."""
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
        # Match sim's column-major flatten via transpose
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
    """Run R141 R119 recipe (joint early-stop) using forward_fn as forward pass.

    forward_fn must accept params and return dict with 'response' key.
    sim_for_eval is used to evaluate quantized binary at end (truth).
    """
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        best_joint_worst = -1e9
        best_joint_state = None

        for step in range(gd_steps):
            opt.zero_grad()
            resp = forward_fn(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
            loss.backward()
            opt.step()

            if (step + 1) % 50 == 0:
                # Joint early-stop: evaluate quantized on TRUTH (analytical sim)
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
    smeans = [r["side_mean"] for r in results]
    flats = sum(1 for r in results if r["flat_top"])
    return {
        "best": max(worsts),
        "mean": float(np.mean(worsts)),
        "min": min(worsts),
        "smean_best": min(smeans),
        "flat": flats,
        "n": len(results),
    }


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
main_lo, main_hi = steer_to_indices(0, 10)

print("=" * 100, flush=True)
print(f"R147 -- Surrogate-in-the-loop optimization at n={n}, inc={inc}, 38GHz, w=10", flush=True)
print(f"  Continuous-aware warm-started surrogate, joint early-stop", flush=True)
print("=" * 100, flush=True)

# 1. Build + warm-start continuous surrogate
print("\n[1/4] Building continuous-aware warm-start surrogate...", flush=True)
surrogate = ContinuousWarmStartSurrogate(n).to("cuda:0")
W_re = sim.pre_calAF[0].real.to(torch.float32).to("cuda:0")
W_im = sim.pre_calAF[0].imag.to(torch.float32).to("cuda:0")
with torch.no_grad():
    surrogate.real_lin.weight.copy_(W_re)
    surrogate.imag_lin.weight.copy_(W_im)

# 2. Verify continuous AND binary match
print("\n[2/4] Verifying continuous surrogate matches analytical sim...", flush=True)
torch.manual_seed(123)
binary_test = torch.bernoulli(torch.full((50, n, n), 0.5)).to("cuda:0")
continuous_test = torch.rand(50, n, n, device="cuda:0") * 2.0
with torch.no_grad():
    bin_pred = surrogate(binary_test)["response"].cpu().numpy()
    bin_true = torch.stack([sim(binary_test[i])["response"] for i in range(50)]).cpu().numpy()
    cont_pred = surrogate(continuous_test)["response"].cpu().numpy()
    cont_true = torch.stack([sim(continuous_test[i])["response"] for i in range(50)]).cpu().numpy()

bin_err = np.mean(np.abs(bin_pred - bin_true))
cont_err = np.mean(np.abs(cont_pred - cont_true))
print(f"  binary input  mean abs err: {bin_err:.6f} dB", flush=True)
print(f"  continuous input mean abs err: {cont_err:.6f} dB", flush=True)
if bin_err > 0.01 or cont_err > 0.01:
    print(f"  WARNING: surrogate doesn't match sim on continuous input — abort.", flush=True)
    sys.exit(1)

# 3. Surrogate-loop optimization
print("\n[3/4] Running surrogate-loop optimization (5 seeds x 1500 steps)...", flush=True)
t0 = time.time()
surr_results = run_optimization(surrogate, sim, main_lo, main_hi)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

# 4. Analytical baseline
print("\n[4/4] Running analytical-truth baseline (5 seeds x 1500 steps)...", flush=True)
t0 = time.time()
truth_results = run_optimization(sim, sim, main_lo, main_hi)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

# Compare
ss = summarize(surr_results)
ts = summarize(truth_results)

print("\n" + "=" * 70, flush=True)
print("Comparison: surrogate-loop vs analytical-baseline", flush=True)
print("=" * 70, flush=True)
print(f"  {'metric':<10} | {'surrogate':>10} | {'analytical':>11} | {'delta':>8}", flush=True)
print(f"  {'-'*45}", flush=True)
for key in ["best", "mean", "min", "smean_best"]:
    delta = ss[key] - ts[key]
    print(f"  {key:<10} | {ss[key]:>+10.2f} | {ts[key]:>+11.2f} | {delta:>+8.2f}", flush=True)
print(f"  {'flat':<10} | {ss['flat']}/{ss['n']:>8} | {ts['flat']}/{ts['n']:>9} |", flush=True)

surr_worsts = " ".join("{:+.2f}".format(r["worst"]) for r in surr_results)
truth_worsts = " ".join("{:+.2f}".format(r["worst"]) for r in truth_results)
print(f"\n  Surrogate per-seed worsts: [{surr_worsts}]", flush=True)
print(f"  Analytical per-seed worsts: [{truth_worsts}]", flush=True)

# Verdict
if abs(ss["mean"] - ts["mean"]) < 0.5 and ss["flat"] >= ts["flat"] - 1:
    print(f"\n  VERDICT: surrogate-loop matches analytical baseline.", flush=True)
    print(f"           Methodology transfers to surrogate gradient.", flush=True)
else:
    print(f"\n  VERDICT: surrogate-loop diverges from analytical baseline.", flush=True)
    print(f"           Investigate gradient quality in R148.", flush=True)
