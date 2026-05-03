"""Round 154 — Multi-frequency in-band beamforming on RIS.

Phase 3 patch transition is blocked on HFSS access. Pivot back to RIS
playground for the most patch-relevant test: optimize a single binary
pattern that works at MULTIPLE frequencies simultaneously (mirroring
patch antenna's bandwidth requirement).

Setup:
  - 3 RIS simulators at f in {36, 38, 40 GHz} (~10% relative bandwidth)
  - Same n=51, inc=51deg, broadside, w=10deg main region
  - Loss = sum over freqs of R119 recipe (worst + ripple + mean)
  - Joint early-stop: minimum worst across freqs > 0 AND all flat-top
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
n = 51
inc = 51.0
w_deg = 10
gd_steps = 1500
n_restarts = 5

freqs_in_band = [36e9, 38e9, 40e9]


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


main_lo, main_hi = steer_to_indices(0, w_deg)


def eval_binary_metrics(resp_np, main_lo, main_hi):
    main = resp_np[main_lo:main_hi]; side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "worst": float(main.min() - side.max()),
        "side_mean": float(side.mean()),
        "ripple": float(main.max() - main.min()),
        "flat_top": int(np.sum(main < -3)) == 0,
    }


def run_optimization(grad_sims, eval_sims, mean_w=1.0, ripple_w=2.0):
    """grad_sims: used for gradient (sum loss). eval_sims: used for joint
    early-stop AND final per-freq reporting."""
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        best_joint_min_worst = -1e9; best_state = None
        for step in range(gd_steps):
            opt.zero_grad()
            total_loss = 0.0
            for sim in grad_sims:
                resp = sim(params)["response"]
                main = resp[main_lo:main_hi]
                side = torch.cat([resp[:main_lo], resp[main_hi:]])
                mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
                loss_freq = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
                total_loss = total_loss + loss_freq
            total_loss.backward()
            opt.step()

            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    phase = (params * torch.pi) % (2 * torch.pi)
                    binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
                    metrics = []
                    for sim in eval_sims:
                        r = sim(binary)["response"].cpu().numpy()
                        metrics.append(eval_binary_metrics(r, main_lo, main_hi))
                all_flat = all(m["flat_top"] for m in metrics)
                min_worst = min(m["worst"] for m in metrics)
                if all_flat and min_worst > best_joint_min_worst:
                    best_joint_min_worst = min_worst
                    best_state = params.detach().clone()

        eval_state = best_state if best_state is not None else params.detach()
        with torch.no_grad():
            phase = (eval_state * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            per_freq = []
            for sim in eval_sims:
                r = sim(binary)["response"].cpu().numpy()
                per_freq.append(eval_binary_metrics(r, main_lo, main_hi))
        seed_results.append({"seed": seed, "per_freq": per_freq})
    return seed_results


def summarize(seed_results, freqs):
    summary = {}
    for f_idx, f in enumerate(freqs):
        worsts = [r["per_freq"][f_idx]["worst"] for r in seed_results]
        flats = sum(1 for r in seed_results if r["per_freq"][f_idx]["flat_top"])
        summary[f] = {
            "best": max(worsts), "mean": float(np.mean(worsts)),
            "min": min(worsts), "flat": flats, "n": len(seed_results),
        }
    return summary


print("=" * 100, flush=True)
print(f"R154 -- Multi-frequency in-band beamforming (n=51, inc=51deg, w=10)", flush=True)
print(f"  in-band freqs: {[f/1e9 for f in freqs_in_band]} GHz (~10% rel BW around 38GHz)", flush=True)
print("=" * 100, flush=True)

sims_inband = [RISSimulator(element_num=n, freq_hz=f, inc_theta_deg=inc) for f in freqs_in_band]
sim_38_only = sims_inband[1]

print(f"\n[1/2] Single-freq baseline: gradient @ 38GHz, eval at all 3 freqs (5 seeds)", flush=True)
t0 = time.time()
single_results = run_optimization(grad_sims=[sim_38_only], eval_sims=sims_inband)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

print(f"\n[2/2] Multi-freq joint: gradient sums over 3 freqs, eval at all 3 (5 seeds)", flush=True)
t0 = time.time()
joint_results = run_optimization(grad_sims=sims_inband, eval_sims=sims_inband)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

print("\n" + "=" * 90, flush=True)
print(f"  {'mode':<14} | {'freq':>7} | {'best':>7} | {'mean':>7} | {'min':>7} | {'flat':>5}", flush=True)
print(f"  {'-'*65}", flush=True)
for label, results in [("single-freq", single_results), ("multi-freq", joint_results)]:
    s = summarize(results, freqs_in_band)
    for f in freqs_in_band:
        m = s[f]
        print(f"  {label:<14} | {f/1e9:>5.0f}GHz | {m['best']:>+7.2f} | {m['mean']:>+7.2f} | "
              f"{m['min']:>+7.2f} | {m['flat']}/{m['n']}", flush=True)
    print(flush=True)

# Verdict
print("=" * 70, flush=True)
single_s = summarize(single_results, freqs_in_band)
joint_s = summarize(joint_results, freqs_in_band)
single_off = [single_s[f]["mean"] for f in freqs_in_band if f != 38e9]
joint_off = [joint_s[f]["mean"] for f in freqs_in_band if f != 38e9]
print(f"  single-freq off-band mean worst: {np.mean(single_off):+.2f}", flush=True)
print(f"  multi-freq off-band mean worst:  {np.mean(joint_off):+.2f}", flush=True)
print(f"  multi-freq @ 38GHz mean (cost):  {joint_s[38e9]['mean']:+.2f}", flush=True)
print(f"  single-freq @ 38GHz mean (ref):  {single_s[38e9]['mean']:+.2f}", flush=True)
print(f"  bandwidth gain: off-band mean improves "
      f"{np.mean(joint_off) - np.mean(single_off):+.2f} dB by jointly optimizing", flush=True)
print("=" * 70, flush=True)
