"""Round 144 — Surrogate trained on optimization-trajectory snapshots.

R142/R143 tried random binary patterns -> surrogate stuck at predicting mean
or heavily overfit. Diagnosis: distribution mismatch — random patterns are
mostly bad solutions; the surrogate needs accuracy on patterns the optimizer
visits during training.

R144 fixes the data:
  1. Run R141's optimization 15 seeds * 1500 GD steps with R119 recipe
  2. Snapshot the params every 30 steps -> ~50 snapshots/seed -> 750 patterns
  3. Quantize each snapshot to 1-bit and query analytical sim
  4. Train physics-aware surrogate on this trajectory-distribution data
  5. Evaluate on:
     - In-distribution: held-out trajectory snapshots
     - Out-of-distribution: random patterns (R142/R143's distribution)
"""
import sys
sys.path.insert(0, "script")
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 31; freq = 38e9; inc = 51.0
gd_steps = 1500
n_train_seeds = 12  # for trajectory data
n_held_out_seeds = 3  # for in-distribution test
snapshot_every = 30


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def collect_trajectory(sim, main_lo, main_hi, seed, mean_w=1.0, ripple_w=2.0):
    """Run R119 recipe for `gd_steps`; snapshot quantized binary every K steps.

    Returns (binary_patterns, responses) tensors.
    """
    torch.manual_seed(seed)
    params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
    opt = torch.optim.Adam([params], lr=0.05)

    binaries = []; responses = []
    for step in range(gd_steps):
        opt.zero_grad()
        resp = sim(params)["response"]
        main = resp[main_lo:main_hi]
        side = torch.cat([resp[:main_lo], resp[main_hi:]])
        mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
        loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
        loss.backward()
        opt.step()

        if (step + 1) % snapshot_every == 0:
            with torch.no_grad():
                phase = (params * torch.pi) % (2 * torch.pi)
                binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
                r = sim(binary)["response"]
            binaries.append(binary.detach().clone())
            responses.append(r.detach().clone())

    return torch.stack(binaries), torch.stack(responses)


class PhysicsRISsurrogate(nn.Module):
    """(1 - 2*x) -> linear (real, imag) -> |.|^2 -> log10."""
    def __init__(self, n_elem, n_angles=361):
        super().__init__()
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles)
        self.db_bias = nn.Parameter(torch.zeros(n_angles))

    def forward(self, x):
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)
        x = (1.0 - 2.0 * x).flatten(1)
        re = self.real_lin(x); im = self.imag_lin(x)
        power = re * re + im * im + 1e-8
        out = 10.0 * torch.log10(power) + self.db_bias
        return out.squeeze(0) if single else out


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
main_lo, main_hi = steer_to_indices(0, 10)

print("=" * 100, flush=True)
print(f"R144 -- Trajectory-distribution surrogate at n={n}, inc={inc}, freq={freq/1e9}GHz", flush=True)
print(f"  {n_train_seeds} train seeds + {n_held_out_seeds} test seeds, snapshot every {snapshot_every} steps", flush=True)
print("=" * 100, flush=True)

print(f"\n[1/4] Collecting trajectory data ({n_train_seeds + n_held_out_seeds} runs x {gd_steps} steps)...", flush=True)
t0 = time.time()
all_binaries, all_responses = [], []
for seed in range(n_train_seeds + n_held_out_seeds):
    b, r = collect_trajectory(sim, main_lo, main_hi, seed)
    all_binaries.append(b)
    all_responses.append(r)
    print(f"  seed {seed}: {len(b)} snapshots collected", flush=True)

n_per_seed = len(all_binaries[0])
train_binaries = torch.cat(all_binaries[:n_train_seeds])
train_responses = torch.cat(all_responses[:n_train_seeds])
test_binaries = torch.cat(all_binaries[n_train_seeds:])
test_responses = torch.cat(all_responses[n_train_seeds:])

# OOD test: random patterns (R142/R143 distribution)
torch.manual_seed(2024)
ood_binaries = torch.bernoulli(torch.full((300, n, n), 0.5)).to("cuda:0")
ood_responses = torch.stack([sim(ood_binaries[i])["response"] for i in range(300)])

print(f"  done in {time.time()-t0:.1f}s.", flush=True)
print(f"  train: {train_binaries.shape}, test (in-dist): {test_binaries.shape}, OOD random: {ood_binaries.shape}", flush=True)

print(f"\n[2/4] Building physics-aware surrogate...", flush=True)
model = PhysicsRISsurrogate(n).to("cuda:0")
print(f"  params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

print(f"\n[3/4] Training (300 epochs, batch=128, Adam lr=3e-4, early stop on in-dist)...", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()
batch = 128

best, best_state, stale, patience = float("inf"), None, 0, 30
t0 = time.time()
for epoch in range(300):
    perm = torch.randperm(len(train_binaries))
    model.train()
    epoch_loss = 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i+batch]
        pred = model(train_binaries[idx])
        loss = loss_fn(pred, train_responses[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= len(perm)

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(test_binaries), test_responses).item()
        ood_loss = loss_fn(model(ood_binaries), ood_responses).item()

    if test_loss < best - 1e-3:
        best = test_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:>4d} | train MSE {epoch_loss:.3f} | test (in-dist) MSE {test_loss:.3f}"
              f" | OOD random MSE {ood_loss:.3f} | best {best:.3f}", flush=True)

    if stale >= patience:
        print(f"  Early stop at epoch {epoch+1}", flush=True)
        break

print(f"  total training: {time.time()-t0:.1f}s, best in-dist test MSE = {best:.3f}", flush=True)
model.load_state_dict(best_state)

print(f"\n[4/4] Final fit quality:", flush=True)
model.eval()
for tag, x, y in [("in-dist (trajectory)", test_binaries, test_responses),
                  ("OOD (random)", ood_binaries, ood_responses)]:
    with torch.no_grad():
        yp = model(x).cpu().numpy()
    yt = y.cpu().numpy()
    abs_err = np.abs(yp - yt)
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    main_err = np.abs(yp[:, main_lo:main_hi] - yt[:, main_lo:main_hi])
    print(f"\n  {tag}:", flush=True)
    print(f"    R^2:                        {r2:.4f}", flush=True)
    print(f"    mean |abs err| (all bins):  {float(np.mean(abs_err)):.3f} dB", flush=True)
    print(f"    max  |abs err| (any bin):   {float(np.max(abs_err)):.3f} dB", flush=True)
    print(f"    mean main-region |abs err|: {float(np.mean(main_err)):.3f} dB", flush=True)

torch.save(model.state_dict(), "outputs/r144_trajectory_surrogate.pt")
print(f"\n  Saved: outputs/r144_trajectory_surrogate.pt", flush=True)
