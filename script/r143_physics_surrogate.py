"""Round 143 — physics-aware surrogate for RIS sim.

R142's CNN surrogate learned only the mean (R^2 ~ 0). Diagnosis: standard
CNN with ReLU can't easily express |F * pattern|^2 followed by log10.

R143 tries an architecture that respects the physics:
  pattern (n*n) -> complex linear (-> 361 complex amplitudes)
  |.|^2 -> 361 powers
  10 * log10(power + eps) -> 361 dB

In real-valued PyTorch: two parallel real linear layers (real, imag),
combine by real^2 + imag^2, log.

Goal: prove that with the right inductive bias, even tiny model learns the
RIS forward pass to high accuracy. This sets up R144 to use it in the loop.
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


def generate_dataset(n_samples, sim, seed=42):
    torch.manual_seed(seed)
    patterns = torch.bernoulli(torch.full((n_samples, n, n), 0.5)).to("cuda:0")
    responses = []
    for i in range(n_samples):
        with torch.no_grad():
            r = sim(patterns[i])["response"]
        responses.append(r)
    return patterns, torch.stack(responses)


class PhysicsRISsurrogate(nn.Module):
    """Physics-aware: pattern -> complex-amp linear -> |.|^2 -> log."""
    def __init__(self, n_elem, n_angles=361):
        super().__init__()
        self.n = n_elem
        # Real and imag parts of the complex amplitude transform
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles, bias=True)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles, bias=True)
        # Per-angle bias for the dB output (learned offset)
        self.db_bias = nn.Parameter(torch.zeros(n_angles))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0); single = True
        else:
            single = False
        # Binary 0/1 -> phase 0/pi -> complex amp +1/-1.
        # Use (1 - 2*x) so the linear layer sees the actual amplitude.
        x = (1.0 - 2.0 * x).flatten(1)  # (B, n*n)
        re = self.real_lin(x)
        im = self.imag_lin(x)
        power = re * re + im * im + 1e-8
        out = 10.0 * torch.log10(power) + self.db_bias
        return out.squeeze(0) if single else out


print("=" * 100, flush=True)
print(f"R143 -- Physics-aware surrogate at n={n}, inc={inc}, freq={freq/1e9}GHz", flush=True)
print("=" * 100, flush=True)

sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)

print("\n[1/4] Generating dataset (10000 train + 1000 test)...", flush=True)
t0 = time.time()
xtr, ytr = generate_dataset(10000, sim, seed=0)
xte, yte = generate_dataset(1000, sim, seed=999)
print(f"  done in {time.time()-t0:.1f}s", flush=True)

print("\n[2/4] Building physics-aware surrogate...", flush=True)
model = PhysicsRISsurrogate(n).to("cuda:0")
n_params = sum(p.numel() for p in model.parameters())
print(f"  PhysicsRISsurrogate: {n_params:,} parameters", flush=True)

print("\n[3/4] Training (300 epochs, batch=256, Adam lr=3e-4, early stop)...", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()
batch = 256

best, best_state, stale, patience = float("inf"), None, 0, 30
t0 = time.time()
for epoch in range(300):
    perm = torch.randperm(len(xtr))
    model.train()
    epoch_loss = 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i+batch]
        pred = model(xtr[idx])
        loss = loss_fn(pred, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= len(perm)

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(xte), yte).item()

    if test_loss < best - 1e-3:
        best = test_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:>4d} | train MSE {epoch_loss:.4f} | test MSE {test_loss:.4f}"
              f" | best {best:.4f} | stale {stale}", flush=True)

    if stale >= patience:
        print(f"  Early stop at epoch {epoch+1}", flush=True)
        break

print(f"  total training: {time.time()-t0:.1f}s, best test MSE = {best:.4f}", flush=True)
model.load_state_dict(best_state)

print("\n[4/4] Final fit quality on test:", flush=True)
model.eval()
with torch.no_grad():
    yp = model(xte).cpu().numpy()
yt = yte.cpu().numpy()
abs_err = np.abs(yp - yt)
ss_res = float(np.sum((yp - yt) ** 2))
ss_tot = float(np.sum((yt - yt.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
main_lo, main_hi = 170, 190
main_err = np.abs(yp[:, main_lo:main_hi] - yt[:, main_lo:main_hi])
print(f"  R^2:                        {r2:.4f}", flush=True)
print(f"  mean |abs err| (all bins):  {float(np.mean(abs_err)):.3f} dB", flush=True)
print(f"  max  |abs err| (any bin):   {float(np.max(abs_err)):.3f} dB", flush=True)
print(f"  mean max err per pattern:   {float(np.mean(np.max(abs_err, axis=1))):.3f} dB", flush=True)
print(f"  mean main-region |abs err|: {float(np.mean(main_err)):.3f} dB", flush=True)

torch.save(model.state_dict(), "outputs/r143_physics_surrogate_n31.pt")
print(f"  Saved: outputs/r143_physics_surrogate_n31.pt", flush=True)
