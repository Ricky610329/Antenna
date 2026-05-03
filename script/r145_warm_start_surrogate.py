"""Round 145 — Warm-start surrogate from analytical pre_calAF coefficients.

R142/R143/R144 all failed because surrogate cold-starts from random init and
can't find the right weight manifold. R145 warm-starts using the analytical
sim's pre-computed array factor coefficients directly.

Sim core:
  af = pre_calAF * exp(j*phase)         # complex (361, n*n)
  AF = |sum af over elements|           # (361,)
  response = 20*log10(AF / max(AF))     # normalized

For binary phase 0/pi: exp(j*phase) = (1 - 2*pattern). So:
  af = pre_calAF * (1 - 2*pattern)
  F_real = sum  Re(pre_calAF) * (1 - 2*pattern)
  F_imag = sum  Im(pre_calAF) * (1 - 2*pattern)

This means the analytical real_lin/imag_lin weights ARE Re/Im(pre_calAF[θ, *]).
We can extract them directly and validate the architecture is sufficient.

R145 flow:
  1. Extract pre_calAF[broadside_phi=90deg, *, *] -> (361, n*n) complex
  2. Initialize PhysicsRISsurrogate with these weights
  3. Implement the per-pattern normalization (subtract max) in forward
  4. Validate: untrained warm-start should already give R^2 ~ 1
  5. (If yes) light finetuning on trajectory data
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


class WarmStartSurrogate(nn.Module):
    """Physics-aware with explicit normalization. Init from pre_calAF."""
    def __init__(self, n_elem, n_angles=361):
        super().__init__()
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)

    def forward(self, x):
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)
        # binary 0/1 -> amplitude +1/-1
        x = (1.0 - 2.0 * x).flatten(1)
        re = self.real_lin(x)            # (B, 361)
        im = self.imag_lin(x)
        amp = torch.sqrt(re * re + im * im + 1e-12)  # (B, 361)
        peak = amp.max(dim=1, keepdim=True).values
        out = 20.0 * torch.log10(amp / (peak + 1e-12) + 1e-12)
        return out.squeeze(0) if single else out


def extract_analytical_weights(sim):
    """Extract Re/Im of pre_calAF at broadside (phi=90 deg row)."""
    pre_calAF = sim.pre_calAF.detach()  # complex (n_phi, n_theta, n_elem*n_elem)
    # phi grid is np.arange(0, 361, 2). Broadside phi=90 -> index = 45.
    phi_idx = 45
    af_slice = pre_calAF[phi_idx]   # (361, n*n) complex
    return af_slice.real.to(torch.float32), af_slice.imag.to(torch.float32)


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)

print("=" * 100, flush=True)
print(f"R145 -- Warm-start surrogate from analytical pre_calAF coefficients", flush=True)
print(f"  n={n}, inc={inc}, freq={freq/1e9}GHz, phi=90 deg slice", flush=True)
print("=" * 100, flush=True)

print(f"\n[1/4] Extracting analytical weights from sim.pre_calAF...", flush=True)
W_re, W_im = extract_analytical_weights(sim)
print(f"  W_re shape: {W_re.shape}, dtype: {W_re.dtype}", flush=True)
print(f"  W_im shape: {W_im.shape}", flush=True)

print(f"\n[2/4] Building warm-start surrogate...", flush=True)
model = WarmStartSurrogate(n).to("cuda:0")
# nn.Linear stores weights as (out_features, in_features). Our W_re/W_im are
# (n_angles, n*n) = (out, in). Direct assignment.
with torch.no_grad():
    model.real_lin.weight.copy_(W_re.to("cuda:0"))
    model.imag_lin.weight.copy_(W_im.to("cuda:0"))
print(f"  Initialized real_lin/imag_lin from pre_calAF", flush=True)

print(f"\n[3/4] Validate untrained warm-start fit...", flush=True)
torch.manual_seed(42)
n_test = 200
test_binaries = torch.bernoulli(torch.full((n_test, n, n), 0.5)).to("cuda:0")
test_responses = torch.stack([sim(test_binaries[i])["response"] for i in range(n_test)])

model.eval()
with torch.no_grad():
    pred = model(test_binaries).cpu().numpy()
true = test_responses.cpu().numpy()
abs_err = np.abs(pred - true)
ss_res = float(np.sum((pred - true) ** 2))
ss_tot = float(np.sum((true - true.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
main_err = np.abs(pred[:, 170:190] - true[:, 170:190])
print(f"  Untrained warm-start fit on 200 random patterns:", flush=True)
print(f"    R^2:                        {r2:.4f}", flush=True)
print(f"    mean |abs err| (all bins):  {float(np.mean(abs_err)):.4f} dB", flush=True)
print(f"    max  |abs err| (any bin):   {float(np.max(abs_err)):.4f} dB", flush=True)
print(f"    mean main-region |abs err|: {float(np.mean(main_err)):.4f} dB", flush=True)

if r2 > 0.99:
    print(f"\n  *** WARM-START WORKS: surrogate matches analytical sim ***", flush=True)
    print(f"      Architecture is sufficient — only cold-start optimization was the problem.", flush=True)
elif r2 > 0.80:
    print(f"\n  Partial match — possible phi axis indexing offset, investigate.", flush=True)
else:
    print(f"\n  Match poor — architecture may be missing something (e.g., normalization detail).", flush=True)

print(f"\n[4/4] Quick fine-tune (50 epochs) on small trajectory dataset to test stability...", flush=True)
# Generate small trajectory dataset for fine-tune
def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

torch.manual_seed(0)
params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
opt = torch.optim.Adam([params], lr=0.05)
ft_binaries, ft_responses = [], []
for step in range(1500):
    opt.zero_grad()
    resp = sim(params)["response"]
    main = resp[170:190]; side = torch.cat([resp[:170], resp[190:]])
    mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
    loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
    loss.backward(); opt.step()
    if (step + 1) % 50 == 0:
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            r = sim(binary)["response"]
        ft_binaries.append(binary.detach().clone())
        ft_responses.append(r.detach().clone())

ft_binaries = torch.stack(ft_binaries)
ft_responses = torch.stack(ft_responses)
print(f"  Fine-tune set: {ft_binaries.shape}", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-5)  # very small lr to preserve weights
loss_fn = nn.MSELoss()
for ep in range(50):
    model.train()
    pred = model(ft_binaries)
    l = loss_fn(pred, ft_responses)
    opt.zero_grad(); l.backward(); opt.step()
    if (ep + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            yp = model(test_binaries).cpu().numpy()
            yt = test_responses.cpu().numpy()
            err = np.mean(np.abs(yp - yt))
            ss_res = float(np.sum((yp - yt) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot
        print(f"  ft epoch {ep+1}: train MSE {l.item():.4f}, test R^2 {r2:.4f}, mean err {err:.3f} dB", flush=True)

torch.save(model.state_dict(), "outputs/r145_warmstart_surrogate.pt")
print(f"\n  Saved: outputs/r145_warmstart_surrogate.pt", flush=True)
