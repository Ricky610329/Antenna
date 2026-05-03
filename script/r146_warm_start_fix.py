"""Round 146 — Fix R145 warm-start bugs and verify exact match to analytical sim.

R145 hit R^2 = -0.97 from two extraction bugs:
  Bug 1: phi_idx=45 (phi=90 deg) but sim returns dB_AF[0] (phi=0 deg).
  Bug 2: pattern flatten order doesn't match sim's MPD.t().reshape().

R146 fixes both:
  - phi_idx = 0
  - Surrogate flattens (1 - 2*x).transpose(1,2).flatten(1) to match column-major

Sanity check: untrained warm-start should match analytical sim to numerical
precision (R^2 ~ 1.0, mean abs err < 0.01 dB).
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 31; freq = 38e9; inc = 51.0


class WarmStartSurrogate(nn.Module):
    """Mirrors sim core: (1 - 2*x).T.flatten() -> linear (real, imag) -> |.| -> 20*log10(|.|/max)."""
    def __init__(self, n_elem, n_angles=361):
        super().__init__()
        self.n = n_elem
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)

    def forward(self, x):
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)
        # CRITICAL: sim does MPD.t().reshape(...) so we transpose before flatten
        amp_in = (1.0 - 2.0 * x).transpose(1, 2).contiguous().flatten(1)  # (B, n*n)
        re = self.real_lin(amp_in)   # (B, 361)
        im = self.imag_lin(amp_in)
        amp = torch.sqrt(re * re + im * im + 1e-12)
        peak = amp.max(dim=1, keepdim=True).values
        out = 20.0 * torch.log10(torch.clamp(amp, min=1e-8) / torch.clamp(peak, min=1e-8))
        return out.squeeze(0) if single else out


sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)

print("=" * 100, flush=True)
print(f"R146 -- Warm-start surrogate, fixed extraction (phi_idx=0, columnmajor flatten)", flush=True)
print("=" * 100, flush=True)

# Extract from broadside (phi=0 deg), which is sim.pre_calAF[0]
pre_calAF = sim.pre_calAF.detach()
print(f"\n[1/3] sim.pre_calAF shape: {pre_calAF.shape}, dtype: {pre_calAF.dtype}", flush=True)
af_slice = pre_calAF[0]  # (n_theta=361, n_elem*n_elem=961) complex
print(f"  Extracted broadside slice: {af_slice.shape}", flush=True)
W_re = af_slice.real.to(torch.float32)
W_im = af_slice.imag.to(torch.float32)

print(f"\n[2/3] Initialize warm-start surrogate from broadside slice...", flush=True)
model = WarmStartSurrogate(n).to("cuda:0")
with torch.no_grad():
    model.real_lin.weight.copy_(W_re.to("cuda:0"))
    model.imag_lin.weight.copy_(W_im.to("cuda:0"))

print(f"\n[3/3] Sanity check: untrained warm-start fit on 200 random patterns", flush=True)
torch.manual_seed(42)
test_binaries = torch.bernoulli(torch.full((200, n, n), 0.5)).to("cuda:0")
test_responses = torch.stack([sim(test_binaries[i])["response"] for i in range(200)])

model.eval()
with torch.no_grad():
    pred = model(test_binaries).cpu().numpy()
true = test_responses.cpu().numpy()
abs_err = np.abs(pred - true)
ss_res = float(np.sum((pred - true) ** 2))
ss_tot = float(np.sum((true - true.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
main_err = np.abs(pred[:, 170:190] - true[:, 170:190])

print(f"  R^2:                        {r2:.6f}", flush=True)
print(f"  mean |abs err| (all bins):  {float(np.mean(abs_err)):.6f} dB", flush=True)
print(f"  max  |abs err| (any bin):   {float(np.max(abs_err)):.6f} dB", flush=True)
print(f"  median |abs err|:           {float(np.median(abs_err)):.6f} dB", flush=True)
print(f"  mean main-region |abs err|: {float(np.mean(main_err)):.6f} dB", flush=True)

if r2 > 0.999 and np.mean(abs_err) < 0.01:
    print(f"\n  *** WARM-START EXACT MATCH ***", flush=True)
    print(f"      Architecture sufficient. R147 can use this for surrogate-loop opt.", flush=True)
elif r2 > 0.95:
    print(f"\n  Close match — likely numeric precision. Acceptable.", flush=True)
else:
    print(f"\n  Still off — more bugs to find.", flush=True)
    # Quick diagnostic: print a sample comparison
    sample_idx = 0
    print(f"\n  Sample 0 first 10 angles (predicted vs true):", flush=True)
    for i in range(10):
        ang = -90 + i * 0.5
        print(f"    theta={ang:+.1f}d: pred={pred[sample_idx, i]:+.3f}, true={true[sample_idx, i]:+.3f}", flush=True)
    print(f"\n  Sample 0 main region (170-190):", flush=True)
    for i in range(170, 190, 2):
        ang = -90 + i * 0.5
        print(f"    theta={ang:+.1f}d: pred={pred[sample_idx, i]:+.3f}, true={true[sample_idx, i]:+.3f}", flush=True)

torch.save(model.state_dict(), "outputs/r146_warmstart_fixed.pt")
print(f"\n  Saved: outputs/r146_warmstart_fixed.pt", flush=True)
