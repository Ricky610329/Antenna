"""驗證 bit-flip invariance: 全 0/1 翻轉後 |E|² 不變。"""

import numpy as np
import torch

from antenna.ris import RISSimulator
from antenna.utils.config import config


config.device = "cuda:0"
sim = RISSimulator(element_num=21, freq_hz=38e9, inc_theta_deg=51.0)

# Random pattern
torch.manual_seed(0)
pat = (torch.rand(21, 21, device="cuda:0") > 0.5).float()
flipped = 1.0 - pat

with torch.no_grad():
    r1 = sim(pat)["response"].cpu().numpy()
    r2 = sim(flipped)["response"].cpu().numpy()

print(f"Original sum: {pat.sum():.0f}, flipped sum: {flipped.sum():.0f}")
print(f"Max diff: {np.abs(r1 - r2).max():.6f} dB")
print(f"Mean diff: {np.abs(r1 - r2).mean():.6f} dB")
if np.abs(r1 - r2).max() < 1e-3:
    print("Bit-flip invariance CONFIRMED — can use as free 2x augmentation")
else:
    print("Bit-flip changes response — cannot use")

# Test spatial reflections
print()
print("=== Spatial reflection tests ===")
for name, transformed in [
    ("flipud (x → -x)", torch.flipud(pat)),
    ("fliplr (y → -y)", torch.fliplr(pat)),
    ("transpose", pat.t()),
    ("rot90", torch.rot90(pat)),
    ("rot180", torch.rot90(pat, 2)),
]:
    with torch.no_grad():
        r = sim(transformed.float())["response"].cpu().numpy()
    diff = np.abs(r1 - r).max()
    status = "INVARIANT" if diff < 1e-3 else f"changes ({diff:.2f} dB)"
    print(f"  {name}: {status}")
