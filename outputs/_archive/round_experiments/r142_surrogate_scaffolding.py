"""Round 142 — Surrogate-in-the-loop scaffolding (Phase 2 begins).

Train a tiny CNN to learn the RIS analytical forward pass at one config:
  n=31, inc=51deg, 38GHz, broadside, width=10deg

Why this matters: patch transition needs surrogate-in-the-loop. If our recipe
+ loss design + joint early-stop survive moving from analytical sim ->
learned surrogate gradient, the methodology transfers cleanly.

R142 is the prep step: build + train surrogate, validate R^2 + worst error.
R143 will plug this surrogate into the optimizer and compare to analytical.
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
n = 31
freq = 38e9
inc = 51.0


# ---------- Dataset generation ----------
def generate_dataset(n_samples, sim, seed=42):
    """Generate (binary_pattern, response) pairs by querying analytical sim."""
    torch.manual_seed(seed)
    patterns = torch.bernoulli(torch.full((n_samples, n, n), 0.5)).to("cuda:0")
    responses = []
    for i in range(n_samples):
        with torch.no_grad():
            r = sim(patterns[i])["response"]
        responses.append(r)
    responses = torch.stack(responses)  # (n_samples, 361)
    return patterns, responses


# ---------- Tiny CNN surrogate (small, dropout) ----------
class TinyRISsurrogate(nn.Module):
    """Compact conv surrogate: (1, n, n) binary -> (361,) response.

    Design notes:
      - Smaller channel widths (8/16) since output is only 361 bins
      - Heavy dropout (0.3) to combat overfit on limited data
      - Smaller fc bottleneck (256)
    """
    def __init__(self, n_elem):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 361)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.flatten(1)
        h = self.dropout(F.relu(self.fc1(h)))
        out = self.fc2(h)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        return out


print("=" * 100, flush=True)
print(f"R142 -- Surrogate scaffolding at n={n}, inc={inc}, freq={freq/1e9}GHz", flush=True)
print("=" * 100, flush=True)

sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)

print("\n[1/4] Generating dataset (10000 train + 1000 test random binary patterns)...", flush=True)
t0 = time.time()
patterns_train, responses_train = generate_dataset(10000, sim, seed=0)
patterns_test, responses_test = generate_dataset(1000, sim, seed=999)
print(f"  done in {time.time()-t0:.1f}s. train shape: {patterns_train.shape}, "
      f"resp shape: {responses_train.shape}", flush=True)

print("\n[2/4] Building surrogate model...", flush=True)
model = TinyRISsurrogate(n).to("cuda:0")
n_params = sum(p.numel() for p in model.parameters())
print(f"  TinyRISsurrogate: {n_params:,} parameters", flush=True)

print("\n[3/4] Training surrogate (200 epochs, batch=128, AdamW lr=1e-3, early stop on test)...", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
batch_size = 128

best_test_mse = float("inf")
best_state = None
patience = 30
stale = 0

t0 = time.time()
for epoch in range(200):
    perm = torch.randperm(len(patterns_train))
    model.train()
    epoch_loss = 0
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i+batch_size]
        x = patterns_train[idx]
        y = responses_train[idx]
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= len(perm)

    # Eval each epoch
    model.eval()
    with torch.no_grad():
        test_pred = model(patterns_test)
        test_loss = loss_fn(test_pred, responses_test).item()

    if test_loss < best_test_mse - 1e-3:
        best_test_mse = test_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:>4d} | train MSE {epoch_loss:.4f} | test MSE {test_loss:.4f}"
              f" | best {best_test_mse:.4f} | stale {stale}", flush=True)

    if stale >= patience:
        print(f"  Early stop at epoch {epoch+1} (no test improvement for {patience} epochs)", flush=True)
        break

print(f"  total training: {time.time()-t0:.1f}s, best test MSE = {best_test_mse:.4f}", flush=True)
model.load_state_dict(best_state)

print("\n[4/4] Final fit quality:", flush=True)
model.eval()
with torch.no_grad():
    test_pred = model(patterns_test).cpu().numpy()
    test_true = responses_test.cpu().numpy()

abs_err = np.abs(test_pred - test_true)
mean_abs_err = float(np.mean(abs_err))
max_abs_err = float(np.max(abs_err))
mean_max_err_per_sample = float(np.mean(np.max(abs_err, axis=1)))

# R^2
ss_res = float(np.sum((test_pred - test_true) ** 2))
ss_tot = float(np.sum((test_true - test_true.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot

# Quality on main region (broadside w=10): indices 170-190
main_lo, main_hi = 170, 190
main_err = np.abs(test_pred[:, main_lo:main_hi] - test_true[:, main_lo:main_hi])
print(f"  R^2:                        {r2:.4f}", flush=True)
print(f"  mean |abs err| (all bins):  {mean_abs_err:.3f} dB", flush=True)
print(f"  max  |abs err| (any bin):   {max_abs_err:.3f} dB", flush=True)
print(f"  mean max err per pattern:   {mean_max_err_per_sample:.3f} dB", flush=True)
print(f"  mean main-region |abs err|: {float(np.mean(main_err)):.3f} dB", flush=True)
print(f"  max  main-region |abs err|: {float(np.max(main_err)):.3f} dB", flush=True)

# Save model for R143
torch.save(model.state_dict(), "outputs/r142_surrogate_n31.pt")
print(f"\n  Saved: outputs/r142_surrogate_n31.pt", flush=True)

print("\n" + "=" * 70, flush=True)
print("Verdict: surrogate is usable for optimization if:", flush=True)
print("  - R^2 > 0.95", flush=True)
print("  - main-region mean abs err < 1 dB (tight enough for worst-case loss)", flush=True)
print("=" * 70, flush=True)
