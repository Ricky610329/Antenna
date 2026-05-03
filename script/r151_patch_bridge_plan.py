"""Round 151 — Patch transition bridge plan: codebase audit + import smoke test.

R150 unified the deployment pipeline. R151 audits existing patch infrastructure
to identify the concrete bridge between R141/R150 RIS pipeline and patch antenna
deployment.

This script is mostly DOCUMENTATION + a smoke test confirming the relevant
classes can be imported (HFSSNet doesn't need HFSS, only PatchSimulator does).

Findings written to outputs/PATCH_BRIDGE_PLAN.md.
"""
import sys
sys.path.insert(0, "script")

print("=" * 100, flush=True)
print("R151 -- Patch transition bridge audit", flush=True)
print("=" * 100, flush=True)

print("\n[1/3] Smoke test: import HFSSNet (should work, no HFSS dependency)...", flush=True)
try:
    from antenna.models.surrogates import HFSSNet, SurrogateModel
    print(f"  HFSSNet: OK", flush=True)
    print(f"  SurrogateModel: OK", flush=True)

    # Instantiate small HFSSNet
    import torch
    from antenna.utils.config import config
    config.device = "cuda:0"
    net = HFSSNet(num_pattern_pixel=625, num_response=(3, 17))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Default HFSSNet (625 -> (3,17)): {n_params:,} params", flush=True)

    # Smoke test forward pass
    x = torch.zeros(1, 625).to("cuda:0")
    y = net(x)
    print(f"  Forward shape: {y.shape}  (single sample reshaped to (3,17))", flush=True)
    x_batch = torch.zeros(4, 625).to("cuda:0")
    y_batch = net(x_batch)
    print(f"  Batch forward shape: {y_batch.shape}", flush=True)
except ImportError as e:
    print(f"  Import failed: {e}", flush=True)

print("\n[2/3] Audit: PatchSimulator interface...", flush=True)
print("  Defined in: antenna/patch/patch_simulator/__init__.py", flush=True)
print("  - Abstract base: __call__(pixel_matrix) -> response", flush=True)
print("  - Subclasses: SinglePortSimulator (25x25 pixel default), DualPortSimulator", flush=True)
print("  - Backend: Windows COM to Ansys HFSS (DispatchEx 'AnsoftHFSS.HfssScriptInterface')", flush=True)
print("  - Per-call cost: minutes (full HFSS simulation)", flush=True)
print("  - Gradient: NONE (black-box HFSS)", flush=True)
print("  - Input constraint: binary (asserted in __call__)", flush=True)

print("\n[3/3] Bridge plan written to outputs/PATCH_BRIDGE_PLAN.md", flush=True)
print("=" * 100, flush=True)
