"""
Round 80 — Sobolev training: MSE(f̂, f) + λ · MSE(∇f̂, ∇f)

R79 量測證實 R72 surrogate gradient quality cosine ~0.001 (random)。
這版加 gradient supervision: 訓練時讓 surrogate gradient 跟真實 gradient 匹配。

對 patch antenna: HFSS finite-diff 提供 ground truth gradient (慢但可行 patch geometry < 30 維)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from antenna.ris import RISSimulator
from antenna.utils.config import config

sys.path.insert(0, "script")
from train_surrogate import SurrogateCNN


def soft_max(x, beta=20.0):
    return (1/beta) * torch.logsumexp(beta * x, dim=-1)


def soft_min(x, beta=20.0):
    return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0):
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    main_min = soft_min(main, beta)
    side_max = soft_max(side, beta)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max(main, beta)
        loss = loss + ripple_weight * (main_max - main_min)
    return loss


class SobolevDataset(Dataset):
    """讀 dataset_v2 + 預計算 ground truth ∂loss/∂pattern。"""

    def __init__(self, root: Path, max_n: int = 41, device: str = "cuda:0"):
        self.root = root
        self.max_n = max_n
        self.device = device
        self.entries: list[dict] = []
        with open(root / "entries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for p in entry["pareto"]:
                    self.entries.append({
                        "config": entry["config"],
                        "main_idx_range": entry["main_idx_range"],
                        "ripple_weight": p["ripple_weight"],
                        "pattern_file": p["pattern_file"],
                        "response_file": p["response_file"],
                    })

        # Pre-compute ground truth gradients
        print(f"Pre-computing gradients for {len(self.entries)} entries...")
        self._cached_grads = []
        config.device = device
        for i, e in enumerate(self.entries):
            cfg = e["config"]
            sim = RISSimulator(element_num=cfg["n"], freq_hz=cfg["freq"], inc_theta_deg=cfg["inc"])
            pat = np.load(root / e["pattern_file"]).astype(np.float32)
            pat_t = torch.tensor(pat, device=device, requires_grad=True)
            main_lo, main_hi = e["main_idx_range"]
            resp = sim(pat_t)["response"]
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0,
                                   ripple_weight=e["ripple_weight"])
            grad = torch.autograd.grad(loss, pat_t)[0]
            self._cached_grads.append(grad.detach().cpu().numpy())
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(self.entries)}")
        print(f"Done.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        pat = np.load(self.root / e["pattern_file"]).astype(np.float32)
        n = pat.shape[0]
        padded = np.zeros((self.max_n, self.max_n), dtype=np.float32)
        offset = (self.max_n - n) // 2
        padded[offset:offset + n, offset:offset + n] = pat
        mask = np.zeros((self.max_n, self.max_n), dtype=np.float32)
        mask[offset:offset + n, offset:offset + n] = 1.0
        # Pad gradient too
        grad_padded = np.zeros((self.max_n, self.max_n), dtype=np.float32)
        grad_padded[offset:offset + n, offset:offset + n] = self._cached_grads[idx]
        resp = np.load(self.root / e["response_file"]).astype(np.float32)
        cfg = e["config"]
        config_vec = np.array([
            cfg["freq_ghz"] / 100.0,
            cfg["n"] / 41.0,
            cfg["target_theta_c"] / 90.0,
            cfg["target_width_deg"] / 90.0,
            cfg["inc"] / 90.0,
            e["ripple_weight"] / 5.0,
        ], dtype=np.float32)

        return {
            "pattern": padded,
            "mask": mask,
            "config": config_vec,
            "response": resp,
            "loss_grad": grad_padded,
            "main_lo": e["main_idx_range"][0],
            "main_hi": e["main_idx_range"][1],
            "ripple_weight": e["ripple_weight"],
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_grad", type=float, default=1.0,
                   help="Sobolev gradient loss weight")
    p.add_argument("--out_dir", type=str, default="outputs/r80_sobolev")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = SobolevDataset(Path(args.dataset), device=device)
    print(f"Dataset: {len(ds)} entries\n")

    n_train = int(0.8 * len(ds))
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(ds))
    train_idx = indices[:n_train].tolist()
    test_idx = indices[n_train:].tolist()
    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)

    torch.set_default_device("cpu")
    # Manual shuffle workaround
    class ManualShuffleSampler:
        def __init__(self, n, seed):
            self.n = n
            self.rng = np.random.RandomState(seed)
        def __iter__(self):
            return iter(self.rng.permutation(self.n).tolist())
        def __len__(self):
            return self.n
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=ManualShuffleSampler(len(train_ds), args.seed))
    torch.set_default_device(device)

    surrogate = SurrogateCNN(channels=32, depth=4).to(device)
    opt = torch.optim.Adam(surrogate.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in surrogate.parameters())
    print(f"Surrogate: {n_params:,} params, λ_grad={args.lambda_grad}\n")

    for epoch in range(args.epochs):
        surrogate.train()
        ep_value_loss = 0.0
        ep_grad_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            pat = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            cfg = batch["config"].to(device)
            resp_true = batch["response"].to(device)
            grad_true = batch["loss_grad"].to(device)
            main_lo = batch["main_lo"].to(device)
            main_hi = batch["main_hi"].to(device)
            rw = batch["ripple_weight"].to(device)

            # Forward with gradient tracking
            pat_grad = pat.detach().clone().requires_grad_(True)
            resp_pred = surrogate(pat_grad, mask, cfg)
            value_loss = F.mse_loss(resp_pred, resp_true)

            # Compute predicted ∂loss/∂pat for each batch element
            # Use create_graph=True so we can backprop through gradient
            loss_per_sample = []
            for i in range(pat.shape[0]):
                ll = worst_case_loss(resp_pred[i], int(main_lo[i].item()),
                                     int(main_hi[i].item()),
                                     beta=20.0, ripple_weight=float(rw[i].item()))
                loss_per_sample.append(ll)
            loss_total = torch.stack(loss_per_sample).sum()
            grad_pred = torch.autograd.grad(loss_total, pat_grad, create_graph=True)[0]
            grad_loss = F.mse_loss(grad_pred, grad_true)

            total_loss = value_loss + args.lambda_grad * grad_loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            ep_value_loss += value_loss.item() * pat.shape[0]
            ep_grad_loss += grad_loss.item() * pat.shape[0]
            ep_n += pat.shape[0]

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:3d}  value_mse={ep_value_loss/ep_n:.3f}  "
                  f"grad_mse={ep_grad_loss/ep_n:.6f}")

    # Save
    torch.save(surrogate.state_dict(), out / "surrogate_sobolev.pt")
    print(f"\nsaved to {out}")

    # Quick gradient quality eval on test set
    print("\n=== Gradient quality on test set ===")
    surrogate.eval()
    cos_sims = []
    rel_errs = []
    func_errs = []
    for sample in test_ds:
        pat = torch.from_numpy(sample["pattern"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(sample["mask"]).unsqueeze(0).to(device)
        cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
        grad_true = torch.from_numpy(sample["loss_grad"]).to(device)
        main_lo = sample["main_lo"]
        main_hi = sample["main_hi"]
        rw = sample["ripple_weight"]

        pat_g = pat.detach().clone().requires_grad_(True)
        resp_pred = surrogate(pat_g, mask, cfg)[0]
        loss_pred = worst_case_loss(resp_pred, main_lo, main_hi, beta=20.0, ripple_weight=rw)
        grad_pred = torch.autograd.grad(loss_pred, pat_g)[0][0]

        cos = F.cosine_similarity(grad_pred.flatten().unsqueeze(0),
                                   grad_true.flatten().unsqueeze(0), dim=1).item()
        rel = (grad_pred - grad_true).norm().item() / (grad_true.norm().item() + 1e-8)
        # function value error
        loss_true_val = worst_case_loss(
            torch.from_numpy(sample["response"]).to(device),
            main_lo, main_hi, beta=20.0, ripple_weight=rw,
        ).item()
        cos_sims.append(cos)
        rel_errs.append(rel)
        func_errs.append(abs(loss_pred.item() - loss_true_val))

    print(f"  Mean cosine similarity:  {np.mean(cos_sims):+.3f}")
    print(f"  Median cosine similarity: {np.median(cos_sims):+.3f}")
    print(f"  Mean relative error:     {np.mean(rel_errs):.3f}")
    print(f"  Mean function error:     {np.mean(func_errs):.3f} dB")
    print(f"\n  R72 baseline (no Sobolev): cos~0.001, rel~1.0, func~3.3")


if __name__ == "__main__":
    main()
