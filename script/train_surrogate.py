"""
Round 68 — RIS Surrogate Proof-of-Concept

訓練 (binary_pattern + config) → response 的 NN 替代 RIS simulator。
這是 patch antenna 移植的 critical step：HFSS 太慢 → 需要 NN surrogate。

如果 dataset_v1 (72 entries) 能讓小 NN 學到合理 forward map，
證明 dataset schema + worst-case label 對 patch 也有效。

如果學不會，揭露 schema gap (e.g. 不夠多樣性、變數間 confound)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class RISDataset(Dataset):
    """讀 dataset_v1/entries.jsonl + 對應 .npy 檔。"""

    def __init__(self, root: Path, max_n: int = 41):
        self.root = root
        self.max_n = max_n  # padding to fixed size for batching
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
                        "metrics": p["metrics"],
                    })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        pat = np.load(self.root / e["pattern_file"]).astype(np.float32)
        # Pad pattern to (max_n, max_n)
        n = pat.shape[0]
        padded = np.zeros((self.max_n, self.max_n), dtype=np.float32)
        offset = (self.max_n - n) // 2
        padded[offset:offset + n, offset:offset + n] = pat

        # mask: 1 where pattern exists
        mask = np.zeros((self.max_n, self.max_n), dtype=np.float32)
        mask[offset:offset + n, offset:offset + n] = 1.0

        resp = np.load(self.root / e["response_file"]).astype(np.float32)

        cfg = e["config"]
        config_vec = np.array([
            cfg["freq_ghz"] / 100.0,            # ~0.28 / 0.38
            cfg["n"] / 41.0,                    # 0.51 / 0.76
            cfg["target_theta_c"] / 90.0,       # -0.33 / 0 / +0.33
            cfg["target_width_deg"] / 90.0,     # 0.11 / 0.22 / 0.33
            cfg["inc"] / 90.0,                  # 0.57
            e["ripple_weight"] / 5.0,           # 0 / 0.4
        ], dtype=np.float32)

        return {
            "pattern": padded,
            "mask": mask,
            "config": config_vec,
            "response": resp,
            "metrics": e["metrics"],
        }


class SurrogateMLP(nn.Module):
    """簡單 MLP: (pattern_flat + config) → response."""

    def __init__(self, max_n: int = 41, config_dim: int = 6, response_dim: int = 361,
                 hidden: int = 512, depth: int = 3):
        super().__init__()
        in_dim = max_n * max_n * 2 + config_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, response_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, pattern, mask, config):
        b = pattern.shape[0]
        x = torch.cat([pattern.reshape(b, -1), mask.reshape(b, -1), config], dim=1)
        return self.net(x)


class SurrogateCNN(nn.Module):
    """CNN: (pattern_2d + mask_2d + config_broadcast) → response."""

    def __init__(self, max_n: int = 41, config_dim: int = 6, response_dim: int = 361,
                 channels: int = 32, depth: int = 4):
        super().__init__()
        # Input: 2 (pattern + mask) + config_dim broadcast = 2 + 6 = 8 channels
        c_in = 2 + config_dim
        layers: list[nn.Module] = []
        c = c_in
        for i in range(depth):
            layers += [nn.Conv2d(c, channels, kernel_size=3, padding=1), nn.GELU()]
            c = channels
            if i < depth - 1:
                layers += [nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        # Global pool then dense to response
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Linear(256, response_dim),
        )

    def forward(self, pattern, mask, config):
        b, h, w = pattern.shape
        cfg_map = config[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([pattern.unsqueeze(1), mask.unsqueeze(1), cfg_map], dim=1)
        feat = self.conv(x)
        return self.head(feat)


def evaluate_metrics(pred_resp: np.ndarray, true_resp: np.ndarray, main_lo: int, main_hi: int) -> dict:
    """Compare predicted vs true response on key metrics."""
    def supp_metrics(r):
        m = r[main_lo:main_hi]
        s = np.delete(r, np.arange(main_lo, main_hi))
        return {
            "headline": float(m.max() - s.max()),
            "worst": float(m.min() - s.max()),
            "ripple": float(m.max() - m.min()),
            "side_max": float(s.max()),
            "main_min": float(m.min()),
        }
    return {"true": supp_metrics(true_resp), "pred": supp_metrics(pred_resp)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v1")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--arch", type=str, default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="outputs/r68_surrogate")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = RISDataset(Path(args.dataset))
    print(f"Dataset: {len(ds)} entries")

    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(args.seed))
    print(f"  train: {len(train_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.arch == "mlp":
        model = SurrogateMLP(hidden=args.hidden, depth=args.depth).to(device)
    else:
        model = SurrogateCNN(channels=args.channels, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch.upper()} {n_params:,} params")

    # Train
    train_log = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pat = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            cfg = batch["config"].to(device)
            tgt = batch["response"].to(device)

            pred = model(pat, mask, cfg)
            loss = nn.functional.mse_loss(pred, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * pat.shape[0]
        train_loss /= len(train_ds)

        # Test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                pat = batch["pattern"].to(device)
                mask = batch["mask"].to(device)
                cfg = batch["config"].to(device)
                tgt = batch["response"].to(device)
                pred = model(pat, mask, cfg)
                test_loss += nn.functional.mse_loss(pred, tgt).item() * pat.shape[0]
        test_loss /= len(test_ds)
        train_log.append((epoch, train_loss, test_loss))
        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:3d}  train_mse={train_loss:.3f}  test_mse={test_loss:.3f}")

    # Final eval: per-sample metrics on test set
    print("\n=== Test set metric reproduction ===")
    print(f"{'idx':>3} | {'config':<45} | {'true_worst':>10} | {'pred_worst':>10} | "
          f"{'true_main_min':>13} | {'pred_main_min':>13}")
    print("-" * 110)
    model.eval()
    abs_diffs = []
    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            pat = torch.from_numpy(sample["pattern"]).unsqueeze(0).to(device)
            mask = torch.from_numpy(sample["mask"]).unsqueeze(0).to(device)
            cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
            pred = model(pat, mask, cfg).cpu().numpy()[0]
            true = sample["response"]

            # find main_idx_range from config
            theta_c = sample["config"][2] * 90.0
            tw = sample["config"][3] * 90.0
            main_lo = max(0, int(round((theta_c + 90) * 2 - tw)))
            main_hi = min(361, int(round((theta_c + 90) * 2 + tw)))

            m = evaluate_metrics(pred, true, main_lo, main_hi)
            cfg_str = (f"f={sample['config'][0]*100:.0f}GHz n={sample['config'][1]*41:.0f} "
                       f"θc={theta_c:+.0f} w={tw:.0f} rw={sample['config'][5]*5:.1f}")
            print(f"{i:3d} | {cfg_str:<45} | {m['true']['worst']:+10.2f} | "
                  f"{m['pred']['worst']:+10.2f} | "
                  f"{m['true']['main_min']:+13.2f} | {m['pred']['main_min']:+13.2f}")
            abs_diffs.append(abs(m["true"]["worst"] - m["pred"]["worst"]))

    print(f"\nMean abs error in worst_supp: {np.mean(abs_diffs):.2f} dB")
    print(f"Max abs error in worst_supp:  {np.max(abs_diffs):.2f} dB")

    # save
    torch.save(model.state_dict(), out / "surrogate.pt")
    np.save(out / "train_log.npy", np.array(train_log))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
