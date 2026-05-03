"""
Round 69 — Metric Surrogate (預測 scalar metrics 而非 full response)

R68 forward surrogate (361-dim) 太高維對 72 examples 學不會。
這版改成預測關鍵 scalar metrics:
- worst_supp, headline_supp, main_min, main_max, ripple, side_max

對 patch 移植: surrogate 預測 worst-case S11 / BW / peak gain 而非 full S11(f) 曲線
通常更可行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


METRIC_NAMES = ["worst_supp", "headline_supp", "main_min", "main_max", "main_ripple", "side_max"]


class MetricDataset(Dataset):
    def __init__(self, root: Path, max_n: int = 41):
        self.root = root
        self.max_n = max_n
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
                        "ripple_weight": p["ripple_weight"],
                        "pattern_file": p["pattern_file"],
                        "metrics": p["metrics"],
                    })

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

        cfg = e["config"]
        config_vec = np.array([
            cfg["freq_ghz"] / 100.0,
            cfg["n"] / 41.0,
            cfg["target_theta_c"] / 90.0,
            cfg["target_width_deg"] / 90.0,
            cfg["inc"] / 90.0,
            e["ripple_weight"] / 5.0,
        ], dtype=np.float32)

        metrics_vec = np.array([e["metrics"][k] for k in METRIC_NAMES], dtype=np.float32)

        return {
            "pattern": padded,
            "mask": mask,
            "config": config_vec,
            "metrics": metrics_vec,
        }


class MetricCNN(nn.Module):
    def __init__(self, max_n: int = 41, config_dim: int = 6, n_metrics: int = 6,
                 channels: int = 32, depth: int = 4):
        super().__init__()
        c_in = 2 + config_dim
        layers: list[nn.Module] = []
        c = c_in
        for i in range(depth):
            layers += [nn.Conv2d(c, channels, kernel_size=3, padding=1), nn.GELU()]
            c = channels
            if i < depth - 1:
                layers += [nn.Conv2d(c, c, kernel_size=3, padding=1, stride=2), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Linear(128, n_metrics),
        )

    def forward(self, pattern, mask, config):
        b, h, w = pattern.shape
        cfg_map = config[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([pattern.unsqueeze(1), mask.unsqueeze(1), cfg_map], dim=1)
        return self.head(self.conv(x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v1")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="outputs/r69_metric_surrogate")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = MetricDataset(Path(args.dataset))
    print(f"Dataset: {len(ds)} entries")

    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Compute label statistics for normalization
    all_metrics = []
    for i in range(len(train_ds)):
        all_metrics.append(train_ds[i]["metrics"])
    all_metrics = np.stack(all_metrics)
    mean = all_metrics.mean(axis=0)
    std = all_metrics.std(axis=0) + 1e-6
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    print(f"Metric stats: mean={mean.round(2)}, std={std.round(2)}")

    model = MetricCNN(channels=args.channels, depth=args.depth, n_metrics=len(METRIC_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params\n")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pat = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            cfg = batch["config"].to(device)
            tgt = (batch["metrics"].to(device) - mean_t) / std_t

            pred = model(pat, mask, cfg)
            loss = nn.functional.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * pat.shape[0]
        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                pat = batch["pattern"].to(device)
                mask = batch["mask"].to(device)
                cfg = batch["config"].to(device)
                tgt = (batch["metrics"].to(device) - mean_t) / std_t
                pred = model(pat, mask, cfg)
                test_loss += nn.functional.mse_loss(pred, tgt).item() * pat.shape[0]
        test_loss /= len(test_ds)
        if (epoch + 1) % 50 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:3d}  train_norm_mse={train_loss:.3f}  test_norm_mse={test_loss:.3f}")

    # Per-metric error analysis on test set
    print("\n=== Per-metric test errors ===")
    print(f"{'metric':>15} | {'true_mean':>10} | {'true_std':>9} | "
          f"{'mae':>7} | {'rmse':>7} | {'mae/std':>7}")
    print("-" * 70)
    model.eval()
    pred_all = []
    true_all = []
    with torch.no_grad():
        for batch in test_loader:
            pat = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            cfg = batch["config"].to(device)
            tgt = batch["metrics"].numpy()
            pred_norm = model(pat, mask, cfg).cpu().numpy()
            pred = pred_norm * std + mean
            pred_all.append(pred)
            true_all.append(tgt)
    pred_all = np.concatenate(pred_all)
    true_all = np.concatenate(true_all)
    for i, name in enumerate(METRIC_NAMES):
        mae = np.abs(pred_all[:, i] - true_all[:, i]).mean()
        rmse = np.sqrt(((pred_all[:, i] - true_all[:, i]) ** 2).mean())
        print(f"{name:>15} | {true_all[:, i].mean():+10.2f} | {true_all[:, i].std():9.2f} | "
              f"{mae:7.2f} | {rmse:7.2f} | {mae/(true_all[:, i].std()+1e-6):7.2f}")

    # Sample predictions
    print("\n=== Sample test predictions ===")
    print(f"{'config':<45} | {'true_worst':>10} | {'pred_worst':>10} | "
          f"{'true_ripple':>11} | {'pred_ripple':>11}")
    print("-" * 100)
    for i in range(min(15, len(test_ds))):
        sample = test_ds[i]
        pat = torch.from_numpy(sample["pattern"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(sample["mask"]).unsqueeze(0).to(device)
        cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(pat, mask, cfg).cpu().numpy()[0]
        pred = pred_norm * std + mean
        true = sample["metrics"]

        cfg_str = (f"f={sample['config'][0]*100:.0f}GHz n={sample['config'][1]*41:.0f} "
                   f"θc={sample['config'][2]*90:+.0f} w={sample['config'][3]*90:.0f} "
                   f"rw={sample['config'][5]*5:.1f}")
        print(f"{cfg_str:<45} | {true[0]:+10.2f} | {pred[0]:+10.2f} | "
              f"{true[4]:11.2f} | {pred[4]:11.2f}")

    torch.save(model.state_dict(), out / "metric_surrogate.pt")


if __name__ == "__main__":
    main()
