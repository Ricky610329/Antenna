"""
Round 73 — Conditional Generator (config + ripple_mode → pattern)

R71 揭露: 同 config 下 rw=0 vs rw=2 的最優 pattern hamming ~50% (multimodal)。
R1-R30 lab generator 失敗是因為 unconditional → 學 mean of multimodal = garbage。

這版測試: 把 ripple_weight 當顯式 mode condition 加進 generator input。
如果 hypothesis 對:
  - 同 config 下 rw=0 vs rw=2 預測 pattern 應 hamming ~50%
  - 對 simulator 跑過後 metrics 應接近 ground truth

如果失敗:
  - 預測 collapse 到同一 pattern (rw 沒效)
  - Metrics 平庸

對 patch 移植: (geometry_target + use_case_mode) → geometry params 是同樣 pattern。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from antenna.ris import RISSimulator
from antenna.utils.config import config


class GenDataset(Dataset):
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
                        "main_idx_range": entry["main_idx_range"],
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
            e["ripple_weight"] / 5.0,  # explicit mode condition
        ], dtype=np.float32)

        return {
            "pattern": padded,
            "mask": mask,
            "config": config_vec,
            "n": n,
            "main_lo": e["main_idx_range"][0],
            "main_hi": e["main_idx_range"][1],
            "true_metrics": e["metrics"],
            "freq_ghz": cfg["freq_ghz"],
            "inc": cfg["inc"],
        }


class ConditionalGenerator(nn.Module):
    """config → 41×41 logits (deconv from latent)."""

    def __init__(self, config_dim: int = 6, max_n: int = 41, channels: int = 64, depth: int = 4):
        super().__init__()
        self.max_n = max_n
        # Encode config to spatial feature
        self.config_encoder = nn.Sequential(
            nn.Linear(config_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, channels * 6 * 6),  # 6x6 feature map
            nn.GELU(),
        )
        # Upsample to 41×41 with conv
        layers: list[nn.Module] = []
        c = channels
        for _ in range(depth - 1):
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(c, c, kernel_size=3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(c, 1, kernel_size=3, padding=1)]  # logits
        self.decoder = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, config_vec: torch.Tensor) -> torch.Tensor:
        b = config_vec.shape[0]
        feat = self.config_encoder(config_vec).reshape(b, self.channels, 6, 6)
        logits = self.decoder(feat)
        # Crop/pad to max_n
        h = logits.shape[2]
        if h > self.max_n:
            offset = (h - self.max_n) // 2
            logits = logits[:, :, offset:offset + self.max_n, offset:offset + self.max_n]
        elif h < self.max_n:
            pad = (self.max_n - h) // 2
            logits = nn.functional.pad(logits, (pad, self.max_n - h - pad, pad, self.max_n - h - pad))
        return logits.squeeze(1)


def supp_metrics(resp: np.ndarray, main_lo: int, main_hi: int) -> dict:
    main = resp[main_lo:main_hi]
    side = np.delete(resp, np.arange(main_lo, main_hi))
    return {
        "worst": float(main.min() - side.max()),
        "headline": float(main.max() - side.max()),
        "ripple": float(main.max() - main.min()),
        "main_min": float(main.min()),
        "side_max": float(side.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v2")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="outputs/r73_generator")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config.device = device

    ds = GenDataset(Path(args.dataset))
    print(f"Dataset: {len(ds)} entries")

    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    # Manual deterministic split (avoid cuda generator issue)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(ds))
    train_idx = indices[:n_train].tolist()
    test_idx = indices[n_train:].tolist()
    train_ds = torch.utils.data.Subset(ds, train_idx)
    test_ds = torch.utils.data.Subset(ds, test_idx)

    # Force default device cpu before DataLoader (config.device set torch default to cuda)
    torch.set_default_device("cpu")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ConditionalGenerator(channels=args.channels, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Generator: {n_params:,} params\n")

    # Train (BCE-like for binary patterns)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_total = 0
        for batch in train_loader:
            cfg = batch["config"].to(device)
            tgt = batch["pattern"].to(device)
            mask = batch["mask"].to(device)
            pred_logits = model(cfg)
            loss_per_pixel = bce(pred_logits, tgt) * mask
            loss = loss_per_pixel.sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * cfg.shape[0]
            n_total += cfg.shape[0]
        if (epoch + 1) % 50 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:3d}  bce={total_loss/n_total:.4f}")

    # === Evaluate on test set: SIM-based metrics ===
    print("\n=== Test set: predict pattern → run simulator → compare metrics ===")
    print(f"{'idx':>3} | {'config':<45} | {'true_worst':>10} | {'pred_worst':>10} | "
          f"{'hamming(pred,gt)':>16}")
    print("-" * 110)
    model.eval()
    abs_diffs = []
    pred_metrics_per_test = []
    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
            pred_logits = model(cfg)
            pred_bin_full = (torch.sigmoid(pred_logits) > 0.5).float()[0].cpu().numpy()

            # Crop to actual n
            n = sample["n"]
            offset = (41 - n) // 2
            pred_bin = pred_bin_full[offset:offset + n, offset:offset + n]
            true_bin_full = sample["pattern"]
            true_bin = true_bin_full[offset:offset + n, offset:offset + n]

            # Run real sim on predicted pattern
            sim = RISSimulator(element_num=n, freq_hz=sample["freq_ghz"] * 1e9,
                               inc_theta_deg=sample["inc"])
            pat_t = torch.from_numpy(pred_bin).float().to(device)
            with torch.no_grad():
                pred_resp = sim(pat_t)["response"].cpu().numpy()
            pred_m = supp_metrics(pred_resp, sample["main_lo"], sample["main_hi"])
            true_m = sample["true_metrics"]

            hamming = (pred_bin != true_bin).sum() / pred_bin.size
            cfg_str = (f"f={sample['config'][0]*100:.0f}GHz n={n} "
                       f"θc={sample['config'][2]*90:+.0f} w={sample['config'][3]*90:.0f} "
                       f"rw={sample['config'][5]*5:.1f}")
            print(f"{i:3d} | {cfg_str:<45} | {true_m['worst_supp']:+10.2f} | "
                  f"{pred_m['worst']:+10.2f} | {hamming:>16.2%}")
            abs_diffs.append(abs(true_m["worst_supp"] - pred_m["worst"]))
            pred_metrics_per_test.append({
                "true_worst": true_m["worst_supp"],
                "pred_worst": pred_m["worst"],
                "hamming": hamming,
                "rw": sample["config"][5] * 5,
            })

    print(f"\nMean abs error in worst_supp: {np.mean(abs_diffs):.2f} dB")
    print(f"Mean hamming(pred, gt):       {np.mean([m['hamming'] for m in pred_metrics_per_test]):.2%}")

    # Mode-pair test: same config, two ripple weights → predicted patterns should differ
    print("\n=== Multimodal sanity check: same config, rw=0 vs rw=2 ===")
    print(f"{'config':<40} | {'hamming(pred_rw0, pred_rw2)':>30}")
    print("-" * 75)
    # Group by config base (remove rw)
    base_configs = {}  # (freq, n, θc, w) → list of (rw, pred_pat)
    model.eval()
    with torch.no_grad():
        for sample in test_ds:
            base_key = (sample["config"][0], sample["config"][1], sample["config"][2], sample["config"][3])
            cfg = torch.from_numpy(sample["config"]).unsqueeze(0).to(device)
            pred_logits = model(cfg)
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()[0].cpu().numpy()
            base_configs.setdefault(base_key, []).append({
                "rw": sample["config"][5] * 5,
                "pred": pred_bin,
                "n": sample["n"],
            })

    n_pairs = 0
    pair_hammings = []
    for k, lst in base_configs.items():
        if len(lst) >= 2:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if abs(lst[i]["rw"] - lst[j]["rw"]) > 0.5:
                        n = min(lst[i]["n"], lst[j]["n"])
                        offset = (41 - n) // 2
                        p1 = lst[i]["pred"][offset:offset+n, offset:offset+n]
                        p2 = lst[j]["pred"][offset:offset+n, offset:offset+n]
                        h = (p1 != p2).sum() / p1.size
                        pair_hammings.append(h)
                        n_pairs += 1
                        cfg_str = (f"f={k[0]*100:.0f}GHz n={lst[i]['n']} "
                                   f"θc={k[2]*90:+.0f} w={k[3]*90:.0f}")
                        print(f"{cfg_str:<40} | {h:>30.2%}")
    if n_pairs:
        print(f"\nMean pair hamming: {np.mean(pair_hammings):.2%}")
        print(f"Expected if conditional: ~50% (matches dataset)")
        print(f"Expected if collapse:    near 0%")


if __name__ == "__main__":
    main()
