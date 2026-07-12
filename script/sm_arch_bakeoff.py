# -*- coding: utf-8 -*-
"""
script/sm_arch_bakeoff.py — SM 架構對決（analysis-04;零 HFSS 純離線,Ricky 2026-07-13 批准）。

四臂同資料同 seed：主幹 {MLP(現行), CNN(25×25 卷積+對稱先驗候補)} × 輸出 {curve(現行),
multihead(曲線+直接 wm/oob 頭)}。判準（跑前寫死）：held-out 排序 ρ（wm 與 oob_bad 各自 spearman）
—— 任一臂 oob ρ 顯著壓過現行（≥+0.1 且 p<.05 級）＝該軸有效,贏家整合進 v19 訓練線;
全平＝架構不是瓶頸,回頭支持「飽和是本質」定論。

用法: python -m script.sm_arch_bakeoff [--epochs 40]
資料/切分沿用 sm_reanchor（clean_stores.txt + hash 決定性切分）,標籤 wm/oob 由真響應現算（同一把尺）。
"""
import argparse
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from script.sm_reanchor import _load_clean, _load_harvest, LABELS, _cfg  # noqa: E402
from script.dedust import oob_metrics  # noqa: E402
from antenna.losses import worst_margin  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

N_PTS = sum(_cfg.targets[LABELS[0]]["width"])


def _labels_of(y):
    """真響應 → (wm, oob_bad) 標籤（與判讀同一把尺）。"""
    w, _ = worst_margin(torch.as_tensor(y).reshape(len(LABELS), N_PTS), LABELS, _cfg.targets)
    ob = oob_metrics(np.asarray(y).reshape(len(LABELS), N_PTS))["oob_bad"]
    return float(w), float(ob)


class Trunk(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(625, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU())
            self.out_dim = 256
        else:                                            # cnn:25×25 影像先驗
            self.net = nn.Sequential(
                nn.Unflatten(1, (1, 25, 25)) if False else nn.Identity())
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 12×12
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 6×6
                nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU())
            self.out_dim = 256
        self.kind = kind

    def forward(self, x):
        if self.kind == "mlp":
            return self.net(x)
        return self.conv(x.reshape(-1, 1, 25, 25))


class SMArch(nn.Module):
    def __init__(self, trunk, multihead):
        super().__init__()
        self.trunk = Trunk(trunk)
        self.curve = nn.Linear(self.trunk.out_dim, len(LABELS) * N_PTS)
        self.multihead = multihead
        if multihead:
            self.h_wm = nn.Linear(self.trunk.out_dim, 1)
            self.h_oob = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, x):
        z = self.trunk(x)
        c = self.curve(z)
        if self.multihead:
            return c, self.h_wm(z).squeeze(-1), self.h_oob(z).squeeze(-1)
        return c, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=0.3, help="直接頭損失權重")
    args = ap.parse_args()
    tr, ho = _load_clean()
    replay, _ = _load_harvest(2000, 0)
    print(f"資料: 乾淨 train {len(tr)} / held-out {len(ho)} / 重放 {len(replay)}")

    def prep(items):
        X = torch.stack([x for x, _ in items]).float()
        Y = torch.stack([y for _, y in items]).float().reshape(len(items), -1)
        lb = np.array([_labels_of(y) for _, y in items])
        return X, Y, torch.tensor(lb[:, 0]).float(), torch.tensor(lb[:, 1]).float()

    Xt, Yt, Wt, Ot = prep(list(tr) * 8 + list(replay))
    Xh, Yh, Wh, Oh = prep(ho)
    wm_true, oob_true = Wh.numpy(), Oh.numpy()
    print(f"訓練列 {len(Xt)};held-out {len(Xh)}")
    print("| 臂 | held-out wm ρ | oob ρ | 曲線 MSE |")
    print("|---|---|---|---|")
    for trunk in ("mlp", "cnn"):
        for mh in (False, True):
            torch.manual_seed(0)
            net = SMArch(trunk, mh)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            n = len(Xt)
            for ep in range(args.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, 128):
                    idx = perm[i:i + 128]
                    c, hw, hob = net(Xt[idx])
                    loss = ((c - Yt[idx]) ** 2).mean()
                    if mh:
                        loss = loss + args.lam * (((hw - Wt[idx]) ** 2).mean()
                                                  + ((hob - Ot[idx]) ** 2).mean())
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                c, hw, hob = net(Xh)
                mse = float(((c - Yh) ** 2).mean())
                if mh:                                   # 多頭臂:用直接頭排序
                    wm_p, ob_p = hw.numpy(), hob.numpy()
                else:                                    # 曲線臂:間接算（現行路徑）
                    wm_p = np.array([_labels_of(ci)[0] for ci in c])
                    ob_p = np.array([_labels_of(ci)[1] for ci in c])
            rw, pw = spearmanr(wm_p, wm_true)
            ro, po = spearmanr(ob_p, oob_true)
            tag = f"{trunk}{'+multihead' if mh else ''}"
            print(f"| {tag} | {rw:+.3f} (p={pw:.1e}) | {ro:+.3f} (p={po:.1e}) | {mse:.3f} |", flush=True)
    print("\n判準:任一臂 oob ρ ≥ 現行(mlp) +0.1 且顯著 → 該軸有效,整合 v19;全平=架構非瓶頸。")


if __name__ == "__main__":
    main()
