# -*- coding: utf-8 -*-
"""script/sm_invert.py — 梯度反傳生成器（Ricky 2026-07-14 核准;G 臂/R29 前置,零 HFSS）。

固定 SM（可微 MLP）＋rad 頭,把 pattern 當變數反傳——找 SM「認為」滿足 S11/Gain/rad 的 pattern。
歷史教訓內建：
  ①距離分層 trust-region（terrain 定案:SM 局部可信,遠=重抽）——從錨出發,分帶罰距離;
  ②adversarial 風險=特徵不是 bug——遠帶樣本=SM 過度自信點,HFSS 量回來=主動學習極品（反自餵）;
  ③straight-through 二值化（前向硬二值、反向 sigmoid 梯度）,終產物過可製造統計（報告,不強制）。
用法（開發機,零 HFSS）:
  python -m script.sm_invert run [--sm sm_reanchor30.pth] [--steps 400] [--out tmp/invert_gallery.png]
產物=gallery 圖+預測表;候選要進批次線照常 select/check-dup/HFSS 驗證（R29 G 臂）。
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config, DATASET_PATH
_config.device = "cpu"
import torch

from antenna.training import load_config, setup_responses, PORT_SPECS
from antenna.losses import worst_margin
from antenna.zoo import SURROGATES
from script.dedust import DEFAULT_CFG, FEED, piece_stats, _all_input_folders

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#? 起點錨（名, pid）——冠軍×2＋half 半成品×2＋隨機×1;loadp 式全夾 fallback
ANCHORS = [("margin王", "m23b4_030_r3_001"),
           ("x00", "x00_c21k2"),
           ("p00_half", "n27b1_018_p00"),
           ("t07_half", "n27b1_020_t07"),
           ("random", None)]
BANDS = (10, 25, 60, 625)                                 # 625=自由（無距離罰）


def _find(pid):
    for fol in _all_input_folders():
        f = DATASET_PATH.joinpath(fol, pid + ".pt")
        if f.exists():
            return np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
    raise SystemExit(f"找不到 {pid}")


def run(args):
    cfg = load_config(DEFAULT_CFG)
    setup_responses(cfg)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    cache = os.path.join(REPO, "tmp", "sm_invert")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()
    for p_ in sm.model.parameters():
        p_.requires_grad_(False)
    import torch.nn as nn
    _rh = torch.load(str(DATASET_PATH.joinpath(args.rad_head)), weights_only=False)
    radnet = nn.Sequential(nn.Linear(625, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                           nn.Linear(256, 2 * _rh["K"]))
    radnet.load_state_dict(_rh["state"])
    radnet.eval()
    for p_ in radnet.parameters():
        p_.requires_grad_(False)
    th = np.asarray(_rh["theta"], float)
    phi = np.pi * (th - th.min()) / (th.max() - th.min())
    B = torch.tensor(np.cos(np.arange(_rh["K"]).reshape(-1, 1) * phi.reshape(1, -1)), dtype=torch.float32)
    win = torch.tensor(np.abs(th) <= 45.0)
    i0 = int(np.argmin(np.abs(th)))
    rng = np.random.default_rng(args.seed)
    #? 可微 wm（官方 worst_margin 內部 float() 斷梯度——工具自帶 torch 版,定義逐字對齊 losses.py）
    _bands = []
    for i, lab in enumerate(labels):
        t = cfg.targets[lab]
        w = t["width"]
        _bands.append((i, w[0] + w[1], w[0] + w[1] + w[2], float(t["center"]), t["method"]))

    def wm_torch(pred):
        pr = pred.reshape(len(labels), n_pts)
        ms = [(c - pr[i][a:b].max()) if m == "low" else (pr[i][a:b].min() - c)
              for i, a, b, c, m in _bands]
        return torch.min(torch.stack(ms))

    def invert(p0, band):
        """從 p0 出發、距離帶 band 內反傳。回 (pattern, 診斷 dict)。"""
        theta = torch.tensor(np.where(p0.reshape(-1), 3.0, -3.0) if p0 is not None
                             else rng.normal(0, 1, 625), dtype=torch.float32, requires_grad=True)
        p0t = torch.tensor((p0.reshape(-1) if p0 is not None else np.zeros(625)) > 0.5,
                           dtype=torch.float32)
        opt = torch.optim.Adam([theta], lr=args.lr)
        for step in range(args.steps):
            opt.zero_grad()
            q = torch.sigmoid(theta)
            qb = (q > 0.5).float()
            q_st = qb + q - q.detach()                       # straight-through
            pred = sm.model(q_st)
            wm = wm_torch(pred)
            fits = (radnet(q_st).reshape(2, _rh["K"]) @ B)
            rad_m = torch.min(torch.stack([f[win].min() - (f[i0] - 3.0) for f in fits]))
            pr = pred.reshape(len(labels), n_pts)
            far = list(range(4)) + list(range(n_pts - 4, n_pts))
            oob = pr[1][far].max() - pr[0][far].min()
            d = (q_st - p0t).abs().sum()
            loss = (torch.relu(0.3 - wm) + torch.relu(0.3 - rad_m) * args.w_rad
                    + torch.relu(oob - 8.0) * args.w_oob
                    + torch.relu(d - band) * 0.5)
            loss.backward()
            opt.step()
            with torch.no_grad():                            # feed 區釘死金屬
                theta.data[FEED[0] * 25 + FEED[1]] = 6.0
        with torch.no_grad():
            qf = (torch.sigmoid(theta) > 0.5).numpy().reshape(25, 25)
            qf[FEED] = True
            q_t = torch.tensor(qf.reshape(-1), dtype=torch.float32)
            pred = sm.model(q_t)
            wm, _ = worst_margin(pred, labels, cfg.targets)
            fits = (radnet(q_t).reshape(2, _rh["K"]) @ B)
            rad_m = min(float(f[win].min() - (f[i0] - 3.0)) for f in fits)
            pr = pred.reshape(len(labels), n_pts)
            oob = float(pr[1][far].max() - pr[0][far].min())
            st = piece_stats(qf)
            dd = int(np.sum(qf != p0.reshape(25, 25))) if p0 is not None else -1
        return qf, dict(wm=round(float(wm), 2), rad=round(rad_m, 2), oob=round(oob, 2),
                        d=dd, metal=st["metal_px"], dust=st["n_1px"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
    fig, axes = plt.subplots(len(ANCHORS), len(BANDS) + 1, figsize=(2.1 * (len(BANDS) + 1), 2.1 * len(ANCHORS)))
    print(f"{'起點':<10} {'帶':>5} {'pred_wm':>8} {'pred_rad':>9} {'pred_oob':>9} {'d':>5} {'metal':>6} {'塵':>3}")
    for r, (name, pid) in enumerate(ANCHORS):
        p0 = _find(pid) if pid else None
        if p0 is not None:
            axes[r, 0].imshow(p0, cmap="gray_r", interpolation="nearest")
        axes[r, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        axes[r, 0].set_title("錨" if r == 0 else "", fontsize=9)
        for c, band in enumerate(BANDS, start=1):
            qf, info = invert(p0 if p0 is not None else rng.random((25, 25)) > 0.5, band)
            axes[r, c].imshow(qf, cmap="gray_r", interpolation="nearest")
            axes[r, c].set_title(f"d≤{band}" if (r == 0 and band < 625) else ("自由" if r == 0 else ""), fontsize=9)
            axes[r, c].text(0.02, 0.02, f"wm{info['wm']:+.2f} rad{info['rad']:+.2f}",
                            transform=axes[r, c].transAxes, fontsize=6.5, va="bottom", color="crimson")
            print(f"{name:<10} {band:>5} {info['wm']:>+8.2f} {info['rad']:>+9.2f} {info['oob']:>9.2f} "
                  f"{info['d']:>5} {info['metal']:>6} {info['dust']:>3}")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("sm_invert gallery — SM 梯度反傳（預測值,未經 HFSS;遠帶=SM 自信度探測）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\ngallery → {out}\n⚠ 全部是 SM 預測值——adversarial 判定要靠 HFSS（R29 G 臂）。")


def main():
    ap = argparse.ArgumentParser(description="SM 梯度反傳生成器（零 HFSS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sm", default="sm_reanchor30.pth")
    r.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    r.add_argument("--steps", type=int, default=400)
    r.add_argument("--lr", type=float, default=0.08)
    r.add_argument("--w-rad", type=float, default=1.0, dest="w_rad")
    r.add_argument("--w-oob", type=float, default=0.1, dest="w_oob")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=os.path.join(REPO, "tmp", "invert_gallery.png"))
    r.set_defaults(fn=run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
