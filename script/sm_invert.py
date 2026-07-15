# -*- coding: utf-8 -*-
"""script/sm_invert.py — 梯度反傳生成器（Ricky 2026-07-14 核准;G 臂/R29 主力,零 HFSS）。

固定 SM（可微 MLP）＋rad 頭,把 pattern 當變數反傳——找 SM「認為」滿足 S11/Gain/rad 的 pattern。
歷史教訓內建：
  ①距離分層 trust-region（terrain 定案:SM 局部可信,遠=重抽）——從錨出發,分帶罰距離;
  ②adversarial 風險=特徵不是 bug——遠帶樣本=SM 過度自信點,HFSS 量回來=主動學習極品（反自餵）;
  ③straight-through 二值化（前向硬二值、反向 sigmoid 梯度）,終產物過可製造統計（報告,不強制）;
  ④承重圖知情約束（R28 §4）:half 手術帶凍結命脈塊（t07h {4,7}/p00h {3,6}——ablate/halve 皆崩）。
用法（開發機,零 HFSS）:
  python -m script.sm_invert run [--sm ...] [--steps 400]        # 5 錨×4 帶 gallery 圖+預測表
  python -m script.sm_invert gen --out-dir tmp/invert_stage_r29b1  # G 臂候選 76 筆(四帶 mix)→staging
候選進批次線一律走 select-r29 讀 staging → check-dup → HFSS（R29 G 臂）。
"""
import argparse
import json
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

#? 起點錨（名, pid）——gallery 用;gen 的帶配置見 gen()
ANCHORS = [("margin王", "s28b3_005_a024"),
           ("x00", "x00_c21k2"),
           ("p00_half", "n27b1_018_p00"),
           ("t07_half", "n27b1_020_t07"),
           ("random", None)]
BANDS = (10, 25, 60, 625)                                 # 625=自由（無距離罰）
#? 命脈塊凍結（R28 承重圖真值:ablate/halve 皆 −7~−10 的塊;塊 id=高斯 σ0.8×0.6 label 順序）
FREEZE_BLOCKS = {"n27b1_020_t07": (4, 7), "n27b1_018_p00": (3, 6)}


def _find(pid):
    for fol in _all_input_folders():
        f = DATASET_PATH.joinpath(fol, pid + ".pt")
        if f.exists():
            return np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
    raise SystemExit(f"找不到 {pid}")


def _freeze_mask(p0, blocks):
    """命脈塊 bool mask（與 dedust._skeleton 同口徑,label 編號一致）。"""
    from scipy.ndimage import gaussian_filter, label
    dens = gaussian_filter(p0.astype(float), 0.8, mode="constant")
    lab, _ = label(dens > 0.6, structure=np.ones((3, 3), dtype=bool))
    m = np.zeros_like(p0)
    for g in blocks:
        m |= (lab == g)
    return m


class Inverter:
    """SM+rad 頭載入一次,invert() 可重複呼叫（run 的 gallery 與 gen 的量產共用）。"""

    def __init__(self, sm_name, rad_head_name):
        cfg = load_config(DEFAULT_CFG)
        setup_responses(cfg)
        self.labels = PORT_SPECS[cfg.port]["labels"]
        self.n_pts = sum(cfg.targets[self.labels[0]]["width"])
        self.cfg = cfg
        cache = os.path.join(REPO, "tmp", "sm_invert")
        os.makedirs(cache, exist_ok=True)
        self.sm = SURROGATES["mlp"](cache, 25 * 25, (len(self.labels), self.n_pts))
        self.sm.pre_load_model(DATASET_PATH.joinpath(sm_name), strict=True)
        self.sm.model.eval()
        for p_ in self.sm.model.parameters():
            p_.requires_grad_(False)
        import torch.nn as nn
        _rh = torch.load(str(DATASET_PATH.joinpath(rad_head_name)), weights_only=False)
        self.radnet = nn.Sequential(nn.Linear(625, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 2 * _rh["K"]))
        self.radnet.load_state_dict(_rh["state"])
        self.radnet.eval()
        for p_ in self.radnet.parameters():
            p_.requires_grad_(False)
        self.K = _rh["K"]
        th = np.asarray(_rh["theta"], float)
        phi = np.pi * (th - th.min()) / (th.max() - th.min())
        self.B = torch.tensor(np.cos(np.arange(self.K).reshape(-1, 1) * phi.reshape(1, -1)),
                              dtype=torch.float32)
        self.win = torch.tensor(np.abs(th) <= 45.0)
        self.i0 = int(np.argmin(np.abs(th)))
        #? 可微 wm（官方 worst_margin 內部 float() 斷梯度——工具自帶 torch 版,定義逐字對齊 losses.py）
        self._bands = []
        for i, lab in enumerate(self.labels):
            t = cfg.targets[lab]
            w = t["width"]
            self._bands.append((i, w[0] + w[1], w[0] + w[1] + w[2], float(t["center"]), t["method"]))
        self._far = list(range(4)) + list(range(self.n_pts - 4, self.n_pts))

    def wm_torch(self, pred):
        pr = pred.reshape(len(self.labels), self.n_pts)
        ms = [(c - pr[i][a:b].max()) if m == "low" else (pr[i][a:b].min() - c)
              for i, a, b, c, m in self._bands]
        return torch.min(torch.stack(ms))

    def invert(self, p0, band, rng, steps=400, lr=0.08, w_rad=1.0, w_oob=0.1,
               oob_target=8.0, freeze=None, jitter=0.0):
        """從 p0（None=隨機 init）出發、距離帶 band 內反傳。回 (pattern, 診斷 dict)。
        freeze=bool 625 mask——命脈塊釘死原值（承重圖知情約束）;
        jitter>0=theta 初始加噪（量產同錨多樣化;seed 定=可重現）。"""
        init = (np.where(p0.reshape(-1), 3.0, -3.0) if p0 is not None else rng.normal(0, 1, 625))
        if jitter > 0:
            init = init + rng.normal(0, jitter, 625)
        theta = torch.tensor(init, dtype=torch.float32, requires_grad=True)
        p0t = torch.tensor((p0.reshape(-1) if p0 is not None else np.zeros(625)) > 0.5,
                           dtype=torch.float32)
        fidx = np.flatnonzero(freeze.reshape(-1)) if freeze is not None else None
        fval = None
        if fidx is not None and p0 is not None:
            fval = torch.tensor(np.where(p0.reshape(-1)[fidx], 6.0, -6.0), dtype=torch.float32)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            q = torch.sigmoid(theta)
            qb = (q > 0.5).float()
            q_st = qb + q - q.detach()                       # straight-through
            pred = self.sm.model(q_st)
            wm = self.wm_torch(pred)
            fits = (self.radnet(q_st).reshape(2, self.K) @ self.B)
            rad_m = torch.min(torch.stack([f[self.win].min() - (f[self.i0] - 3.0) for f in fits]))
            pr = pred.reshape(len(self.labels), self.n_pts)
            oob = pr[1][self._far].max() - pr[0][self._far].min()
            d = (q_st - p0t).abs().sum()
            loss = (torch.relu(0.3 - wm) + torch.relu(0.3 - rad_m) * w_rad
                    + torch.relu(oob - oob_target) * w_oob
                    + torch.relu(d - band) * 0.5)
            loss.backward()
            opt.step()
            with torch.no_grad():                            # feed 區＋命脈塊釘死
                theta.data[FEED[0] * 25 + FEED[1]] = 6.0
                if fval is not None:
                    theta.data[fidx] = fval
        with torch.no_grad():
            qf = (torch.sigmoid(theta) > 0.5).numpy().reshape(25, 25)
            qf[FEED] = True
            q_t = torch.tensor(qf.reshape(-1), dtype=torch.float32)
            pred = self.sm.model(q_t)
            wm, _ = worst_margin(pred, self.labels, self.cfg.targets)
            fits = (self.radnet(q_t).reshape(2, self.K) @ self.B)
            rad_m = min(float(f[self.win].min() - (f[self.i0] - 3.0)) for f in fits)
            pr = pred.reshape(len(self.labels), self.n_pts)
            oob = float(pr[1][self._far].max() - pr[0][self._far].min())
            st = piece_stats(qf)
            dd = int(np.sum(qf != p0.reshape(25, 25))) if p0 is not None else -1
        return qf, dict(wm=round(float(wm), 2), rad=round(rad_m, 2), oob=round(oob, 2),
                        d=dd, metal=st["metal_px"], dust=st["n_1px"])


def run(args):
    inv = Inverter(args.sm, args.rad_head)
    rng = np.random.default_rng(args.seed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
    fig, axes = plt.subplots(len(ANCHORS), len(BANDS) + 1,
                             figsize=(2.1 * (len(BANDS) + 1), 2.1 * len(ANCHORS)))
    print(f"{'起點':<10} {'帶':>5} {'pred_wm':>8} {'pred_rad':>9} {'pred_oob':>9} {'d':>5} {'metal':>6} {'塵':>3}")
    for r, (name, pid) in enumerate(ANCHORS):
        p0 = _find(pid) if pid else None
        if p0 is not None:
            axes[r, 0].imshow(p0, cmap="gray_r", interpolation="nearest")
        axes[r, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        axes[r, 0].set_title("錨" if r == 0 else "", fontsize=9)
        for c, band in enumerate(BANDS, start=1):
            qf, info = inv.invert(p0 if p0 is not None else rng.random((25, 25)) > 0.5, band, rng,
                                  steps=args.steps, lr=args.lr, w_rad=args.w_rad, w_oob=args.w_oob)
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\ngallery → {args.out}\n⚠ 全部是 SM 預測值——adversarial 判定要靠 HFSS。")


def gen(args):
    """G 臂候選量產（R29;Ricky 拍板配額）——四帶 mix → staging 夾（本地）,select-r29 讀入:
      free   隨機 init 自由帶（多樣性主力）
      surg   half 手術帶 d≤60＋命脈塊凍結（承重圖知情約束;rad 目標）
      champ  冠軍中帶 d≤25（margin 王/前任交替）
      oobp   超規格帶外期望（oob 目標 ≤--oob-push,w_oob ×5=低側資料泵;half+可用帶外王混錨）
    每筆記 band/anchor/pred_*——HFSS 回來後 pred vs realized=各帶 adversarial 率=SM 盲區地圖。"""
    inv = Inverter(args.sm, args.rad_head)
    rng = np.random.default_rng(args.seed)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    surg_anchors = [("n27b1_020_t07", "t07h"), ("n27b1_018_p00", "p00h")]
    champ_anchors = [("s28b3_005_a024", "king"), ("m23b4_030_r3_001", "exking")]
    oob_anchors = [("m24b2_015_o1_029_vg033", "uoob"), ("n27b1_020_t07", "t07h"),
                   ("n27b1_018_p00", "p00h")]
    pats = {pid: _find(pid) for pid, _ in surg_anchors + champ_anchors + oob_anchors}
    fmasks = {pid: _freeze_mask(pats[pid], FREEZE_BLOCKS[pid]) for pid in FREEZE_BLOCKS
              if pid in pats}
    plan = ([("free", None, 625, {}) for _ in range(args.n_free)]
            + [("surg", surg_anchors[k % 2], 60,
                {"freeze": True}) for k in range(args.n_surg)]
            + [("champ", champ_anchors[k % 2], 25, {}) for k in range(args.n_champ)]
            + [("oobp", oob_anchors[k % 3], 60,
                {"oob_target": float(args.oob_push), "w_oob": 0.5}) for k in range(args.n_oob)])
    manifest, seen = [], set()
    n_try = 0
    for band_name, anc, band, opts in plan:
        made = False
        while not made and n_try < len(plan) * 8:
            n_try += 1
            if anc is None:
                p0, aname = (rng.random((25, 25)) > float(rng.uniform(0.35, 0.65))), "rand"
                base = None
            else:
                pid, aname = anc
                p0, base = pats[pid], pats[pid]
            fz = fmasks.get(anc[0]) if (anc and opts.get("freeze")) else None
            qf, info = inv.invert(base if base is not None else p0, band, rng,
                                  steps=args.steps, lr=args.lr, w_rad=args.w_rad,
                                  w_oob=opts.get("w_oob", args.w_oob),
                                  oob_target=opts.get("oob_target", 8.0), freeze=fz,
                                  jitter=0.8 if anc is not None else 0.0)
            kb = qf.tobytes()
            if kb in seen or not (150 <= info["metal"] <= 560):
                continue
            seen.add(kb)
            k = len(manifest)
            pid_out = f"stage_{k:03d}"
            torch.save(torch.tensor(qf, dtype=torch.float32), os.path.join(out, pid_out + ".pt"))
            manifest.append(dict(id=pid_out, band=band_name, anchor=aname, dlim=band, **info))
            made = True
    with open(os.path.join(out, "staging_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    from collections import Counter
    cnt = Counter(m["band"] for m in manifest)
    print(f"staging → {out}: {len(manifest)} 筆 {dict(cnt)}")
    print("⚠ 預測值未經 HFSS;進批次線走 select-r29 --gstage（查重/打分/編 id 在那邊）。")


def main():
    ap = argparse.ArgumentParser(description="SM 梯度反傳生成器（零 HFSS）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(s):
        s.add_argument("--sm", default="sm_reanchor33.pth")
        s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
        s.add_argument("--steps", type=int, default=400)
        s.add_argument("--lr", type=float, default=0.08)
        s.add_argument("--w-rad", type=float, default=1.0, dest="w_rad")
        s.add_argument("--w-oob", type=float, default=0.1, dest="w_oob")
        s.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="5 錨×4 帶 gallery（可信半徑可視化）")
    _common(r)
    r.add_argument("--out", default=os.path.join(REPO, "tmp", "invert_gallery.png"))
    r.set_defaults(fn=run)

    g = sub.add_parser("gen", help="G 臂候選量產（四帶 mix → staging;R29）")
    _common(g)
    g.add_argument("--out-dir", default=os.path.join(REPO, "tmp", "invert_stage"), dest="out_dir")
    g.add_argument("--n-free", type=int, default=28, dest="n_free")
    g.add_argument("--n-surg", type=int, default=24, dest="n_surg")
    g.add_argument("--n-champ", type=int, default=12, dest="n_champ")
    g.add_argument("--n-oob", type=int, default=12, dest="n_oob")
    g.add_argument("--oob-push", type=float, default=6.0, dest="oob_push",
                   help="oobp 帶的超規格帶外目標（低側資料泵）")
    g.set_defaults(fn=gen)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
