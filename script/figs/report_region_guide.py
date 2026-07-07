# -*- coding: utf-8 -*-
"""report_region_guide.py — 冠軍「區調整指南」（docs/report/assets/region_guide.png）。
把承重圖升級成可操作的設計地圖:每個冠軍一列 × 四欄——
  ①帶內 Δwm（紅=遮了就崩=承重勿動 / 藍=低成本可編輯）
  ②帶外 Δoob_bad（藍=遮了帶外更乾淨=帶外元凶,可底緣精修）
  ③rad Δ旋鈕（遮了 rad 變好/變差=輻射旋鈕）
  ④綜合建議（四類:承重勿動/可調低成本/帶外可精修/rad旋鈕）
資料:dedust_occl2 (c21/a15) 5×5 遮蔽掃描,oob 從響應曲線回算。零 HFSS。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, GRID, SURF, RED, ORANGE, GREEN, PURPLE, plt, save)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402
from script.dedust import oob_metrics  # noqa: E402

FEED = (24, 12)
CHAMPS = [("c21_sm", "c21（旗艦, wm +0.20）"), ("a15_k4", "a15（rad 王, wm +0.03）")]


def load_maps():
    man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_occl2_input", "manifest.json")), encoding="utf-8"))}
    res = json.load(open(str(DATASET_PATH.joinpath("dedust_occl2", "results.json")), encoding="utf-8"))
    rv = json.load(open(str(DATASET_PATH.joinpath("dedust_ref2v", "results.json")), encoding="utf-8"))
    resp = {}
    for sname in ("dedust_occl2", "dedust_ref2v"):
        st = SampleStore(DATASET_PATH.joinpath(sname), verbose=False)
        for k in range(len(st)):
            x, y = st[k]
            resp[(np.asarray(x).reshape(-1) > 0.5).tobytes()] = np.asarray(y).reshape(2, -1)
    out = {}
    for cid, _lab in CHAMPS:
        pat = np.asarray(torch.load(str(DATASET_PATH.joinpath("dedust_ref2_input", f"{cid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
        b_wm, b_rad = rv[cid]["wm"][2], rv[cid]["rad_margin"]
        b_ob = oob_metrics(resp[pat.reshape(-1).tobytes()])["oob_bad"]
        W = np.full((5, 5), np.nan); O = np.full((5, 5), np.nan); Rr = np.full((5, 5), np.nan)
        for m in man.values():
            if m["source_id"] != cid or m["id"] not in res or "wm" not in res[m["id"]]:
                continue
            br, bc = m["block"]
            r = res[m["id"]]
            W[br, bc] = r["wm"][2] - b_wm
            Rr[br, bc] = r.get("rad_margin", b_rad) - b_rad
            key = (np.asarray(torch.load(str(DATASET_PATH.joinpath("dedust_occl2_input", f"{m['id']}.pt")), weights_only=True)).reshape(-1) > 0.5).tobytes()
            if key in resp:
                O[br, bc] = oob_metrics(resp[key])["oob_bad"] - b_ob
        out[cid] = (pat, W, O, Rr)
    return out


def _overlay(ax, pat, H, cmap, vlim, title):
    ax.imshow(pat.astype(int), cmap=ListedColormap([SURF, "#cdd8e8"]), vmin=0, vmax=1, origin="upper", interpolation="nearest")
    img = np.kron(H, np.ones((5, 5)))
    im = ax.imshow(img, cmap=cmap, vmin=-vlim, vmax=vlim, alpha=0.74, origin="upper", interpolation="nearest")
    for k in range(6):
        ax.axhline(k * 5 - 0.5, color=SURF, lw=1.0)
        ax.axvline(k * 5 - 0.5, color=SURF, lw=1.0)
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=42, color=GREEN, zorder=5, edgecolor=SURF, lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, color=INK, fontsize=10)
    return im


def classify(W, O, Rr):
    """每塊分四類（優先序,帶外元凶優先浮現——它們多與承重重疊,只做邊緣精修不移除）:
    3=帶外精修區 / 2=承重勿動 / 1=rad旋鈕 / 0=可調低成本 / nan=無金屬。"""
    C = np.full((5, 5), np.nan)
    for r in range(5):
        for c in range(5):
            if np.isnan(W[r, c]):
                continue
            if O[r, c] < -1.2:
                C[r, c] = 3                          # 帶外精修區（遮了帶外更乾淨→邊緣微調而非移除）
            elif W[r, c] < -3.0:
                C[r, c] = 2                          # 承重勿動
            elif abs(Rr[r, c]) > 0.5:
                C[r, c] = 1                          # rad 旋鈕
            else:
                C[r, c] = 0                          # 可調低成本
    return C


def main():
    data = load_maps()
    fig, axes = plt.subplots(len(CHAMPS), 4, figsize=(13.0, 6.6))
    CAT = ListedColormap([GREEN, PURPLE, RED, ORANGE])
    for row, (cid, lab) in enumerate(CHAMPS):
        pat, W, O, Rr = data[cid]
        _overlay(axes[row][0], pat, W, "RdBu", 12, "① 帶內 Δwm（紅=承重勿動）")
        _overlay(axes[row][1], pat, O, "RdBu_r", 6, "② 帶外 Δoob（藍=帶外元凶,可壓）")
        _overlay(axes[row][2], pat, Rr, "PuOr", 1.2, "③ Δrad（輻射旋鈕）")
        ax = axes[row][3]
        C = classify(W, O, Rr)
        ax.imshow(pat.astype(int), cmap=ListedColormap([SURF, "#eeeeee"]), vmin=0, vmax=1, origin="upper", interpolation="nearest")
        ax.imshow(np.kron(C, np.ones((5, 5))), cmap=CAT, vmin=0, vmax=3, alpha=0.82, origin="upper", interpolation="nearest")
        for k in range(6):
            ax.axhline(k * 5 - 0.5, color=SURF, lw=1.0); ax.axvline(k * 5 - 0.5, color=SURF, lw=1.0)
        ax.scatter([FEED[1]], [FEED[0]], marker="^", s=42, color=INK, zorder=5, edgecolor=SURF, lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title("④ 綜合調整建議", color=INK, fontsize=10)
        axes[row][0].set_ylabel(lab, color=INK, fontsize=11, labelpad=8)
    fig.legend(handles=[Patch(color=GREEN, label="可調低成本（盲掃/知情編輯用）"),
                        Patch(color=RED, label="承重勿動（遮了 wm 崩 >3dB）"),
                        Patch(color=ORANGE, label="帶外精修區（帶外元凶,多與承重重疊→只邊緣微調）"),
                        Patch(color=PURPLE, label="rad 旋鈕（遮了 |Δrad|>0.5）")],
               loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("冠軍區調整指南：一張圖看「哪塊能動、哪塊不能、動哪塊壓帶外」（5×5 遮蔽掃描回算,零 HFSS）",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98], h_pad=1.6)
    save(fig, "region_guide.png")


if __name__ == "__main__":
    main()
