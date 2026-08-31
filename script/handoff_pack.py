# -*- coding: utf-8 -*-
"""
script/handoff_pack.py — 送板交付包：從「帶菱形橋且已量測」的樣本挑家族代表，出兩份 PDF 目錄。

背景（2026-08-31，學長鄭國宏要安排送板）：要「好的 pattern」＋「不同譜系各挑幾張」＋「模擬效果
比較」，天線(single) 與濾波器(dual) 各一份 PDF，PDF 上的交付檔名讓外部的 HFSS 模擬檔對得起來。

判準（發車前寫死）：
  · 池 = `<store>_input/hfss_setup.json` 有 `diag_bridge_w > 0`（＝可製造幾何，模擬器在每個真對角
    接點自動放 45° 菱形橋）。負值是挖空槽（R54 slot 臂），不是橋，排除。
  · single 尺 = `worst_margin` = results 的 `wm[2]`；**夠好 = 過標（≥ 0）**。
  · dual  尺 = `wm_mfg` = min(m1+2, m2+2, m3, m4+5)（學長 2026-08-12 裁定的規格 v2：S11/S22 帶內
    −10、S21 阻帶 −15、帶外退場；`docs/records_dual.json` 的 `wm_r2` 註記）。回驗三個公證值 3/3
    吻合：d70b1_A_11 −5.90、smp073_d_040 −2.39、smp050_L_011 −2.43。
    **夠好 = 現任王 − `--dual-window` dB 之內**（dual 全池零過標，只能取「接近最好表現」）。
  · 家族 = `dedust.cluster_families`（greedy leader，Hamming ≤ `--max-dist` 併同族；呼叫端已按分數
    降冪 → 家族代表＝族內最好者）。
  · 每家族取前 `--top` 名。
  · 同一 pattern 會散在多個 store（原批＋公證重測）→ **依 pattern 內容去重**，只留分數最好那筆，
    否則「前五名」會是同一張板重複五次。

重用（不平行重建）：`dedust.cluster_families` 家族聚類、`single_port.diag_bridge_sites` 橋位
（＝模擬器實際用的那份，含碰撞縮排/跳過規則，所以圖上畫的就是 HFSS 會建的）、`SampleStore`
取響應（檔名＝`fingerprint(x, y)`，無法從 pattern 反推，只能逐筆比對，同 `figs/r9_champ_curves.py`）。

用法（開發機，需掛 NAS）：
    python -m script.handoff_pack index                  # 掃 NAS 建索引（慢，會快取）
    python -m script.handoff_pack report                 # 只看家族數/頁數，不畫圖
    python -m script.handoff_pack report --max-dist 60   # 換家族切法看看
    python -m script.handoff_pack pdf --port single

索引快取 `tmp/handoff_index.json`；PDF 出到 `tmp/handoff/`（均 gitignore）。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import DATASET_PATH
from antenna.utils.store import SampleStore
from antenna.patch.patch_simulator.single_port import diag_bridge_sites
from script.dedust import cluster_families

INDEX_PATH = os.path.join("tmp", "handoff_index.json")
OUT_DIR = os.path.join("tmp", "handoff")
PIXEL_MM = 0.2                      # 25×25 域的格距（同 dedust select-diagbridge）
#! R54 規則②「控 n_sites<17」只適用 single：那條規則量的是「沒帶橋選解 → 事後加橋」的稅
#  （≥17 座稅中位 −0.37/帶內存活 38.6%，門檻型懸崖，round-54 §170）。dual 可製造代（R70 起）
#  是 hfss_setup 一開始就帶 0.075 橋量測，稅已內含在 wm_mfg，再套一次是誤用——實測 dual 候選
#  橋座數 45–75（中位 66），套下去 1721 筆全滅。dual 改為「不濾、但每頁標出座數給學長判蝕刻」。
SITES_RULE = {"single": 17, "dual": 0}          # 0 = 不濾
FREQ_GHZ = 24.0 + 0.5 * np.arange(17)   # HFSS 掃頻 24–32GHz/17 點，中心 28GHz（single_port.py:552）


# ---------------------------------------------------------------- 索引

def build_index() -> list:
    """掃 NAS 所有帶菱形橋的 store，收 results.json 的判讀指標。只讀 NAS，不寫。"""
    root = str(DATASET_PATH)
    rows = []
    for d in sorted(os.listdir(root)):
        if not d.endswith("_input"):
            continue
        setup = os.path.join(root, d, "hfss_setup.json")
        if not os.path.exists(setup):
            continue
        try:
            w = json.load(open(setup, encoding="utf-8")).get("diag_bridge_w")
        except Exception:
            continue
        if not w or w < 0:                      # 負 = 挖空槽，不是橋
            continue
        store = d[:-6]
        rj = os.path.join(root, store, "results.json")
        if not os.path.exists(rj):
            continue
        try:
            res = json.load(open(rj, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(res, dict):
            continue
        for pid, r in res.items():
            if not isinstance(r, dict):
                continue
            if "m1" in r:                                       # dual
                m = [r.get(f"m{i}") for i in range(1, 7)]
                if any(v is None for v in m[:4]):
                    continue
                rows.append(dict(store=store, id=pid, bridge_w=w, port="dual", m=m,
                                 score=min(m[0] + 2, m[1] + 2, m[2], m[3] + 5),
                                 energy_max=r.get("energy_max"), geom=r.get("geom")))
            elif isinstance(r.get("wm"), list) and len(r["wm"]) >= 3:
                rows.append(dict(store=store, id=pid, bridge_w=w, port="single",
                                 score=r["wm"][2], wm=r["wm"][:3],
                                 rad_margin=r.get("rad_margin"), sel=r.get("sel")))
    return rows


def load_index(rebuild=False) -> list:
    if not rebuild and os.path.exists(INDEX_PATH):
        return json.load(open(INDEX_PATH, encoding="utf-8"))
    rows = build_index()
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    json.dump(rows, open(INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return rows


# ---------------------------------------------------------------- 挑選

PCACHE_PATH = os.path.join("tmp", "handoff_patterns.npz")
_PCACHE, _PDIRTY = None, False


def _pattern(store, pid):
    """pattern 讀取（NAS 很慢，快取到本機；工具會反覆調參數重跑）。"""
    global _PCACHE, _PDIRTY
    if _PCACHE is None:
        _PCACHE = dict(np.load(PCACHE_PATH)) if os.path.exists(PCACHE_PATH) else {}
    key = f"{store}|{pid}"
    if key in _PCACHE:
        return _PCACHE[key]
    f = DATASET_PATH.joinpath(f"{store}_input", f"{pid}.pt")
    if not f.exists():
        return None
    m = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
    _PCACHE[key] = m
    _PDIRTY = True
    return m


def save_pattern_cache():
    if _PDIRTY and _PCACHE:
        os.makedirs(os.path.dirname(PCACHE_PATH), exist_ok=True)
        np.savez_compressed(PCACHE_PATH, **_PCACHE)


SENIOR_RE = re.compile(r"^t\d+_top")


def senior_picks(rows, port, top, verbose=True):
    """學長池頂系 `t*_top` 指定納入——交付對象就是學長本人，他的招牌（論文圖 4-4 = `t07_top`,
    見 [[project_senior_showcase_vs_f2]]）一定要在。帶橋後最高僅 +0.19、過不了 R54 規則① 的
    +0.3 門檻，所以**不與家族制競爭**，獨立成 S 區並照實標分數。dual 無此系。"""
    if port != "single":
        return []
    cand = sorted([r for r in rows if r["port"] == port and SENIOR_RE.match(r["id"])],
                  key=lambda r: -r["score"])
    out, seen = [], set()
    for r in cand:
        m = _pattern(r["store"], r["id"])
        if m is None:
            continue
        k = m.tobytes()
        if k in seen:
            continue
        seen.add(k)
        sites, _ = diag_bridge_sites(m, r["bridge_w"], PIXEL_MM)
        out.append(dict(r, n_sites=len(sites), sites=sites, name=r["id"], aliases=[],
                        family=-1, rank=len(out) + 1, mat=m))
        if len(out) >= top:
            break
    if verbose and out:
        print(f"[{port}] 學長池頂系指定納入 {len(out)} 張（{out[0]['score']:+.2f} … "
              f"{out[-1]['score']:+.2f}；未達 +0.3 門檻，照實標示）")
    return out


def select(rows, port, max_dist, top, dual_window, buffer=0.3, max_sites=17, verbose=True):
    cand = [r for r in rows if r["port"] == port]
    if not cand:
        return []
    best = max(r["score"] for r in cand)
    thr = buffer if port == "single" else best - dual_window
    cand = sorted([r for r in cand if r["score"] >= thr], key=lambda r: -r["score"])
    if verbose:
        bar = (f"≥ {buffer}（R54 規則①：過標還要留 buffer，合格線附近 42.4% 會穿越）"
               if port == "single" else f"≥ 王({best:+.2f}) − {dual_window} = {thr:+.2f}")
        print(f"[{port}] 門檻 {bar} → 候選 {len(cand)} 筆")

    seen, uniq, mats, n_dup, n_sites_out = {}, [], [], 0, 0
    for r in cand:                              # 已按分數降冪 → 重複 pattern 留最好那筆
        m = _pattern(r["store"], r["id"])
        if m is None:
            continue
        key = m.tobytes()
        if key in seen:                         # 同 pattern 的公證重測：併為別名，不重複出頁
            n_dup += 1
            kept = seen[key]
            kept["aliases"].append(f"{r['store']}:{r['id']}")
            #? 公證店把 pattern 改名成 r00_rep 之類，對學長沒意義 → 交付名優先取原批的有意義 id
            #  （量測來源 store:id 不動，響應要從那裡撈；別名另外列，可追溯）
            if _anonymous(kept["name"]) and not _anonymous(r["id"]):
                kept["name"] = r["id"]
            continue
        sites, _ = diag_bridge_sites(m, r["bridge_w"], PIXEL_MM)
        if max_sites and len(sites) >= max_sites:   # R54 規則②：n_sites<17 是門檻型懸崖
            n_sites_out += 1
            continue
        r = dict(r, n_sites=len(sites), sites=sites, name=r["id"], aliases=[])
        seen[key] = r
        uniq.append(r)
        mats.append(m)
    if verbose:
        rule2 = (f"、n_sites≥{max_sites} 剔除 −{n_sites_out}（R54 規則②）" if max_sites
                 else "、n_sites 不濾（規則②不適用 dual，見檔頭）")
        print(f"[{port}] 去重 −{n_dup}（重測/重複）{rule2} → 剩 {len(uniq)} 筆")
    if not uniq:
        return []

    labels = cluster_families(mats, max_dist=max_dist)
    picks, count = [], {}
    for r, m, f in zip(uniq, mats, labels):
        f = int(f)
        if count.get(f, 0) >= top:
            continue
        count[f] = count.get(f, 0) + 1
        picks.append(dict(r, family=f, rank=count[f], mat=m))
    if verbose:
        print(f"[{port}] 家族 {len(set(int(x) for x in labels))} 個（Hamming ≤ {max_dist} 併族）"
              f" → 每族取前 {top} → 交付 {len(picks)} 張")
    return picks


# ---------------------------------------------------------------- 響應曲線

def response_map(store_name):
    """pattern bytes → y。store 檔名＝fingerprint(x, y) 無法從 pattern 反推，只能逐筆比對。"""
    out = {}
    d = DATASET_PATH.joinpath(store_name)
    if not d.is_dir():
        return out
    store = SampleStore(d, verbose=False)
    for i in range(len(store)):
        try:
            x, y = store[i]
        except Exception:
            continue
        out[(np.asarray(x).reshape(25, 25) > 0.5).tobytes()] = np.asarray(y)
    return out


# ---------------------------------------------------------------- PDF（gallery 版式）
#! 版式與樣式**全部沿用既有共用模組** `figs/report_r1r10_style`（`style_ax`/`show_pattern`/
#  `polar_rad_ax`）＋ `figs/report_diversity.py` 的一列版式 **[pattern | S11 | Gain | rad 極座標]**
#  ——這是專案既有的固定表示方式，不另立一套（2026-08-31 Ricky 連兩次指正：先找既有的再動手）。
#! 方向鐵則（[[reference_pattern_render_convention]]）：`show_pattern` 已用 `origin="upper"`
#  ＝饋線邊在圖下緣＋feed 三角；橋位**直接呼叫 `diag_bridge_sites`**（嚴禁自寫接點偵測），
#  角點 (cx,cy)mm → `scatter(cy/pmm−0.5, cx/pmm−0.5)`。
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, GRID, SURF, RED, DBLUE, GREEN, ORANGE, AQUA, plt, style_ax, show_pattern, polar_rad_ax)

ROWS_PER_PAGE = 5
# 規格頻帶（single: configs/single_r5_explore.yaml；dual: 學長 2026-08-12 裁定的規格 v2）
S_BAND = (26.5, 29.5)                  # single 帶內 width [5,0,7,0,5] → idx 5-11
D_IN = (26.5, 30.0)                    # dual S11/S22 帶內 idx 5-11
D_PASS = (25.5, 30.5)                  # dual S21 通帶 idx 3-13
D_STOP = [(24.0, 25.0), (31.0, 32.0)]  # dual S21 阻帶 idx 0-2 / 14-16
#! 未達標判定用**頻點索引**（與 `worst_margin`/`worst_margin_dual` 的 mask 同一把尺，不用浮點比對）
S_IN_IDX = list(range(5, 12))                      # single 帶內
D_IN_IDX = list(range(5, 12))                      # dual S11/S22 帶內
D_PASS_IDX = list(range(3, 14))                    # dual S21 通帶
D_STOP_IDX = list(range(0, 3)) + list(range(14, 17))   # dual S21 阻帶


def _mark_fail(ax, f, y, idxs, thr, low):
    """圈出規格區間內**未達標的頻點**（空心紅圈；沿用 `figs/data_map.py:73` 標子集的畫法）。
    low=True＝規格要求「低於 thr」（S11/S22 帶內、S21 阻帶）；False＝要求「高於」。回傳未達點數。"""
    idxs = [i for i in idxs if i < len(f)]
    bad = [i for i in idxs if (y[i] > thr if low else y[i] < thr)]
    if bad:
        ax.scatter(f[bad], y[bad], s=78, facecolors="none", edgecolors=RED, lw=1.7, zorder=6)
    return len(bad)


def _anonymous(pid) -> bool:
    """公證重測店把 pattern 統一改名（r00_rep / *_rep）——這種 id 對學長沒有辨識意義。"""
    return pid.endswith("_rep") or pid.startswith("r00_") or pid.startswith("rep")


def deliver_name(port, p):
    tag = "ANT" if port == "single" else "FLT"
    fam = "S" if p["family"] < 0 else f"F{p['family']:02d}"      # S = 學長池頂系（指定納入）
    return f"{tag}_{fam}_{p['rank']}_{p.get('name', p['id'])}"


def rad_curves(store, pid):
    """原始方向圖曲線：`<store>/rad/<id>.pt` = dict{theta(181), phi0(181), phi90(181)}（dual 無）。"""
    f = DATASET_PATH.joinpath(store, "rad", f"{pid}.pt")
    if not f.exists():
        return None
    try:
        d = torch.load(str(f), weights_only=True)
        return np.asarray(d["theta"]), np.asarray(d["phi0"]), np.asarray(d["phi90"])
    except Exception:
        return None


def _curve(ax, f, y, bands, thrs, yl, title, xl="", ylim=None):
    for lo, hi, alpha in bands:
        ax.axvspan(lo, hi, color=GRID, alpha=alpha)
    ax.plot(f, y, color=DBLUE, lw=1.9)
    for val in thrs:
        ax.axhline(val, color=RED, ls=":", lw=1.3)
    ax.set_xlim(24, 32)
    ax.set_xticks([24, 26, 28, 30, 32])
    if ylim:                                   # 全域固定刻度（single/dual 同一把尺）
        ax.set_ylim(*ylim)
    style_ax(ax, xl, yl, title, tfs=9.6)


_RMAPS = {}


def _resp(p):
    """取這筆的響應曲線（store→pattern bytes 的對照表跨 port 共用快取）。"""
    if p["store"] not in _RMAPS:
        _RMAPS[p["store"]] = response_map(p["store"])
    return _RMAPS[p["store"]].get(p["mat"].tobytes())


def axis_ranges(all_picks):
    """掃過**全部要出圖的曲線**取一次全域軸範圍 → single/dual 每一張共用同一把尺。
    （Ricky 2026-08-31：不要每列自己 autoscale、看起來像在亂切換。）取 5 的倍數留 1dB 餘裕。"""
    def rng(vals, pad=1.0, step=5):
        if not vals:
            return None
        lo, hi = min(vals) - pad, max(vals) + pad
        #? 回傳 int：polar_rad_ax 內部用 range() 產環,float 會 TypeError
        return (int(np.floor(lo / step) * step), int(np.ceil(hi / step) * step))

    sp, gain, rad = [], [], []
    for p in all_picks:
        y = _resp(p)
        if y is None:
            continue
        y = np.asarray(y)
        if p["port"] == "single":
            sp += [float(y[0].min()), float(y[0].max())]
            gain += [float(y[1].min()), float(y[1].max())]
        else:
            for i in range(y.shape[0]):
                sp += [float(y[i].min()), float(y[i].max())]
        rc = rad_curves(p["store"], p["id"]) if p["port"] == "single" else None
        if rc is not None:
            rad += [float(rc[1].min()), float(rc[1].max()),
                    float(rc[2].min()), float(rc[2].max())]
    out = {"S": rng(sp), "GAIN": rng(gain), "RAD": rng(rad)}
    if out["RAD"]:                    # 方向圖跨距上限 30dB（同 polar_rad_ax 自身預設,太寬會看不清）
        lo, hi = out["RAD"]
        out["RAD"] = (max(lo, hi - 30), hi)
    return out


def render(port, picks, out_path, png_dir=None, ylim=None):
    from matplotlib.backends.backend_pdf import PdfPages

    ylim = ylim or {}
    n_missing = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ratios = [1, 1.25, 1.25, 1.45] if port == "single" else [1, 1.3, 1.3, 1.3]

    with PdfPages(out_path) as pdf:
        for start in range(0, len(picks), ROWS_PER_PAGE):
            chunk = picks[start:start + ROWS_PER_PAGE]
            rows = len(chunk)
            fig_h = 3.15 * rows + 1.9           # +1.9 吋 = 頁首標題 + x 軸標籤 + 兩行頁尾圖說的固定留白
            fig = plt.figure(figsize=(13.6, fig_h))
            gs = fig.add_gridspec(rows, 4, width_ratios=ratios,
                                  left=0.035, right=0.975,
                                  top=1 - 0.80 / fig_h, bottom=1.05 / fig_h,
                                  hspace=0.62, wspace=0.32)
            for r, p in enumerate(chunk):
                y = _resp(p)
                last = (r == rows - 1)

                # --- pattern（＋菱形橋位；show_pattern 已畫 feed 三角、origin=upper）
                ax = fig.add_subplot(gs[r, 0])
                head = (f"{deliver_name(port, p)}\n"
                        f"{'wm' if port == 'single' else 'wm_mfg'}{p['score']:+.2f} dB"
                        f"　菱形橋 {p['n_sites']} 座 @ {p['bridge_w']} mm")
                show_pattern(ax, p["mat"], head, tfs=8.6)
                if port == "dual":                            # dual_feed = 底邊中央 + 頂邊中央
                    ax.scatter([12], [0], marker="v", s=46, color=AQUA, zorder=5,
                               edgecolor=SURF, lw=0.8)
                for sx, sy, _w in p["sites"]:                 # 角點 mm → 像素格線交點
                    ax.plot(sy / PIXEL_MM - 0.5, sx / PIXEL_MM - 0.5, marker="D", ms=3.2,
                            mfc="none", mec=ORANGE, mew=0.9, alpha=0.9)

                if y is None:
                    n_missing += 1
                    for c in (1, 2, 3):
                        a = fig.add_subplot(gs[r, c])
                        a.text(.5, .5, "（找不到響應）", ha="center", va="center", color=INK2)
                        a.set_xticks([]); a.set_yticks([])
                    continue
                y = np.asarray(y)
                f = FREQ_GHZ[:y.shape[1]]

                if port == "single":
                    a1 = fig.add_subplot(gs[r, 1])
                    nb = _mark_fail(a1, f, y[0], S_IN_IDX, -10, True)
                    _curve(a1, f, y[0], [(*S_BAND, 0.45)], [-10], "S11 (dB)",
                           f"S11 — 帶內餘裕 {p['wm'][0]:+.2f} dB" + (f"（未達 {nb} 點）" if nb else ""),
                           "頻率 (GHz)" if last else "", ylim.get("S"))
                    a2 = fig.add_subplot(gs[r, 2])
                    nb = _mark_fail(a2, f, y[1], S_IN_IDX, 4, False)
                    _curve(a2, f, y[1], [(*S_BAND, 0.45)], [4], "Gain (dB)",
                           f"Realized Gain — 帶內餘裕 {p['wm'][1]:+.2f} dB" + (f"（未達 {nb} 點）" if nb else ""),
                           "頻率 (GHz)" if last else "", ylim.get("GAIN"))
                    rc = rad_curves(p["store"], p["id"])
                    if rc is None:
                        a = fig.add_subplot(gs[r, 3])
                        a.text(.5, .5, "（此批無方向圖檔）", ha="center", va="center", color=INK2)
                        a.set_xticks([]); a.set_yticks([])
                    else:
                        th, g0c, g90c = rc
                        axp = fig.add_subplot(gs[r, 3], projection="polar")
                        bi = int(np.abs(th).argmin())
                        polar_rad_ax(axp, th, [(g0c, DBLUE, "φ=0°", 1.7),
                                               (g90c, GREEN, "φ=90°", 1.7)],
                                     window=45, floor_db=3,
                                     g0_ref=max(float(g0c[bi]), float(g90c[bi])),
                                     rmin=(ylim.get("RAD") or (None, None))[0],
                                     rmax=(ylim.get("RAD") or (None, None))[1])
                        rm = p.get("rad_margin")
                        axp.set_title("radiation" + ("" if rm is None else f" 餘裕 {rm:+.2f} dB"),
                                      color=INK, fontsize=9.6, pad=10)
                        if r == 0:
                            axp.legend(fontsize=7.5, loc="lower right",
                                       bbox_to_anchor=(1.18, -0.12), framealpha=.9)
                else:
                    m = p["m"]
                    #! 標題用**規格 v2 四軸** m1′/m2′/m3/m4′（＝ wm_mfg 的組成，records_dual `wm_r2`
                    #  註記的命名）。原始 m1/m2 是對量測 config 的 −12 算的，直接顯示會與畫在
                    #  −10（v2）的紅線/紅圈矛盾——第一列 m1 −0.03 卻沒有任何未達點就是這個坑。
                    m1a, m2a, m4a = m[0] + 2, m[1] + 2, m[3] + 5
                    a1 = fig.add_subplot(gs[r, 1])
                    nb = _mark_fail(a1, f, y[0], D_IN_IDX, -10, True)
                    _curve(a1, f, y[0], [(*D_IN, 0.45)], [-10], "S11 (dB)",
                           f"S11 — m1′ 帶內 {m1a:+.2f} dB" + (f"（未達 {nb} 點）" if nb else ""),
                           "頻率 (GHz)" if last else "", ylim.get("S"))
                    a2 = fig.add_subplot(gs[r, 2])
                    nb = (_mark_fail(a2, f, y[1], D_PASS_IDX, -3, False)
                          + _mark_fail(a2, f, y[1], D_STOP_IDX, -15, True))
                    _curve(a2, f, y[1],
                           [(*D_PASS, 0.45)] + [(lo, hi, 0.22) for lo, hi in D_STOP], [-3, -15],
                           "S21 (dB)",
                           f"S21 — m3 通帶 {m[2]:+.2f} / m4′ 阻帶 {m4a:+.2f} dB"
                           + (f"（未達 {nb} 點）" if nb else ""),
                           "頻率 (GHz)" if last else "", ylim.get("S"))
                    a3 = fig.add_subplot(gs[r, 3])
                    y22 = y[2] if y.shape[0] > 2 else y[0]
                    nb = _mark_fail(a3, f, y22, D_IN_IDX, -10, True)
                    _curve(a3, f, y22, [(*D_IN, 0.45)], [-10], "S22 (dB)",
                           f"S22 — m2′ 帶內 {m2a:+.2f} dB" + (f"（未達 {nb} 點）" if nb else ""),
                           "頻率 (GHz)" if last else "", ylim.get("S"))

            page = start // ROWS_PER_PAGE + 1
            fig.suptitle(f"送板交付目錄 — {'天線 (single-port)' if port == 'single' else '濾波器 (dual-port)'}"
                         f"　第 {page} 頁", color=INK, fontsize=13, y=1 - 0.28 / fig_h)
            note = ("灰帶＝工作頻帶 26.5–29.5 GHz；紅虛線＝門檻（S11 −10 / Gain +4 dB）\n"
                    "極座標＝主波束朝上、金＝±45° 窗、紅虛圈＝峰值−3dB 門檻；"
                    "橘◇＝45° 菱形橋位（HFSS 實際建的幾何）、青三角＝饋點；紅空心圈＝帶內未達標的頻點"
                    if port == "single" else
                    "判準 wm_mfg=min(m1′, m2′, m3, m4′)＝規格 v2 四軸（m1′=m1+2、m2′=m2+2、m4′=m4+5）\n"
                    "灰帶＝S11/S22 帶內 26.5–30 GHz；S21 深灰＝通帶 25.5–30.5、淺灰＝阻帶；"
                    "紅虛線＝門檻；橘◇＝45° 菱形橋位；紅空心圈＝規格區間內未達標的頻點。dual 不輸出遠場，故無方向圖")
            fig.text(0.5, 0.20 / fig_h, note, ha="center", color=INK2, fontsize=8.4)
            pdf.savefig(fig, facecolor=SURF)
            if png_dir:
                os.makedirs(png_dir, exist_ok=True)
                fig.savefig(os.path.join(png_dir, f"page{page:02d}.png"), dpi=120, facecolor=SURF)
            plt.close(fig)
    if n_missing:
        print(f"    ⚠ {n_missing} 筆找不到響應曲線")
    return (len(picks) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE



# ---------------------------------------------------------------- 匯出 dedust 輸入夾

def _src_setup(store):
    """原批的 hfss_setup（求解設定）——重解要用同一組，數字才能與 PDF 對帳。"""
    f = DATASET_PATH.joinpath(f"{store}_input", "hfss_setup.json")
    if not f.exists():
        return {}
    try:
        return json.load(open(str(f), encoding="utf-8"))
    except Exception:
        return {}


def export_inputs(picks_by_port, prefix="handoff", dry=False):
    """交付集 → dedust 輸入夾（正式機 `run` 燒 HFSS 用）。

    · **依 port × hfss_setup 分夾**：setup 是整夾生效的，交付集橫跨多種橋寬/網格設定。
    · **id 直接用交付名**（`ANT_F00_1_...`）→ results.json 與專案檔對照表都能對上 PDF。
    · `kind="repeat"`：蓄意重測已知 pattern，check-dup 豁免集合（`dedust.py:4897`）。
    · `hfss_setup` ＝原批設定 ＋ `keep_project: true`（跑完保留 .aedt，⚠ 要人工清）。
    """
    import shutil
    groups = {}
    for port, picks in picks_by_port.items():
        for p in picks:
            setup = _src_setup(p["store"])
            groups.setdefault((port, json.dumps(setup, sort_keys=True)), []).append(p)

    made = []
    for (port, skey), ps in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        setup = json.loads(skey)
        w = setup.get("diag_bridge_w")
        tag = f"db{int(round(w * 1000)):03d}" if w else "base"
        name = f"{prefix}_{'ant' if port == 'single' else 'flt'}_{tag}"
        d = DATASET_PATH.joinpath(f"{name}_input")
        print(f"  {name}_input ← {len(ps)} 筆（橋 {w} mm，setup {setup}）")
        if dry:
            made.append((name, len(ps)))
            continue
        if d.is_dir() and any(d.glob("*.pt")):
            raise SystemExit(f"{d.name} 已存在且非空——拒寫（防覆寫，同 dedust select-* 慣例）")
        os.makedirs(str(d), exist_ok=True)
        manifest = []
        for p in ps:
            vid = deliver_name(port, p)
            shutil.copyfile(str(DATASET_PATH.joinpath(f"{p['store']}_input", f"{p['id']}.pt")),
                            str(d.joinpath(f"{vid}.pt")))
            manifest.append(dict(id=vid, kind="repeat", family=f"HANDOFF_{port}",
                                 port=port,          #! 埠數宣告：run() 會與 --config 對帳（2026-08-31 實犯：
                                 #  匯出時漏帶 port,dual 批被當 single 量,幾何與數字全錯）
                                 parent_id=p["id"], src_folder=p["store"],
                                 src_score=round(float(p["score"]), 3),
                                 n_sites=p["n_sites"], diag_bridge_w=p["bridge_w"],
                                 gen_ver="handoff-v1", sm=None, heads="probe"))
        json.dump(dict(setup, keep_project=True),
                  open(str(d.joinpath("hfss_setup.json")), "w", encoding="utf-8"), indent=1)
        tmp = d.joinpath("manifest.json.tmp")
        json.dump(manifest, open(str(tmp), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(str(tmp), str(d.joinpath("manifest.json")))
        made.append((name, len(ps)))

    print(f"\n共 {len(made)} 個輸入夾 / {sum(n for _, n in made)} 筆。發車：")
    for name, _n in made:
        print(f"  python -m script.dedust check-dup --input {name}_input")
    for name, _n in made:
        print(f"  python -m script.dedust jobs-add --input {name}_input --store {name} --prio 2")
    return made



def collect(work_root, out_dir, inputs=None):
    """收檔打包：`keep_<store>/project/` 的 HFSS 專案檔 → 一個 pattern 一個資料夾。

    **在正式機跑**（`keep_*` 工作目錄在該機本地碟，不在 NAS）。

    專案檔名是 `project_<config.ID>_<num>.aedt`，num＝manifest 索引（`sim.start(num)`）——
    **不帶 pattern 名**。這裡不改檔名（改名會弄斷 `.aedt` ↔ `.aedtresults` 的關聯），改成
    **資料夾名＝交付名**，一樣對得回 PDF（Ricky 2026-08-31：資料夾對到就好）。
    同時寫 `對照表.csv`（交付名 ↔ 原 id ↔ 原批 ↔ 專案檔），沒收到的也會列出來、不靜默漏掉。
    """
    import csv
    import glob
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    rows_out, n_ok, n_miss = [], 0, 0
    dirs = inputs or [os.path.basename(d) for d in glob.glob(os.path.join(work_root, "keep_*"))]
    for keep in dirs:
        kd = keep if os.path.isdir(keep) else os.path.join(work_root, keep)
        store = os.path.basename(kd)[len("keep_"):]
        man_f = DATASET_PATH.joinpath(f"{store}_input", "manifest.json")
        if not man_f.exists():
            print(f"  ⚠ 找不到 manifest：{man_f}"); continue
        manifest = json.load(open(str(man_f), encoding="utf-8"))
        #! 專案檔在 `<out>/HFSS/project/`——`PatchSimulator.__init__` 有多包一層
        #  `Path(record_path).joinpath("HFSS")`（`patch_simulator/__init__.py:59`）。
        #  2026-08-31 實犯：只找 `<out>/project/` → 39 筆全標「缺」。保留舊路徑當後備。
        proj = next((c for c in (os.path.join(kd, "HFSS", "project"), os.path.join(kd, "project"))
                     if os.path.isdir(c)), None)
        if proj is None:
            found = glob.glob(os.path.join(kd, "**", "*.aedt"), recursive=True)
            print(f"  ⚠ {kd} 找不到 project 夾；整棵樹的 .aedt: {len(found)} 個"
                  + (f"，例如 {found[0]}" if found else "（一個都沒有）"))
            proj = os.path.join(kd, "HFSS", "project")
        for num, m in enumerate(manifest):
            hits = [f for f in glob.glob(os.path.join(proj, f"*_{num}.aedt"))
                    if os.path.basename(f).rsplit("_", 1)[1] == f"{num}.aedt"]
            dest = os.path.join(out_dir, m["id"])
            if not hits:
                n_miss += 1
                rows_out.append([m["id"], m.get("parent_id", ""), m.get("src_folder", ""),
                                 store, "", "缺（未產出或未跑到）"])
                continue
            os.makedirs(dest, exist_ok=True)
            base = os.path.splitext(hits[0])[0]
            for src in [hits[0]] + glob.glob(base + ".aedtresults") + glob.glob(base + ".aedt.*"):
                tgt = os.path.join(dest, os.path.basename(src))
                (shutil.copytree if os.path.isdir(src) else shutil.copyfile)(src, tgt)
            n_ok += 1
            rows_out.append([m["id"], m.get("parent_id", ""), m.get("src_folder", ""),
                             store, os.path.basename(hits[0]), "OK"])
    with open(os.path.join(out_dir, "對照表.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["交付名（＝資料夾名，對應 PDF）", "原 pattern id", "原批 store",
                    "重測 store", "HFSS 專案檔", "狀態"])
        w.writerows(rows_out)
    print(f"打包完成：{n_ok} 個 OK / {n_miss} 個缺 → {out_dir}（對照表.csv 已寫）")
    return n_ok, n_miss



DELIVER_RE = re.compile(r"^(ANT|FLT)_(F(\d+)|S)_(\d+)_(.+)$")


def deliver_picks(prefix="handoff"):
    """從**已重跑的交付 store** 反建 picks → PDF 數字＝附給學長的 .aedt 跑出來的數字。

    交付名本身就編碼了家族與名次（`ANT_F00_1_<name>`／`ANT_S_1_<name>`），所以不必重跑挑選邏輯，
    直接把 `handoff_*_input/manifest.json` ＋ 對應 store 的 `results.json` 讀回來即可。
    這樣 PDF 與交付包同源；不這麼做的話 PDF 用的是原批數字，會出現「PDF 說 +0.69、
    附的專案檔按下 solve 跑出 +0.77」的不一致（2026-08-31 實測 39 筆中有 2 筆會這樣）。"""
    out = {"single": [], "dual": []}
    root = str(DATASET_PATH)
    for d in sorted(os.listdir(root)):
        if not (d.startswith(f"{prefix}_") and d.endswith("_input")):
            continue
        store = d[:-6]
        man_f, res_f = os.path.join(root, d, "manifest.json"), os.path.join(root, store, "results.json")
        if not (os.path.exists(man_f) and os.path.exists(res_f)):
            continue
        manifest = json.load(open(man_f, encoding="utf-8"))
        res = json.load(open(res_f, encoding="utf-8"))
        for m in manifest:
            r = res.get(m["id"])
            if not r or "error" in r:
                print(f"    ⚠ 跳過（無結果）：{m['id']}")
                continue
            g = DELIVER_RE.match(m["id"])
            if not g:
                print(f"    ⚠ 跳過（交付名格式不符）：{m['id']}")
                continue
            port = m.get("port", "single")
            mat = _pattern(store, m["id"])
            sites, _ = diag_bridge_sites(mat, m["diag_bridge_w"], PIXEL_MM)
            pick = dict(store=store, id=m["id"], name=g.group(5), port=port, aliases=[],
                        bridge_w=m["diag_bridge_w"], n_sites=len(sites), sites=sites, mat=mat,
                        family=-1 if g.group(2) == "S" else int(g.group(3)), rank=int(g.group(4)))
            if port == "single":
                pick.update(wm=r["wm"][:3], score=r["wm"][2],
                            rad_margin=r.get("rad_margin"), sel=r.get("sel"))
            else:
                mm = [r.get(f"m{i}") for i in range(1, 7)]
                pick.update(m=mm, energy_max=r.get("energy_max"),
                            score=min(mm[0] + 2, mm[1] + 2, mm[2], mm[3] + 5))
            out[port].append(pick)
    for port in out:
        out[port].sort(key=lambda p: (p["family"] < 0, p["family"], p["rank"]))
        if out[port]:
            print(f"[{port}] 交付 store 反建 {len(out[port])} 張（數字＝重跑值，與附的 .aedt 同源）")
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="送板交付包：家族代表挑選 + PDF 目錄")
    ap.add_argument("cmd", choices=["index", "report", "pdf", "export", "collect"])
    ap.add_argument("--port", choices=["single", "dual", "both"], default="both")
    ap.add_argument("--max-dist", type=int, default=100, help="家族 Hamming 門檻（同 dedust 預設 100）")
    ap.add_argument("--top", type=int, default=5, help="每家族取前幾名（single）")
    ap.add_argument("--top-dual", type=int, default=12,
                    help="dual 每家族取前幾名（池同質、要多看排行前段，Ricky 2026-08-31）")
    ap.add_argument("--dual-window", type=float, default=1.0, help="dual 取王 −N dB 之內")
    ap.add_argument("--buffer", type=float, default=0.3, help="single 門檻（R54 規則①，預設 0.3）")
    ap.add_argument("--max-sites", type=int, default=None,
                    help="橋座數上限（預設依 SITES_RULE：single 17、dual 不濾）")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--png", action="store_true", help="同時輸出每頁 PNG")
    ap.add_argument("--deliver", action="store_true",
                    help="pdf：改用已重跑的 handoff_* store 反建（數字與交付的 .aedt 同源）")
    ap.add_argument("--dry", action="store_true", help="export：只列出要寫什麼，不寫 NAS")
    ap.add_argument("--work", default=".", help="collect：keep_* 工作目錄所在（正式機本地碟）")
    ap.add_argument("--pack", default=None, help="collect：打包輸出路徑（預設 NAS handoff_package）")
    a = ap.parse_args()

    if a.cmd == "collect":
        collect(a.work, a.pack or str(DATASET_PATH.joinpath("handoff_package")))
        return

    rows = load_index(rebuild=(a.cmd == "index"))
    print(f"索引 {len(rows)} 筆（single {sum(1 for r in rows if r['port'] == 'single')} / "
          f"dual {sum(1 for r in rows if r['port'] == 'dual')}）")
    if a.cmd == "index":
        return

    chosen = {}
    if a.deliver:
        chosen = deliver_picks()
    else:
      for port in (["single", "dual"] if a.port == "both" else [a.port]):
        ms = SITES_RULE[port] if a.max_sites is None else a.max_sites
        top = a.top if port == "single" else a.top_dual
        picks = select(rows, port, a.max_dist, top, a.dual_window,
                       buffer=a.buffer, max_sites=ms)
        picks += senior_picks(rows, port, a.top)      # 學長池頂系獨立 S 區
        chosen[port] = picks

    if a.cmd == "export":
        export_inputs(chosen, dry=a.dry)

    if a.cmd == "pdf":
        ylim = axis_ranges([p for ps in chosen.values() for p in ps])
        print(f"全域固定軸範圍（single/dual 共用）：S參數 {ylim['S']}　Gain {ylim['GAIN']}　"
              f"方向圖 {ylim['RAD']}")
        for port, picks in chosen.items():
            if not picks:
                continue
            out = os.path.join(a.out_dir, f"handoff_{port}.pdf")
            png_dir = os.path.join(a.out_dir, f"png_{port}") if a.png else None
            print(f"[{port}] → {out}（{render(port, picks, out, png_dir, ylim)} 頁）")
            for p in picks:
                print(f"    {deliver_name(port, p):48s} {p['score']:+.2f} dB  "
                      f"橋{p['n_sites']:3d}座  ({p['store']}:{p['id']})")
    save_pattern_cache()


if __name__ == "__main__":
    main()
