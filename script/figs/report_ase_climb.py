"""日月光「歷來實驗總整理」第四代：四個任務各一張爬升圖（橫軸＝累計 HFSS 模擬筆數，縱軸＝最差餘裕 dB）。

用法（repo 根目錄）：
    python -m script.figs.report_ase_climb [--only single,sym,dual,wideband] [--refresh] [--root <dataset>]
輸出：docs/進度報告/10月_歷來實驗總整理/figs/gen4_ase/climb_{single,sym,dual,wideband}.png（16:7，dpi 200）

圖面只放淡色點（每筆模擬）、粗階梯（累計最佳）、規格線 0、3–5 個白話方法節點；不放標題與圖下小字。
數字條件與出處記在 `docs/進度報告/10月_歷來實驗總整理/notes/gen4_爬升量化.md`。

資料與口徑（四張都只計唯一圖形，同一圖形重跑只算第一次；重跑過的圖形取各次最差值＝保守值）：
- 單埠天線：本機 NAS 備份 dataset 的單埠標準量測（S0、無橋，排除 hfss_setup 幾何變體），7/03 R7 起到
  records.json 的 wm 紀錄 c48nq1p05_16（7/31）那一筆為止；每筆時間＝store 內樣本檔 mtime。
  階梯只算「方向圖窗口 ≥ 0 且可製造（每塊金屬 ≥ 4 像素）」的設計＝records 的三標口徑；其餘畫灰點。
- 雙埠濾波器：同備份的 p01 可製造幾何＋0.075 mm 橋（R70 起；含閒時自產），尺＝規格 v2
  min(m1+2, m2+2, m3, m4+5)，到 records_dual.json 的 wm_mfg 紀錄 smp073_d_040（8/24）為止。
- 對稱天線：沿用 r80_sym.ordered（R80 5,000 筆＋續探，依 store 完成時間）；截到 10/15 續探收齊為止。
- 寬頻濾波器：沿用 r81_v6.rows_v6（11–40 GHz 全部量測，依批次序），以 −15 dB 版尺（v6a）回算。
單埠／雙埠第一次跑要讀約 6 萬個小檔（數分鐘、低優先權），結果快取在 tmp/ase_climb_*.npz。
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "進度報告" / "10月_歷來實驗總整理" / "figs" / "gen4_ase"
BACKUP = Path(r"C:\Users\Ricky\antenna_nas_backup\dataset")

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DOT, OFF, LINE, NODE = "#9ec5f4", "#d3d2cd", "#2a78d6", "#eb6834"
FIG = (16, 7)
TICK, LABEL, TEXT = 15, 17, 17

SINGLE_END = "c48nq1p05_16"          # records.json wm（+0.79）
DUAL_END = "smp073_d_040"            # records_dual.json wm_mfg 第十七跳（−2.39）
SYM_END = dt.datetime(2026, 10, 16)  # 續探 10/15 收齊；10/19–22 慢速兩批未重跑驗證，見 notes


# ---------------------------------------------------------------- 單埠／雙埠：從 dataset 掃 store

def _bits(t):
    return np.packbits(np.asarray(t).reshape(-1) > 0.5).tobytes()


def _scan(root, keep, value):
    """root 下每個 dedust_* store：keep(名, hfss_setup, manifest, results)→是否納入；value(條目)→(wm, rad) 或 None。
    每筆時間＝store 內同圖樣本檔的 mtime（同圖多檔取最早；找不到才用 results.json 的 mtime）。"""
    import torch
    rows = []
    for d in sorted(root.iterdir()):
        rp = d / "results.json"
        if not d.name.startswith("dedust_") or d.name.endswith("_input") or not rp.exists():
            continue
        inp = root / (d.name + "_input")
        sp = inp / "hfss_setup.json"
        setup = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
        man = json.loads((inp / "manifest.json").read_text(encoding="utf-8")) if (inp / "manifest.json").exists() else []
        res = json.loads(rp.read_text(encoding="utf-8"))
        if not keep(d.name, setup, man, res):
            continue
        when = {}
        for f in d.glob("*.pt"):
            b = _bits(torch.load(f, weights_only=True, map_location="cpu")[0])
            when[b] = min(when.get(b, np.inf), f.stat().st_mtime)
        t0 = rp.stat().st_mtime
        for rid, e in res.items():
            v = value(e) if isinstance(e, dict) and "error" not in e else None
            f = inp / (rid + ".pt")
            if v is None or not f.exists():
                continue
            b = _bits(torch.load(f, weights_only=True, map_location="cpu"))
            rows.append((when.get(b, t0), d.name, rid, b, *v))
    return rows


def _single_keep(name, setup, man, res):
    return setup is None and not any(m.get("port") == "dual" for m in man)


def _single_value(e):
    w = e.get("wm")
    if not isinstance(w, list) or len(w) != 3:
        return None
    rad = e.get("rad_margin")
    return float(w[2]), float(rad) if rad is not None else np.nan


def _dual_keep(name, setup, man, res):
    ok = [e for e in res.values() if isinstance(e, dict) and "error" not in e]
    p01 = bool(ok) and all(e.get("geom") == "p01" for e in ok)
    std = setup == {"diag_bridge_w": 0.075} or name == "dedust_r70b1c"   # r70b1c＝裸 p01 乾淨生成臂，零對角接點
    return p01 and std and not name.startswith("dedust_r69") and any(m.get("port") == "dual" for m in man)


def _dual_value(e):
    m = [e.get(k) for k in ("m1", "m2", "m3", "m4")]
    if any(x is None for x in m):
        return None
    return float(min(m[0] + 2, m[1] + 2, m[2], m[3] + 5)), np.nan     # 規格 v2（records_dual wm_r2 註記）


def load_cached(name, root, refresh):
    cache = REPO / "tmp" / f"ase_climb_{name}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        return [tuple(r) for r in zip(z["t"], z["store"], z["id"], [b.tobytes() for b in z["bits"]], z["wm"], z["rad"])]
    try:                                    # 開發機共用：讀大量檔時降優先權
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    keep, value = (_single_keep, _single_value) if name == "single" else (_dual_keep, _dual_value)
    rows = _scan(root, keep, value)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, t=np.array([r[0] for r in rows]), store=np.array([r[1] for r in rows]),
                        id=np.array([r[2] for r in rows]),
                        bits=np.array([np.frombuffer(r[3], np.uint8) for r in rows]),
                        wm=np.array([r[4] for r in rows]), rad=np.array([r[5] for r in rows]))
    print("cached", cache, len(rows))
    return rows


def _min_piece(b):
    from scipy.ndimage import label
    p = np.unpackbits(np.frombuffer(b, np.uint8))[:625].reshape(25, 25).astype(bool)
    lab, n = label(p, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    return int(np.bincount(lab.ravel())[1:].min()) if n else 0


def series(rows, end_id, gate=None):
    """依時間排序、截到 end_id 那一筆；唯一圖形只留第一次，值取全部重跑中的最差（保守）。"""
    rows = sorted(rows, key=lambda r: (r[0], r[1], r[2]))
    cons = {}
    for r in rows:
        cons[r[3]] = min(cons.get(r[3], np.inf), r[4])
    t_end = next(r[0] for r in rows if r[2] == end_id)
    seen, out = set(), []
    for r in rows:
        if r[0] > t_end:
            break
        if r[3] not in seen:
            seen.add(r[3])
            out.append(r)
    wm = np.array([cons[r[3]] for r in out])
    ok = np.array([gate(r) for r in out]) if gate else np.ones(len(out), bool)
    return dict(wm=wm, ok=ok, ids=[r[2] for r in out], t=[r[0] for r in out], stores=[r[1] for r in out])


# ---------------------------------------------------------------- 對稱天線、寬頻濾波器：沿用既有讀取

def sym_series():
    from script import r80_sym_climb as sc
    from script.figs.r80_sym import ordered
    from script.profiled_batch import load_profile_config
    _, Y, ids, sts = ordered(load_profile_config(sc.CONFIG))
    t = [os.path.getmtime((sc.V2_ROOT if s.startswith("dedust_r80sc") else sc.R80_ROOT) / s / "results.json")
         for s in sts]
    k = np.array([dt.datetime.fromtimestamp(x) < SYM_END for x in t])
    wm = sc.margins(Y).min(1)
    return dict(wm=wm[k], ok=np.ones(int(k.sum()), bool), ids=[i for i, kk in zip(ids, k) if kk],
                t=[x for x, kk in zip(t, k) if kk], stores=[s for s, kk in zip(sts, k) if kk])


def wideband_series():
    from script import r81_curve_sm as sm
    from script.figs.r81_v6 import V6, V6A, rows_v6
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = sm.frequency_grid(v6.measurement)
    rows = rows_v6(v6, grid)
    wm = np.array([sm.true_wm(r, v6a, grid)["worst_margin"] for r in rows])
    return dict(wm=wm, ok=np.ones(len(rows), bool), ids=[r["id"] for r in rows],
                t=[os.path.getmtime(sm.V2_DATASET / r["store"] / "results.json") for r in rows],
                stores=[str(r["store"]) for r in rows])


# ---------------------------------------------------------------- 畫圖

def climb_figure(out, s, lo, hi, nodes, end_fmt="{:+.2f} dB"):
    """nodes：[(id, 文字, (文字 x, 文字 y), ha)]，第一個是起點節點（第一輪模擬的最佳）；
    其餘是方法節點，標在該 id 所在位置的累計最佳上。低於 lo 的點壓在底線。"""
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    wm, ok = s["wm"], s["ok"]
    x = np.arange(1, len(wm) + 1)
    best = np.maximum.accumulate(np.where(ok, wm, -np.inf))
    k0 = int(np.flatnonzero(np.isfinite(best))[0])
    fig, ax = plt.subplots(figsize=FIG, dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    if not ok.all():
        ax.scatter(x[~ok], np.clip(wm[~ok], lo, None), s=6, color=OFF, lw=0, zorder=1)
    ax.scatter(x[ok], np.clip(wm[ok], lo, None), s=7, color=DOT, lw=0, zorder=2)
    tail = x[-1] * 1.02          # 最後一階往右延一小段，終點值才看得到是線的一部分（不然被終點標記蓋住）
    ax.step(np.r_[x[k0:], tail], np.r_[best[k0:], best[-1]], where="post", color=LINE, lw=3.2, zorder=4)
    ax.axhline(0, color=INK, lw=1.4, ls="--", zorder=3)
    ax.plot(x[-1], best[-1], "o", ms=11, color=LINE, mec="white", mew=1.5, zorder=5)
    span = x[-1]

    def pos(key):
        i = s["ids"].index(key)
        return x[i], best[i]

    for key, text, txy, ha in nodes:
        px, py = pos(key)
        ax.annotate(text, (px, py), txy, fontsize=TEXT, color=INK, ha=ha, va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=NODE, lw=1.4, shrinkA=2, shrinkB=5))
        ax.plot(px, py, "o", ms=8, color=NODE, mec="white", mew=1.2, zorder=6)
    ax.annotate(end_fmt.format(best[-1]).replace("-", "−"), (x[-1], best[-1]), (-12, 14), textcoords="offset points", ha="right",
                va="bottom", fontsize=TEXT + 1, color=LINE, fontweight="bold", zorder=6)
    ax.set_xlim(-span * 0.01, span * 1.03)
    ax.set_ylim(lo - (hi - lo) * 0.01, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("累計 HFSS 模擬筆數", fontsize=LABEL, color=MUTED)
    ax.set_ylabel("最差餘裕 (dB)", fontsize=LABEL, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=TICK)
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=DOT, label="每筆模擬")]
    if not ok.all():
        handles.append(Line2D([], [], ls="none", marker="o", ms=7, color=OFF, label="未過方向圖或可製造條件的模擬"))
    handles += [Line2D([], [], color=LINE, lw=3.2, label="累計最佳"),
                Line2D([], [], color=INK, lw=1.4, ls="--", label="規格線 0")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=TICK, framealpha=0.92,
                    edgecolor="none", facecolor=SURF)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURF)
    plt.close(fig)
    print(f"saved {out}  n={len(wm)}  起點節點={pos(nodes[0][0])[1]:+.3f}  final={best[-1]:+.3f}")


def summary(name, s):
    best = np.maximum.accumulate(np.where(s["ok"], s["wm"], -np.inf))
    t = [dt.datetime.fromtimestamp(v) for v in s["t"]]
    st0 = s["stores"][0]
    first = [w for w, o, st in zip(s["wm"], s["ok"], s["stores"]) if st == st0 and o]
    print(f"[{name}] n={len(s['wm'])} 第一個資料夾 {st0} 最佳 {max(first):+.3f}；終點 {best[-1]:+.3f}"
          f"（{s['ids'][int(np.argmax(np.where(s['ok'], s['wm'], -np.inf)))]}）；{min(t):%Y-%m-%d %H:%M} → {max(t):%Y-%m-%d %H:%M}"
          f"；通過規格 {int(((s['wm'] >= 0) & s['ok']).sum())} 筆")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="single,sym,dual,wideband")
    ap.add_argument("--refresh", action="store_true", help="重掃 dataset（單埠／雙埠快取）")
    ap.add_argument("--root", type=Path, default=None, help="dataset 根目錄；預設本機 NAS 備份")
    args = ap.parse_args(argv)
    root = args.root
    if root is None:
        from antenna.utils import DATASET_PATH
        root = BACKUP if BACKUP.exists() else Path(str(DATASET_PATH))
    todo = args.only.split(",")

    if "single" in todo:
        s = series(load_cached("single", root, args.refresh), SINGLE_END,
                   gate=lambda r: _min_piece(r[3]) >= 4 and np.nan_to_num(r[5], nan=-9) >= 0)
        summary("single", s)
        climb_figure(OUT / "climb_single.png", s, lo=-4.0, hi=1.5, nodes=[
            ("p03_d3", "從前代累積的設計出發", (700, -3.35), "left"),
            ("a00_k2","部分左右對稱化後局部修改\n首次通過規格", (2200, -1.15), "left"),
            ("o29b2_011_o26b2_007_o2", "大量產生變體\n代理模型初篩後批次模擬", (9300, 1.12), "left"),
            (SINGLE_END, "移植其他設計的局部結構\n再從好設計局部修改", (24600, 1.2), "right")])
    if "sym" in todo:
        s = sym_series()
        summary("sym", s)
        climb_figure(OUT / "climb_sym.png", s, lo=-4.0, hi=1.2, end_fmt="{:+.3f} dB", nodes=[
            ("r80g001_1039_23133a9e", "從既有天線設計出發", (250, -2.6), "left"),
            ("r80bestrep01_f6ffb938", "代理模型挑選、大量生成對稱設計\n首次通過規格", (1250, 0.75), "left"),
            ("r80sc_b05_014_eb0a73b6", "從好設計左右成對局部修改", (4650, 0.5), "right")])
    if "dual" in todo:
        s = series(load_cached("dual", root, args.refresh), DUAL_END)
        summary("dual", s)
        climb_figure(OUT / "climb_dual.png", s, lo=-9.0, hi=1.0, nodes=[
            ("d70b1_A_11", "從前代累積的設計出發", (700, -6.9), "left"),
            ("d70b3_A_11_d2_00_d2_01", "從好設計局部修改", (1300, -5.0), "left"),
            ("smp002_L_031", "代理模型挑選候選\n模擬機不停機補量", (3300, -1.0), "left"),
            ("smp050_L_011", "自動化流程無人值守運轉", (12300, -1.3), "right")])
    if "wideband" in todo:
        s = wideband_series()
        summary("wideband", s)
        climb_figure(OUT / "climb_wideband.png", s, lo=-12.0, hi=1.6, nodes=[
            ("r81v6_m02_015_324e4cfb", "從既有濾波器設計出發", (70, -9.2), "left"),
            ("r81v2_b62_009_57ec5ae8", "兩埠金屬斷開、大步變異", (220, -2.2), "left"),
            ("r81v2_b85_002_83c6d9e5", "代理模型改學\n「改哪幾格、曲線怎麼變」", (700, 1.0), "left"),
            ("r81v6mg05_009_59eb9c0e", "好設計之間互相移植區塊\n首次通過規格", (1180, -3.2), "left"),
            ("r81v6mg34_000_b4bb9425", "通帶兩側成對修改", (2000, 1.15), "right")])


if __name__ == "__main__":
    main()
