"""日月光「歷來實驗總整理」第四代：四個任務各一張爬升圖（橫軸＝累計 HFSS 模擬筆數，縱軸＝最差餘裕 dB）。

用法（repo 根目錄）：
    python -m script.figs.report_ase_climb [--only single,sym,dual,wideband] [--refresh] [--root <dataset>]
    python -m script.figs.report_ase_climb --only best_single,best_sym,best_wideband     # 最佳設計卡（預設不跑）
    python -m script.figs.report_ase_climb --only spec_antenna                          # 天線規格圖（預設不跑）
輸出：docs/進度報告/10月_歷來實驗總整理/figs/gen4_ase/climb_{single,sym,dual,wideband}.png（16:7，dpi 200）
      同夾 best_{single,sym,wideband}.png（最佳設計卡，16:5，dpi 200）
      同夾 spec_antenna.png（天線規格圖，不含左側帶外淡紫區；用組會 meeting_square.spec_ant(extra=False)，不覆寫組會原圖）
對外規則：圖上小數一律兩位（爬升圖終點標籤、設計卡「最差餘裕」），所以投影片產生器不必再遮擋或裁切。

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
最佳設計卡（best_*）：一列四格、每格近正方形，圖上只放量名與單位、規格區（淡紅禁區＋虛線門檻）、整排左上角
「最差餘裕」（放右上會被讀成方向圖的餘裕，第三版審稿 r1）；第一格標「金屬圖形」；不放設計編號、版本代號與圖下註腳。數值一律現場重算並與 records／既有紀錄核對（不一致就中止）：
- best_single：本機 NAS 備份 dedust_c48nq1_p05 的 c48nq1p05_16（records.json wm），尺＝antenna.losses.worst_margin。
- best_sym：r80_sym 同一批 store 的 r80sc_b05_014_eb0a73b6，尺＝r80_sym_climb.margins（與 worst_margin 交叉核對）。
- best_wideband：R81 v2 資料根的 r81v6mg44_016_caea7f52，尺＝r81_curve_sm.true_wm 以 v6a（−15 dB 版）計。
- 天線卡的方向圖格：整圈 θ、以 0° 方向增益正規化，禁區與 −3 dB 虛線同第 5 頁規格圖（spec_antenna）；
  方向圖餘裕用 dedust.rad_window_margin 現場算，單埠另與 store results.json 的 rad 核對。
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

def climb_figure(out, s, lo, hi, nodes):
    """nodes：[(id, 文字, (文字 x, 文字 y), ha)]，第一個是起點節點（第一輪模擬的最佳）；
    其餘是方法節點，標在該 id 所在位置的累計最佳上。低於 lo 的點壓在底線。終點標籤一律兩位小數（對外規則）。"""
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
    ax.annotate(f"{best[-1]:+.2f} dB".replace("-", "−"), (x[-1], best[-1]), (-12, 14), textcoords="offset points", ha="right",
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


# ---------------------------------------------------------------- 最佳設計卡（對外乾淨版）

CARD = (16, 5)
CARD_SINGLE = ("dedust_c48nq1_p05", "c48nq1p05_16")   # records.json wm（+0.79）
CARD_SYM = "r80sc_b05_014_eb0a73b6"                   # round-80 對稱最佳（+0.599）
CARD_WIDE = "r81v6mg44_016_caea7f52"                  # round-101 寬頻最佳（+0.41，v6a 尺）
BAD = "#e34948"
CTICK, CTITLE, CNOTE = 13, 17, 23


def _card_check(name, got, ref, tol, what="最差餘裕"):
    ok = abs(got - ref) <= tol
    print(f"[{name}] 重算{what} {got:+.4f} dB；對照 {ref:+g}（容差 {tol:g}）→ {'一致' if ok else '不一致'}")
    if not ok:
        raise SystemExit(f"{name}：重算 {got:+.4f} 與對照值 {ref:+g} 不符，中止出圖")


def _card_ax(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=CTITLE, color=INK, pad=10)
    ax.tick_params(colors=MUTED, labelsize=CTICK)
    for sp in ax.spines.values():
        sp.set_color("#bdbcb6")


def _card_pattern(ax, p):
    """25×25 金屬圖形；imshow origin="upper"＝第一索引朝下，饋線邊（row 24）在圖下緣。"""
    ax.imshow(np.asarray(p).reshape(25, 25) > 0.5, cmap="Greys", origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    _card_ax(ax, "金屬圖形")
    ax.set_box_aspect(1)


def _card_curve(ax, f, y, title, bands, xlim, ylim, xticks, yticks):
    """bands：[(起頻, 止頻, 門檻, "le"|"ge")]；le＝曲線須在門檻下，禁區是門檻以上的淡紅塊。"""
    for lo, hi, thr, rel in bands:
        y0, y1 = (thr, ylim[1]) if rel == "le" else (ylim[0], thr)
        ax.fill_between([lo, hi], y0, y1, color=BAD, alpha=0.10, lw=0, zorder=1)
        ax.plot([lo, hi], [thr, thr], color=INK, lw=1.8, ls="--", zorder=3)
    ax.plot(f, y, color=LINE, lw=3.0, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(list(yticks))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("頻率 (GHz)", fontsize=CTICK + 1, color=MUTED)
    _card_ax(ax, title)
    ax.set_box_aspect(1)


RAD_WIN, RAD_FLOOR, RAD_LO = 45, 3, -40   # 規格 ±45° 內不比 0° 方向低 3 dB 以上；圓心 −40 dB，兩張卡都不截
RAD_LABEL_AT = 15                          # 半徑刻度字放在 ±45° 禁區裡（曲線依規格進不來）；0 dB 那圈壓在曲線上，不標字


def _card_polar(ax, rad):
    """28 GHz 方向圖兩切面，畫整圈 θ −180°～180°（rad 檔本來就是整圈，2° 一點），以 0° 方向增益正規化成 0 dB。
    畫法同第 5 頁規格圖（meeting_square.spec_ant）：淡紅＝±45° 內低於 0° 方向 3 dB 的禁區、虛線＝−3 dB 門檻。
    rad 是 dB(GainTotal)；HFSS 角度原樣（θ 順時針為正，φ 標籤不隨 pattern 圖轉向而交換）。
    回傳兩切面方向圖餘裕，用 dedust 記帳同一個函式 rad_window_margin 算。"""
    from script.dedust import rad_window_margin
    th = np.asarray(rad["theta"], dtype=float)
    assert th.min() <= -178 and th.max() >= 178, f"rad 檔 θ 只有 {th.min()}～{th.max()}，不是整圈"
    i0 = int(np.argmin(np.abs(th)))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    w = np.radians(np.linspace(-RAD_WIN, RAD_WIN, 91))
    ax.fill_between(w, RAD_LO, -RAD_FLOOR, color=BAD, alpha=0.10, lw=0, zorder=1)
    ax.plot(w, np.full_like(w, -RAD_FLOOR), color=INK, lw=1.8, ls="--", zorder=3)
    margins = {}
    for k, color, label in (("phi0", LINE, "φ = 0°"), ("phi90", NODE, "φ = 90°")):
        g = np.asarray(rad[k], dtype=float)
        norm = g - g[i0]
        assert norm.min() >= RAD_LO, f"{k} 最低 {norm.min():.1f} dB 低於圓心 {RAD_LO}，會被截"
        margins[k] = rad_window_margin(th, g, RAD_WIN, RAD_FLOOR)
        ax.plot(np.radians(th), norm, color=color, lw=3.0, label=label, zorder=4)
    ax.set_ylim(RAD_LO, 3)
    ax.set_yticks([-30, -20, -10, 0])
    ax.set_rlabel_position(RAD_LABEL_AT)
    ax.set_yticklabels(["−30", "−20", "−10", ""], fontsize=CTICK - 2, color=MUTED)
    ax.set_xticks(np.radians(range(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "", "180°", "", "−90°", "−45°"], fontsize=CTICK - 1, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_facecolor(SURF)
    ax.spines["polar"].set_color("#bdbcb6")
    ax.set_title("28 GHz 方向圖，相對 0° (dB)", loc="left", fontsize=CTITLE, color=INK, pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=CTICK + 1,
              handlelength=1.4, columnspacing=1.4)
    return margins


def _card_frame():
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=CARD, dpi=200)
    fig.patch.set_facecolor(SURF)
    # 每格寬＝高≈3.1 吋：寬 15.04/(4+3×0.28)＝3.10、高 0.62×5＝3.10；右側留邊給極座標最外圈的角度字
    gs = fig.add_gridspec(1, 4, left=0.02, right=0.96, bottom=0.16, top=0.78, wspace=0.28)
    return fig, gs


def _card_save(fig, out, wm):
    fig.text(0.02, 0.975, f"最差餘裕 {wm:+.2f} dB".replace("-", "−"), ha="left", va="top",
             fontsize=CNOTE, color=LINE, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURF)
    plt.close(fig)
    print("saved", out)


def _antenna_card(out, p, y, rad, wm, rad_ref=None):
    """單埠／對稱天線共用：[pattern | S11 | 增益 | 28 GHz 方向圖]；規格＝26.5–29.5 GHz S11 ≤ −10 dB、增益 ≥ 4 dBi。"""
    f = np.arange(24, 32.01, 0.5)
    fig, gs = _card_frame()
    _card_pattern(fig.add_subplot(gs[0]), p)
    xs = dict(xlim=(24, 32), xticks=[24, 26, 28, 30, 32])
    _card_curve(fig.add_subplot(gs[1]), f, y[0], "S11 (dB)", [(26.5, 29.5, -10, "le")],
                ylim=(-30, 0), yticks=range(-30, 1, 10), **xs)
    _card_curve(fig.add_subplot(gs[2]), f, y[1], "增益 (dBi)", [(26.5, 29.5, 4, "ge")],
                ylim=(-10, 10), yticks=range(-10, 11, 5), **xs)
    rm = _card_polar(fig.add_subplot(gs[3], projection="polar"), rad)
    print("  方向圖餘裕 " + "、".join(f"{k} {v:+.4f}" for k, v in rm.items()) + f" → {min(rm.values()):+.4f} dB")
    for k, ref in (rad_ref or {}).items():               # results.json 的 rad 是兩位小數
        _card_check(k, rm[k], ref, 0.005, what="方向圖餘裕")
    _card_save(fig, out, wm)
    return rm


def best_single_card(out):
    """單埠最佳（records.json wm c48nq1p05_16）：資料取本機 NAS 備份；尺＝losses.worst_margin（dedust 記帳同一把）。"""
    import torch
    from antenna.losses import worst_margin
    from antenna.training import load_config
    store, rid = CARD_SINGLE
    p = np.asarray(torch.load(BACKUP / (store + "_input") / (rid + ".pt"), weights_only=True)).reshape(25, 25) > 0.5
    y = None
    for f in sorted((BACKUP / store).glob("*.pt")):          # store 是 hash 檔名；以 pattern 比對找這一筆
        pp, yy = torch.load(f, weights_only=True, map_location="cpu")
        if (np.asarray(pp).reshape(25, 25) > 0.5).tolist() == p.tolist():
            y = np.asarray(yy, dtype=float).reshape(2, -1)
            break
    assert y is not None, f"{store} 裡找不到 {rid} 的樣本"
    rad = torch.load(BACKUP / store / "rad" / (rid + ".pt"), weights_only=True)
    cfg = load_config(REPO / "configs" / "single_r5_explore.yaml")
    wm, per = worst_margin(y, ["S11", "Gain"], cfg.targets)
    rec = json.loads((REPO / "docs" / "records.json").read_text(encoding="utf-8"))["wm"]
    assert rec["id"] == rid, f"records.json 的 wm 已換成 {rec['id']}，卡片仍是 {rid}"
    _card_check("best_single", wm, rec["value"], 0.005)
    print(f"  S11 餘裕 {per['S11']:+.4f}、Gain 餘裕 {per['Gain']:+.4f}")
    res = json.loads((BACKUP / store / "results.json").read_text(encoding="utf-8"))[rid]
    _antenna_card(out, p, y, rad, wm, rad_ref=res["rad"])


def best_sym_card(out):
    """對稱最佳：與 r80_sym.card_figure 同一份資料（續探 store）；尺＝r80_sym_climb.margins，另以 worst_margin 交叉核對。"""
    import torch
    from antenna.losses import worst_margin
    from antenna.training import load_config
    from script import r80_sym_climb as sc
    rid = CARD_SYM
    store = next(s for s in sorted(sc.V2_ROOT.glob("dedust_r80scb??s?"))
                 if rid in json.loads((s / "results.json").read_text(encoding="utf-8")))
    e = json.loads((store / "results.json").read_text(encoding="utf-8"))[rid]
    p, y = torch.load(store / e["sample_file"], weights_only=True, map_location="cpu")
    y = np.asarray(y, dtype=float)
    rad = torch.load(store / e["rad_file"], weights_only=False)
    wm = float(sc.margins(y[None]).min())
    cfg = load_config(REPO / "configs" / "single_r5_explore.yaml")
    _card_check("best_sym 交叉（worst_margin）", worst_margin(y, ["S11", "Gain"], cfg.targets)[0], wm, 1e-4)
    _card_check("best_sym", wm, 0.599, 0.0006)
    _antenna_card(out, p, y, rad, wm)                       # 這批 results.json 沒存方向圖餘裕，只現場算


def best_wideband_card(out):
    """寬頻濾波器最佳：[pattern | S21 | S11 | S22]；以 v6a（−15 dB 版）規格畫門檻與禁區，尺＝r81_curve_sm.true_wm。"""
    from script import r81_curve_sm as sm
    from script.figs.r81_v6 import V6, V6A
    rid = CARD_WIDE
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = sm.frequency_grid(v6.measurement)
    store = next(s for s in sorted(sm.V2_DATASET.glob(f"dedust_{rid.split('_')[0]}s?"))
                 if rid in json.loads((s / "results.json").read_text(encoding="utf-8")))
    r = next(r for r in sm.load_store(store, grid) if r["id"] == rid)
    sc = sm.true_wm(r, v6a, grid)
    wm = sc["worst_margin"]
    _card_check("best_wideband", wm, 0.41, 0.005)
    print("  各項餘裕 " + "、".join(f"{k} {v:+.2f}" for k, v in sc["margins"].items()))
    fig, gs = _card_frame()
    _card_pattern(fig.add_subplot(gs[0]), r["pattern"])
    for col, (k, lab) in enumerate(((1, "S21"), (0, "S11"), (2, "S22")), start=1):
        bands = [(b["start_ghz"], b["stop_ghz"], b["threshold"], b["relation"])
                 for b in v6a.score_spec["bands"] if b["label"] == lab]
        _card_curve(fig.add_subplot(gs[col]), grid, r["y"][k], f"{lab} (dB)", bands, xlim=(11, 40), ylim=(-45, 2),
                    xticks=[15, 20, 25, 30, 35, 40], yticks=range(-40, 1, 10))
    _card_save(fig, out, wm)


def spec_antenna_figure(out):
    """天線規格圖（對外版）：S11、增益、28 GHz 方向圖三格，規格淡紅禁區直接畫在圖上，曲線是示範設計。
    沿用組會 `meeting_square.spec_ant`（凍結 JSON、零 HFSS），以 extra=False 不畫左側帶外的淡紫「額外要求」區；
    另存到 out，不覆寫組會原圖。meeting_square 載入時會改全域 rcParams，且同程序先畫的圖（report_r1r10_style 等）
    也改過底色，所以在 rc_context 裡先還原 matplotlib 預設再載入，出圖與執行順序無關、也不影響其他圖。"""
    with plt.rc_context():
        plt.rcdefaults()
        from script.figs import meeting_square as ms
        ms.spec_ant(extra=False, name=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="single,sym,dual,wideband",
                    help="另有 best_single,best_sym,best_wideband（最佳設計卡）、spec_antenna（天線規格圖），預設不跑")
    ap.add_argument("--refresh", action="store_true", help="重掃 dataset（單埠／雙埠快取）")
    ap.add_argument("--root", type=Path, default=None, help="dataset 根目錄；預設本機 NAS 備份")
    args = ap.parse_args(argv)
    root = args.root
    if root is None:
        from antenna.utils import DATASET_PATH
        root = BACKUP if BACKUP.exists() else Path(str(DATASET_PATH))
    todo = args.only.split(",")

    if "best_single" in todo:
        best_single_card(OUT / "best_single.png")
    if "best_sym" in todo:
        best_sym_card(OUT / "best_sym.png")
    if "best_wideband" in todo:
        best_wideband_card(OUT / "best_wideband.png")
    if "spec_antenna" in todo:
        spec_antenna_figure(OUT / "spec_antenna.png")

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
        climb_figure(OUT / "climb_sym.png", s, lo=-4.0, hi=1.2, nodes=[
            ("r80g001_1039_23133a9e", "從既有天線設計出發", (250, -2.6), "left"),
            ("r80bestrep01_f6ffb938", "代理模型挑選、大量生成對稱設計\n首次通過規格", (1250, 0.75), "left"),
            ("r80sc_b05_014_eb0a73b6", "從好設計左右成對局部修改", (4650, 0.5), "right")])
    if "dual" in todo:
        s = series(load_cached("dual", root, args.refresh), DUAL_END)
        summary("dual", s)
        climb_figure(OUT / "climb_dual.png", s, lo=-9.0, hi=1.0, nodes=[
            ("d70b1_A_11", "從前代累積的設計出發", (700, -6.9), "left"),
            ("d70b3_A_11_d2_00_d2_01", "從好設計局部修改", (1300, -5.0), "left"),
            ("smp002_L_031", "代理模型挑選候選\n模擬機不停機補模擬", (3300, -1.0), "left"),
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
