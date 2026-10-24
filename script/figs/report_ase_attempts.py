"""日月光「歷來實驗總整理」第四版：四代嘗試歷程時間軸，仿第二代錢鵬予學長自繪時間軸的樣式。

用法（repo 根目錄）：
    python script/figs/report_ase_attempts.py [--only gen1,gen2,gen3,gen4]
輸出：docs/進度報告/10月_歷來實驗總整理/figs/attempts_timeline/gen1.png … gen4.png（12.5×5.2 吋，dpi 200）

樣式：一條由月份箭頭段組成的橫向箭頭，五色循環沿用學長原圖；每個月往上或往下掛一串橢圓泡泡，
細線連到箭頭；泡泡填該月顏色的淡色、文字粗體＝沿用到碩論或後代的嘗試，白底＝只試過；右上角小圖例，
四張同一位置，右下角在第二、四代會壓到泡泡。
跨年處在箭頭下方標年份。泡泡文字一律白話、不放成效數字與評價字眼，只留 0.21 mm 格寬與 n260 頻段名，
每顆不超過 8 個中文字、最多兩行。

泡泡內容挑自同夾 notes/gen1–gen4_嘗試歷程.md 標為「主要」的嘗試；第二代以學長原圖
figs/attempts_gen2/ga2_timeline_his.png 為骨架，補上回滾、孤島處理、對角導通、雙埠濾波器與 n260。
每顆泡泡的月份、看到什麼與後續寫在 make_pptx_v4.py 的備忘稿。零 NAS、零模擬。

版面由程式自行檢查：泡泡兩兩不重疊、細線不穿過別的泡泡、年份與圖例不壓到泡泡或細線、全部在圖框內，
任一項不過就中止，不出圖。
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Polygon
from PIL import ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "進度報告" / "10月_歷來實驗總整理" / "figs" / "attempts_timeline"
FONT_R, FONT_B = "C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc"

# 學長原圖的五色，依月份循環；取色自 ga2_timeline_his.png
CYCLE = ["#23779b", "#f6b339", "#9fa25d", "#c27131", "#a11306"]
INK, MUTED = "#1b1b1b", "#5a6572"

W, H = 12.5, 5.2                  # 圖框，英吋；放進 16:9 投影片標題下方約 12.2×5.1 吋的區域
FS = 13.5                         # 泡泡字級 pt，放到投影片上約 13 pt
LS = 1.12                         # 兩行泡泡的行距
HB = 0.56                         # 泡泡高，英吋；所有泡泡同高，寬度依文字
AX0, AX1 = 0.22, W - 0.40         # 箭頭本體左右端；最後一段的尖端再往右 TIP
AH, TIP = 0.52, 0.17              # 箭頭高、每段尖端深度
G_UP, G_DN, GAP = 0.16, 0.30, 0.13   # 箭頭到第一顆泡泡的距離，下方多留年份的位置；泡泡間距
LW_EDGE, LW_STEM = 2.6, 2.3
AY = 0.06 + G_DN + 3 * HB + 2 * GAP + AH / 2   # 箭頭中線，讓上下各容得下三層泡泡

# ---------------------------------------------------------------- 四代內容
# 每顆泡泡：(月份位置, 上下, 第幾層, 文字, 是否留下沿用)
#   月份位置＝從第一個月起算的序號；第四代一個月較寬，用 ±0.25 分成月初、月底兩串
#   上下：+1 掛在箭頭上方、−1 掛在下方；第幾層：1 最靠近箭頭
#   排法：上方一串由上往下、下方一串由上往下都是時間先後，與學長原圖相同
GENS = {
    "gen1": dict(start=(2023, 7), n=13, legend=(10.6, H - 0.80), bubbles=[
        (0, +1, 1, "像素化\n定題", True),
        (1, -1, 1, "經 HFSS\n直接反傳", False),
        (2, -1, 2, "網格大小", True),
        (3, +1, 3, "強化學習", False),
        (3, +1, 2, "合成梯度", False),
        (3, +1, 1, "已知解\n驗證題", False),
        (4, -1, 1, "代理模型\n只用當次", False),
        (4, -1, 2, "累積資料\n重訓", True),
        (4, -1, 3, "饋入固定\n金屬", True),
        (5, +1, 3, "只罰\n未達標", True),
        (5, +1, 2, "有無固定\n金屬對照", False),
        (5, +1, 1, "四層\n封裝板", True),
        (6, -1, 1, "固定區\n位置調整", True),
        (7, +1, 1, "代理模型\n正規化層", True),
        (9, +1, 3, "激活函數\n比較", True),
        (9, +1, 2, "延伸到\n濾波天線", True),
        (9, +1, 1, "波束寬\n條件", False),
        (11, +1, 1, "口試委員\n建議", False),
        (12, -1, 1, "卷積出圖", False),
        (12, -1, 2, "對角連通\n分析", True),
    ]),
    "gen2": dict(start=(2024, 8), n=11, legend=(10.6, H - 0.80), bubbles=[
        (0, +1, 3, "接手\n前代流程", False),
        (0, +1, 2, "卷積\n生成器", False),
        (0, +1, 1, "饋入固定\n金屬", True),
        (1, -1, 1, "對稱生成", False),
        (1, -1, 2, "小圖案\n拼接", False),
        (2, +1, 2, "比較\n優化器", False),
        (2, +1, 1, "損失改用\nSmooth L1", True),
        (3, -1, 1, "近似\n二值化", False),
        (3, -1, 2, "隨機輸入", False),
        (4, +1, 1, "非對稱\n正規化", True),
        (5, -1, 1, "週期翻轉\n像素", True),
        (6, +1, 2, "回滾", True),
        (6, +1, 1, "孤島\n形態處理", False),
        (7, -1, 1, "BiScaleNorm", True),
        (8, +1, 3, "代理模型\n梯度誤差", False),
        (8, +1, 2, "保留孤島", True),
        (8, +1, 1, "0.21 mm\n對角導通", True),
        (9, -1, 1, "小圖案\n拼接重啟", False),
        (9, -1, 2, "雙埠\n濾波器", True),
        (10, +1, 1, "n260\n頻段", True),
    ]),
    "gen3": dict(start=(2025, 7), n=12, legend=(10.6, H - 0.80), bubbles=[
        (0, +1, 3, "變異\n自動觸發", False),
        (0, +1, 2, "左右翻轉\n擴增", False),
        (0, +1, 1, "七巧板", False),
        (3, -1, 1, "兩軸對稱\n翻轉", False),
        (3, -1, 2, "只罰\n不達標", True),
        (3, -1, 3, "ACP\n循環排程", True),
        (4, +1, 2, "七巧板\n形狀庫", False),
        (4, +1, 1, "理論貼片\n起點", True),
        (5, -1, 1, "重啟方式\n比較", True),
        (5, -1, 2, "TV 孤島\n懲罰", False),
        (5, -1, 3, "生成模型\n產對稱", False),
        (6, +1, 1, "IS 孤島\n懲罰", False),
        (7, -1, 1, "GC\n閉運算", False),
        (7, -1, 2, "SC\n連通 loss", True),
        (7, -1, 3, "DLF\n樣本篩選", True),
        (8, +1, 2, "饋入\n連通率", True),
        (8, +1, 1, "濾波\n兩埠接通", False),
        (9, -1, 1, "事後補洞", False),
        (10, +1, 1, "饋入固定區\n擴大", False),
        (11, -1, 1, "濾波器\n泛用驗證", True),
    ]),
    "gen4": dict(start=(2026, 6), n=5, legend=(10.6, H - 0.80), bubbles=[
        (-0.25, +1, 2, "復現前代\n加延伸", False),
        (-0.25, +1, 1, "方向圖\n納入評分", True),
        (-0.25, -1, 1, "饋入固定\n金屬", True),
        (0.25, -1, 1, "代理模型\n更新方式", False),
        (0.75, +1, 3, "從前代\n設計出發", True),
        (0.75, +1, 2, "批次\n線上學習", True),
        (0.75, +1, 1, "多台機器\n重跑", True),
        (0.75, -1, 1, "拔碎片\n與接回", False),
        (0.75, -1, 2, "局部\n對稱化", False),
        (1.25, +1, 1, "代理模型\n新架構", True),
        (1.25, -1, 1, "局部修改\n與移植", True),
        (1.75, +1, 2, "可微物理\n模擬", False),
        (1.75, +1, 1, "較嚴網格\n複驗", False),
        (1.75, -1, 1, "菱形橋", True),
        (1.75, -1, 2, "事後完全\n鏡射", False),
        (2.25, +1, 1, "雙埠\n濾波器", False),
        (3.75, +1, 2, "生成時\n精確鏡射", True),
        (3.75, +1, 1, "成對修改", True),
        (4.25, -1, 1, "寬頻濾波\n兩埠斷開", True),
        (4.25, -1, 2, "變化量\n代理模型", True),
    ]),
}


# ---------------------------------------------------------------- 幾何
def _tint(hex_color, a=0.24):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(1 - a + a * rgb)


_FONTS = {}


def _text_w(text, bold):
    """字串寬度，英吋；用正黑體實際字寬量。"""
    key = bold
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(FONT_B if bold else FONT_R, 200)
    return _FONTS[key].getlength(text) / 200 * FS / 72


def bubble_w(text, bold):
    """泡泡寬：每一行的左右兩角都要落在橢圓內，留 9% 邊。"""
    lines = text.split("\n")
    em = FS / 72
    b = HB / 2
    y_out = (0.5 * LS * em if len(lines) == 2 else 0) + 0.42 * em
    k = np.sqrt(1 - (y_out / b) ** 2)
    a = max(_text_w(s, bold) / 2 for s in lines) / k / 0.91
    return max(2 * a, 0.92)


def layout(spec):
    n = spec["n"]
    seg = (AX1 - AX0) / n

    def xc(pos):
        return AX0 + (pos + 0.5) * seg + TIP / 2

    items = []
    for pos, side, lev, text, kept in spec["bubbles"]:
        if side > 0:
            y = AY + AH / 2 + G_UP + (lev - 1) * (HB + GAP) + HB / 2
        else:
            y = AY - AH / 2 - G_DN - (lev - 1) * (HB + GAP) - HB / 2
        items.append(dict(x=xc(pos), y=y, w=bubble_w(text, kept), h=HB, text=text, kept=kept, side=side,
                          lev=lev, col=CYCLE[int(np.floor(pos + 0.5)) % len(CYCLE)],
                          pos=pos))
    # 細線：每一串由箭頭邊連到最外層，中間穿過同串的泡泡
    stems = []
    cols = {}
    for it in items:
        cols.setdefault((it["pos"], it["side"]), []).append(it)
    for (pos, side), group in cols.items():
        group.sort(key=lambda it: it["lev"])
        y0 = AY + side * AH / 2
        for it in group:
            y1 = it["y"] - side * HB / 2
            stems.append(dict(x=it["x"], y0=y0, y1=y1, col=it["col"], key=(pos, side)))
            y0 = it["y"] + side * HB / 2
    for it in items:
        it["key"] = (it["pos"], it["side"])
    return seg, xc, items, stems


def _in_ellipse(px, py, it, grow=0.0):
    a, b = it["w"] / 2 + grow, it["h"] / 2 + grow
    return ((px - it["x"]) / a) ** 2 + ((py - it["y"]) / b) ** 2 <= 1


def check(name, items, stems, boxes):
    """泡泡不重疊、細線不穿過別串泡泡、年份與圖例不壓到東西、全部在圖框內；有問題就中止。"""
    errs = []
    t = np.linspace(0, 2 * np.pi, 180)
    for i, p in enumerate(items):
        if p["x"] - p["w"] / 2 < 0.04 or p["x"] + p["w"] / 2 > W - 0.04 or \
                p["y"] - p["h"] / 2 < 0.04 or p["y"] + p["h"] / 2 > H - 0.04:
            errs.append(f"出界：{p['text']!r}")
        for q in items[i + 1:]:
            px = p["x"] + (p["w"] / 2) * np.cos(t)
            py = p["y"] + (p["h"] / 2) * np.sin(t)
            if np.any(_in_ellipse(px, py, q, grow=0.06)) or _in_ellipse(p["x"], p["y"], q):
                errs.append(f"重疊：{p['text']!r} 與 {q['text']!r}")
    for s in stems:
        ys = np.linspace(min(s["y0"], s["y1"]), max(s["y0"], s["y1"]), 60)
        for q in items:
            if q["key"] == s["key"]:
                continue
            if np.any(_in_ellipse(np.full_like(ys, s["x"]), ys, q, grow=0.05)):
                errs.append(f"細線穿過：{q['text']!r}")
    for label, (x0, y0, x1, y1) in boxes:
        for q in items:
            xs, ys = np.meshgrid(np.linspace(x0, x1, 15), np.linspace(y0, y1, 6))
            if np.any(_in_ellipse(xs, ys, q, grow=0.05)):
                errs.append(f"{label} 壓到 {q['text']!r}")
        for s in stems:
            if x0 - 0.04 <= s["x"] <= x1 + 0.04 and min(s["y0"], s["y1"]) < y1 and max(s["y0"], s["y1"]) > y0:
                errs.append(f"{label} 壓到細線")
        if x0 < 0.02 or x1 > W - 0.02 or y0 < 0.02 or y1 > H - 0.02:
            errs.append(f"{label} 出界")
    if errs:
        raise SystemExit(f"{name} 版面檢查不過：\n  " + "\n  ".join(dict.fromkeys(errs)))


# ---------------------------------------------------------------- 畫圖
def draw(name, spec):
    font_manager.fontManager.addfont(FONT_R)
    font_manager.fontManager.addfont(FONT_B)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=FONT_R).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    seg, xc, items, stems = layout(spec)
    n = spec["n"]
    y_, m_ = spec["start"]
    months = [((y_ * 12 + m_ - 1 + k) // 12, (m_ - 1 + k) % 12 + 1) for k in range(n)]

    # 年份：第一段靠左緣；跨年處置中在十二月與一月的交界，箭頭下方
    boxes = []
    year_lab = []
    for k, (yy, mm) in enumerate(months):
        if k == 0 or mm == 1:
            wid = _text_w(str(yy), True) * 12.5 / FS
            x0 = AX0 + 0.03 if k == 0 else AX0 + k * seg + TIP / 2 - wid / 2
            y1 = AY - AH / 2 - 0.04
            boxes.append((f"年份 {yy}", (x0, y1 - 0.19, x0 + wid, y1)))
            year_lab.append((x0, y1, str(yy)))
    lx, ly = spec["legend"]
    leg_w = 0.42 + 0.1 + _text_w("填色＝留下沿用", True) * 12.5 / FS
    boxes.append(("圖例", (lx, ly - 0.04, lx + leg_w, ly + 0.66)))
    check(name, items, stems, boxes)

    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # 月份箭頭段：第一段左緣平直，其餘左緣內凹；每段右緣是尖端，白色細縫分隔
    yb, yt = AY - AH / 2, AY + AH / 2
    for k, (yy, mm) in enumerate(months):
        x0, x1 = AX0 + k * seg, AX0 + (k + 1) * seg
        pts = [(x0, yb), (x1, yb), (x1 + TIP, AY), (x1, yt), (x0, yt)]
        if k:
            pts.append((x0 + TIP, AY))
        ax.add_patch(Polygon(pts, closed=True, fc=CYCLE[k % len(CYCLE)], ec="white", lw=2.2, zorder=2))
        ax.text(xc(k), AY, str(mm), ha="center", va="center", color="white", fontsize=20, fontweight="bold",
                zorder=3)
    for x0, y1, s in year_lab:
        ax.text(x0, y1, s, ha="left", va="top", color=MUTED, fontsize=12.5, fontweight="bold", zorder=3)

    for s in stems:
        ax.plot([s["x"], s["x"]], [s["y0"], s["y1"]], color=s["col"], lw=LW_STEM, solid_capstyle="butt",
                zorder=1)
    for it in items:
        fc = _tint(it["col"]) if it["kept"] else "white"
        ax.add_patch(Ellipse((it["x"], it["y"]), it["w"], it["h"], fc=fc, ec=it["col"], lw=LW_EDGE, zorder=4))
        ax.text(it["x"], it["y"], it["text"], ha="center", va="center", color=INK, fontsize=FS,
                fontweight="bold" if it["kept"] else "normal", linespacing=LS, zorder=5)

    # 圖例：右上角兩列
    for k, (kept, lab) in enumerate(((True, "填色＝留下沿用"), (False, "白底＝試過"))):
        cy = ly + 0.48 - k * 0.34
        ax.add_patch(Ellipse((lx + 0.21, cy), 0.40, 0.24, fc=_tint(MUTED, 0.22) if kept else "white", ec=MUTED,
                             lw=1.8, zorder=4))
        ax.text(lx + 0.52, cy, lab, ha="left", va="center", color=INK, fontsize=12.5,
                fontweight="bold" if kept else "normal", zorder=5)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    kept = sum(it["kept"] for it in items)
    print(f"{out.relative_to(REPO)}：{len(items)} 顆泡泡，留下 {kept}、試過 {len(items) - kept}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(GENS))
    args = ap.parse_args()
    for name in args.only.split(","):
        spec = GENS[name]
        for _, _, _, text, _ in spec["bubbles"]:
            lines = text.split("\n")
            cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
            assert len(lines) <= 2 and cjk <= 8, text
        draw(name, spec)


if __name__ == "__main__":
    main()
