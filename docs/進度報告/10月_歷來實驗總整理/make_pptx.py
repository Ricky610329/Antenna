# -*- coding: utf-8 -*-
"""給日月光的「歷來實驗總整理」投影片產生器：四代天線與濾波器的方法與成果，以世代為主軸。

版式沿用 09 月 make_pptx.py 的元件（13.33×7.5 in、Blank 版面、Microsoft JhengHei、表格、picture_fit、
content／section 頁），標題樣式照 Ricky 改過的九月份 pptx：30pt 級粗體標題、標題下灰細線加短藍線、
標題下一句描述、表頭淺藍底。每頁底部一條淺灰結語帶，圖表出處用頁底小字；兩者都同步寫進備忘稿，
`FOOTER_BAND`／`SOURCE_LINE` 設成 False 就只留在備忘稿。

數字一律抄自同夾 `實驗總表.md`；分級標記：〔學長報告〕〔圖讀〕〔量測〕〔單次〕〔公證〕，定義見第 4 頁。
跑法：cd 本夾 && python make_pptx.py → 歷來實驗總整理.pptx（本夾在 .gitignore，不進 git）。
時間軸圖由 script/figs/report_generations.py 產生。
"""
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIG = HERE / "figs"
OUT = HERE / "歷來實驗總整理.pptx"

FOOTER_BAND = False     # 每頁底部結語帶；False＝結語只寫進備忘稿
SOURCE_LINE = False     # 頁底出處小字；False＝出處只寫進備忘稿

FONT = "Microsoft JhengHei"
INK = RGBColor(0x15, 0x2C, 0x3D)
BLUE = RGBColor(0x24, 0x5B, 0x85)
MUTED = RGBColor(0x52, 0x6A, 0x7D)
RULE = RGBColor(0xCC, 0xD8, 0xE2)
HEAD = RGBColor(0xEE, 0xF4, 0xF8)
BAND = RGBColor(0xEE, 0xF2, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H = 13.333, 7.5
L, R = 0.583, 12.75
CW = R - L
BODY_TOP = 1.80
BODY_BOTTOM = 6.36

# 數字分級標記（定義見「數字分級」頁）
SR, EYE, MEAS, ONE, NOT = "〔學長報告〕", "〔圖讀〕", "〔量測〕", "〔單次〕", "〔公證〕"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def repo(rel):
    return REPO / rel


# ---------------------------------------------------------------- 基本元件（沿用 09 月 make_pptx.py）
def _run1(par, text, size, bold=False, color=INK):
    r = par.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), bold, color
    rpr = r._r.get_or_add_rPr()                      # 中文字也指定正黑體，避免 PowerPoint 換成佈景字型
    rpr.set("lang", "zh-TW")                         # 繁中斷行規則：右括號不放行首
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)
    return r


TAG = re.compile(r"(〔[^〕]*〕)")


def _run(par, text, size, bold=False, color=INK):
    """分級標記〔…〕自動縮小、改灰色，讓數字本身是主角。"""
    r = None
    for part in TAG.split(text):
        if not part:
            continue
        if TAG.fullmatch(part):
            r = _run1(par, part, round(size * 0.8, 1), False, MUTED)
        else:
            r = _run1(par, part, size, bold, color)
    return r


def textbox(slide, x, y, w, h, lines, size=19, color=INK, align=None, anchor=None, spacing=1.15, after=0):
    """lines: list of str 或 (str, bold) 或 [(str,bold,color), ...]（同段多 run）。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    if anchor is not None:
        tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.line_spacing = spacing
        if after:
            par.space_after = Pt(after)
        if align is not None:
            par.alignment = align
        runs = line if isinstance(line, list) else [line if isinstance(line, tuple) else (line, False)]
        for item in runs:
            text, bold = item[0], item[1]
            _run(par, text, size, bold, item[2] if len(item) > 2 else color)
    return tb


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def picture_fit(slide, path, x, y, w, h, valign="top"):
    """把圖放進 (x,y,w,h) 框內、保持比例、水平置中；valign 決定垂直位置。"""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    top = y if valign == "top" else y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(top), Inches(pw), Inches(ph))


def page_number(slide):
    n = len(prs.slides)
    textbox(slide, 12.15, 7.06, 0.8, 0.32, [str(n)], size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def set_notes(slide, parts):
    text = "\n".join(p for p in parts if p)
    if text:
        slide.notes_slide.notes_text_frame.text = text


def content_slide(title, desc=None, footer=None, source=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    textbox(s, L - 0.04, 0.32, CW, 0.66, [(title, True)], size=28, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, L, 1.07, CW, 0.012, RULE)
    rect(s, L, 1.055, 1.36, 0.04, BLUE)
    if desc:
        textbox(s, L - 0.04, 1.17, CW, 0.5, [desc], size=17, color=INK)
    if footer and FOOTER_BAND:
        rect(s, 0.0, 6.78, W, 0.72, BAND)
        textbox(s, 0.6, 6.78, 12.1, 0.72, [(footer, True)], size=16, color=INK, align=PP_ALIGN.CENTER,
                anchor=MSO_ANCHOR.MIDDLE)
    if source and SOURCE_LINE:
        textbox(s, L - 0.04, 6.42, CW, 0.3, ["出處：" + source], size=10.5, color=MUTED)
    page_number(s)
    set_notes(s, ["結語：" + footer if footer else None, "出處：" + source if source else None, notes])
    return s


def section_slide(num, title, sub=None):
    s = prs.slides.add_slide(BLANK)
    textbox(s, 0, 2.45, W, 1.2, [(num, True)], size=54, color=BLUE, align=PP_ALIGN.CENTER)
    textbox(s, 0, 3.75, W, 0.9, [(title, True)], size=32, align=PP_ALIGN.CENTER)
    if sub:
        textbox(s, 0, 4.65, W, 0.6, [sub], size=18, color=MUTED, align=PP_ALIGN.CENTER)
    page_number(s)
    return s


def numbered(slide, y, n, head, body, size=17):
    o = rect(slide, 0.95, y + 0.06, 0.5, 0.5, BLUE, MSO_SHAPE.OVAL)
    tf = o.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _run(tf.paragraphs[0], str(n), 18, True, WHITE)
    textbox(slide, 1.7, y, 10.9, 1.2, [(head, True), (body, False)], size=size, spacing=1.1)


def _border(cell, color=None, width_pt=0.75):
    """只畫下框線；左右上清空。框線元素要排在填色元素之前（DrawingML 規定順序）。"""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        old = tcPr.find(qn(tag))
        if old is not None:
            tcPr.remove(old)
    for i, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        ln = tcPr.makeelement(qn(tag), {"w": str(int(width_pt * 12700)), "cap": "flat", "cmpd": "sng",
                                        "algn": "ctr"})
        if tag == "a:lnB" and color is not None:
            fill = ln.makeelement(qn("a:solidFill"), {})
            clr = fill.makeelement(qn("a:srgbClr"), {"val": str(color)})
            fill.append(clr)
            ln.append(fill)
        else:
            ln.append(ln.makeelement(qn("a:noFill"), {}))
        tcPr.insert(i, ln)


def table(slide, x, y, rows, col_w, header_size=15, body_size=14, row_h=0.5, bold_first_col=False, header_h=None):
    """rows[0] 是表頭；儲存格可給 str 或 (str, bold)。回傳 shape。"""
    n_r, n_c = len(rows), len(rows[0])
    shp = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n_r))
    t = shp.table
    t.horz_banding = False
    t.first_row = True
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches((header_h or min(row_h, 0.5)) if i == 0 else row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            _border(cell, RULE if i < n_r - 1 else BLUE, 0.75 if i else 1.0)
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEAD if i == 0 else WHITE
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            text = val[0] if isinstance(val, tuple) else val
            bold = i == 0 or (isinstance(val, tuple) and val[1]) or (bold_first_col and j == 0)
            for k, part in enumerate(text.split("\n")):
                par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                if k == 0:
                    par.text = ""
                par.line_spacing = 1.0
                _run(par, part, header_size if i == 0 else body_size, bold, INK)
    return shp


# ---------------------------------------------------------------- 頁型
def table_slide(title, desc, rows, col_w, footer=None, source=None, notes=None, y=BODY_TOP, **kw):
    s = content_slide(title, desc, footer, source, notes)
    table(s, L, y, rows, col_w, **kw)
    return s


def fig_slide(title, desc, img, footer=None, source=None, notes=None, box=None):
    s = content_slide(title, desc, footer, source, notes)
    x, y, w, h = box or (L, BODY_TOP, CW, BODY_BOTTOM - BODY_TOP)
    picture_fit(s, img, x, y, w, h, valign="center")
    return s


def fig_text_slide(title, desc, img, items, footer=None, source=None, notes=None, img_w=7.3, head=None,
                   size=16):
    """左圖右文；items 為右欄條列，head 為右欄藍色小標。"""
    s = content_slide(title, desc, footer, source, notes)
    picture_fit(s, img, L, BODY_TOP, img_w, BODY_BOTTOM - BODY_TOP, valign="center")
    tx = L + img_w + 0.3
    lines = [(head, True, BLUE)] if head else []
    lines += [("・" + it, False) if isinstance(it, str) else it for it in items]
    textbox(s, tx, BODY_TOP + 0.05, R - tx, BODY_BOTTOM - BODY_TOP, lines, size=size, spacing=1.12, after=7)
    return s


def bullets_slide(title, desc, cols, footer=None, source=None, notes=None, size=16):
    """cols: [(小標, [條列...]), ...]，一欄或兩欄文字。"""
    s = content_slide(title, desc, footer, source, notes)
    n = len(cols)
    gap = 0.4
    cw = (CW - gap * (n - 1)) / n
    for k, (head, items) in enumerate(cols):
        x = L + k * (cw + gap)
        lines = [(head, True, BLUE)] + [("・" + it, False) for it in items]
        textbox(s, x, BODY_TOP, cw, BODY_BOTTOM - BODY_TOP, lines, size=size, spacing=1.12, after=8)
    return s


def appendix_table(title, desc, rows, source, footer=None):
    """附錄精簡總表：編號、日期、實驗、結果與結論；字小、一列一行。"""
    return table_slide(title, desc, [["編號", "日期", "實驗", "結果與結論"]] + rows, [0.85, 1.45, 3.35, 6.52],
                       footer=footer, source=source, header_size=12, body_size=11, row_h=0.29)


# ================================================================ 封面
s = prs.slides.add_slide(BLANK)
rect(s, 0.9, 2.2, 0.09, 2.0, BLUE)
textbox(s, 1.25, 2.2, 11.3, 0.9, [("像素化天線與濾波器 AI 設計", True)], size=40)
textbox(s, 1.25, 3.1, 11.3, 0.7, [("歷來實驗總整理：四代方法與成果，2023–2026", True)], size=26, color=BLUE)
textbox(s, 1.25, 4.45, 11.3, 0.5, ["鄒穎麒　中正大學電機所　2026 年 10 月"], size=18, color=MUTED)
page_number(s)
set_notes(s, ["範圍：陳廷茂、錢鵬予、吳維文三代與我們第四代的天線與濾波器工作，含十月最新成果。"
              "數字以同夾 實驗總表.md 為準。"])

# ================================================================ 摘要
table_slide(
    "四代一覽", "同一個問題做了四代：用 AI 生成 25×25 像素金屬圖形，讓 HFSS 模擬的響應通過規格。",
    [["世代", "期間", "核心方法", "代表成果"],
     [("第一代\n陳廷茂", True), "2023/07–\n2024/08", "MPGN 生成器加 GEN 代理模型的線上學習，不需資料集",
      "四組設計 loss 收斂到接近 0" + SR + "；每次設計 1,500 次模擬"],
     [("第二代\n錢鵬予", True), "2024/08–\n2025/07", "BiScaleNorm 解二值化梯度飽和；變異加回滾的二階段生成",
      "約第 500 次收斂，原設計約第 1,000 次" + SR + "；開始向日月光報告"],
     [("第三代\n吳維文", True), "2025/07–\n2026/06", "ACP 單階段、熱啟動、DLF 菁英篩選、SC 連通 loss",
      "三機制合用 loss 0.99、饋入連通率 61.98%" + SR + "；5 個實作樣品量測"],
     [("第四代\n我們", True), "2026/06–\n2026/10", "批次線上學習：從前代資料池起步，判準寫死、SM 初篩、三台 HFSS 批次驗證",
      "單埠 +0.79 dB、對稱天線 +0.599 dB、寬頻濾波器 +0.4104 dB 達標" + NOT]],
    [1.25, 1.45, 4.6, 4.87], "四代共用同一個骨架：生成器提案、HFSS 驗證；第四代補上可製造與公證",
    "實驗總表 §1", row_h=0.95, body_size=14)

table_slide(
    "目前最佳成果", "第四代各條線的最佳值與狀態；WM 是帶內最差餘裕，正值代表全部規格通過。",
    [["項目", "最佳值", "條件", "狀態"],
     ["單埠天線", "+0.79 dB" + NOT, "S0 網格、無橋", "合格；加 0.10 mm 菱形橋後 +0.58" + NOT],
     ["可製造天線代表", "+0.69／+0.77 dB" + ONE, "S0 網格、0.10 mm 橋", "8/31 已交付"],
     ["完全左右對稱天線", "+0.599 dB" + NOT, "S0 網格、0.10 mm 橋", "合格"],
     ["雙埠濾波器，規格 v2", "−2.39 dB" + NOT, "可製造幾何、0.075 mm 橋", "未過規格"],
     ["寬頻濾波器 v6a，阻帶 −15 dB", "+0.4104 dB" + NOT, "與天線同級網格、0.075 mm 橋", "達標，交付首選"],
     ["寬頻濾波器 v6，阻帶 −20 dB", "−3.05 dB" + ONE, "同上", "未達標，10/23 暫停"],
     ["實作量測", "第三代 5 個樣品", "量測通帶比模擬高約 0.5–1.6 GHz", "第四代 39 個設計尚未實測"]],
    [3.2, 2.45, 3.2, 3.32], "所有第四代數字都是 HFSS 模擬；實作驗證要等板子回來",
    "實驗總表 §1、§2.10、§2.11；docs/records.json、records_dual.json", row_h=0.56)

bullets_slide(
    "範圍與限制", "這份整理只彙整既有紀錄，不新跑模擬；先講清楚能比什麼、不能比什麼。",
    [("範圍", ["四代：陳廷茂、錢鵬予、吳維文、我們，2023/07–2026/10",
             "天線與濾波器兩類工作，方法與成果並重，共整理 223 項實驗",
             "來源：碩論、投影片、週報、會議紀錄、第四代研究日誌與紀錄檔",
             "日月光合作從第一代的天線系列就開始；與日月光的接洽由鄭國宏學長負責"]),
     ("限制", ["幾乎全是 HFSS 模擬；實作量測只有第三代 2026/03/24 報告的 5 個樣品與 pixel 顯微量測",
             "第四代 8/31 交付的 39 個設計尚未實測",
             "各代 loss 定義不同，loss 數值不跨代比；跨代只看規格過了沒，第四代起用 dB 尺 WM",
             "前三代以 S11 與增益為目標，不含 radiation pattern 與可製造性；定義不同，不構成評價",
             "天線紀錄只在 S0 網格下成立，換較細網格會移動約 0.7–1.0 dB"])],
    "模擬不等於量測：所有性能數字都要等實作驗證", "實驗總表 §0、§4.3", size=16)

table_slide(
    "數字分級與讀法", "每個性能數字後面都帶一個分級標記，說明這個數字從哪裡來、可信到什麼程度。",
    [["標記", "意思", "例子"],
     ["〔學長報告〕", "前三代論文、投影片、週報或結果夾名中的模擬值", "第三代三機制合用 Response loss 0.99"],
     ["〔圖讀〕", "原檔只有曲線圖，從圖上肉眼讀出的近似值", "第一代四層板 26.5 GHz S11 約 −9.3 dB"],
     ["〔量測〕", "打樣後實際量到的值", "第三代 S1 樣品 28 GHz 增益 3.0 dBi"],
     ["〔單次〕", "第四代只模擬一次的值，或由單次模擬算出的統計量", "寬頻濾波器 v6 最佳 −3.05 dB"],
     ["〔公證〕", "第四代同設定重跑兩次以上、結果一致才承認", "單埠 WM 紀錄 +0.79 dB"],
     ["WM", "帶內最差餘裕：每條規格在每個評分頻點離門檻多少 dB，取最差一點", "正值代表全部規格通過"]],
    [2.0, 5.6, 4.57], "HFSS 在固定設定下位元級可重現，所以公證值可以直接引用",
    "實驗總表 §0", row_h=0.56)

# ================================================================ 00 總覽
section_slide("00", "總覽", "時間軸、日月光歷次要求、規格演進")

s = content_slide("四代時間軸", "四代各一條帶，上方是日月光要求與學長傳達；2026/06 之後時間軸放大 3.5 倍。",
                  "四代一條主線：第一代立骨架，第四代把可製造、公證與交付補進流程", None,
                  "出處：script/figs/report_generations.py；日期取自實驗總表 §1、§2.10、§2.11。"
                  "2026/06 之後第四代事件密集，圖上時間軸放大 3.5 倍並以淺底標示。")
picture_fit(s, FIG / "timeline_generations.png", L - 0.15, 1.62, CW + 0.3, 5.1, valign="center")

table_slide(
    "日月光歷次要求：2025 年", "前兩代的報告多在講 AI 方法；明確寫下的日月光要求集中在 2025 年。",
    [["日期", "來源", "要求內容", "回應"],
     ["2024/12/05", "sunmoon 簡報", "書面上最早標明日月光的簡報，報 AI 方法進度", "此後約一到兩個月一次"],
     ["2025/05/01", "日月光工程師", "濾波天線從架構上改動；新增不同截面的響應來看", "第二代：雙埠濾波、3D 場型"],
     ["2025/06", "日月光 6 月簡報", "「根據上次要求，嘗試更改規格」", "第二代：n260 遷移、雙埠濾波"],
     ["2025/10/02", "日月光", "先做天線、暫停濾波天線；更改 loss 計算方式；做至少一軸對稱的天線", "第三代 MinMax loss、Mirror 8；對稱由第四代回應"],
     ["2025/10", "月報備註、會議紀錄", "雙埠 S21 改成 −3 dB；不希望分階段訓練", "第三代 One-Stop 單階段"],
     ["2025/10–11", "Review 回饋，來源未標", "結合選擇與對稱、U 型 C 型小圖案、用現有好設計當起點", "第三代形狀庫、熱啟動"]],
    [1.45, 2.2, 4.85, 3.67], "2025/10 的三項要求定下後續方向：先天線、改 loss、至少一軸對稱",
    "實驗總表 §1、§2.10；Q34、Q35 說明來源分級", row_h=0.6)

table_slide(
    "日月光要求與學長傳達：2026 年", "第四代的規格與方向多由鄭國宏學長傳達，以下照時間列出與我們的回應。",
    [["日期", "來源", "內容", "回應"],
     ["2026/05/28", "ASE-CCU 會議", "第三代下一步：場型規格納入 loss", "第四代 6 月接上 radiation pattern"],
     ["2026/07 上旬", "日月光工程師", "提供三篇像素化逆設計文獻", "第四代在對稱化空間內做 GA 對照"],
     ["2026/08/07", "學長傳達", "單埠 pattern 好就可收尾；帶外壓不下去可交給後級濾波器", "單埠只管帶內、radiation pattern、可製造"],
     ["2026/08/12", "學長裁定", "雙埠規格白天六軸全硬，當晚放寬成 v2：阻帶 −15 dB、帶外退場", "雙埠濾波器改用 v2 計分"],
     ["2026/08/31", "鄭國宏學長安排", "送板交付", "交付 39 個設計"],
     ["2026/09", "委託方", "平台兩條硬約束：沒有 AI 也要能跑一整晚、不綁定特定策略", "emforge 平台受控試用"],
     ["2026/10/07 起", "學長傳達", "對稱天線；寬頻濾波器規格 v1 到 v6", "對稱天線 +0.599、寬頻 v6a +0.4104" + NOT]],
    [1.6, 1.95, 4.75, 3.87], "第四代的每項要求都有對應的實驗與結果",
    "實驗總表 §1、§2.10、§2.11；notes/gen4 §4", row_h=0.56)

table_slide(
    "規格演進：天線", "n257、S11 ≤ −10 dB、增益 ≥ 4 dBi 從第一代沿用至今，之後逐步加上方向與製造條件。",
    [["世代", "時間", "規格", "備註"],
     ["第一代", "2023/07 定題", "雙層 PCB 5×5×0.508 mm、28 GHz、增益 5 dBi、頻寬 3 GHz、3dB 波束寬 80°", "程式改成 24–32 GHz 17 點"],
     ["第一代", "論文", "n257 26.5–29.5 GHz 7 點：S11 < −10 dB、增益 > 4 dBi", "第二至四代沿用"],
     ["第一代", "論文濾波版", "24–32 GHz 17 點；帶外 S11 > −3 dB、增益 < −10 dBi", "第二代沿用"],
     ["第二代", "2025/06", "換頻段：n260 37–40 GHz", "只有初步曲線"],
     ["第三代", "2025/10", "天線只算 26.5–29.5 GHz，S11 ≤ −10、增益 ≥ 4，只罰沒達標的一側", "日月光要求先做天線"],
     ["第四代", "6/24–7/12", "加 radiation pattern：28 GHz 兩切面 ±45° 內不低於峰值 −3 dB；合格另要 WM ≥ +0.15", "窗口由 ±55° 改 ±45°"],
     ["第四代", "8/07–10/07", "帶外交給後級濾波器；出圖帶 0.10 mm 菱形橋；金屬精確左右鏡射", "學長傳達"]],
    [1.15, 1.75, 6.4, 2.87], "天線規格的核心沒變，第四代加上 radiation pattern、可製造與對稱",
    "實驗總表 §2.10", row_h=0.56)

table_slide(
    "規格演進：濾波器", "濾波器從第二代的雙埠開始，第四代的寬頻規格在一個月內改了七版。",
    [["世代", "時間", "規格", "備註"],
     ["第二代", "2025/05", "雙埠濾波：看 S11、S22、S21", "沒有寫數字規格"],
     ["第三代", "2025/07–10", "S11、S21、S22 上下限；S21 帶內改成 −3 dB", "−3 dB 出自日月光備註"],
     ["第三代", "2026/01–03", "改成 Return loss 與 S21 兩張圖；量測規格 RL > 10 dB", "—"],
     ["第四代", "8/10 v1", "S11/S22 ≤ −12；S21 通帶 25.5–30.5 GHz ≥ −3；阻帶 24–25、31–32 GHz ≤ −20", "前代資料最佳差 7.20 dB" + ONE],
     ["第四代", "8/12 v2", "S11/S22 ≤ −10；阻帶放寬到 ≤ −15；帶外退場", "學長：「−20 太難，先 −15」"],
     ["第四代", "10/07 寬頻 v1", "S21 通帶 26–30 GHz ≥ −3；阻帶 16–20、36–40 GHz ≤ −20；16–40 GHz", "之後一個月改七版"],
     ["第四代", "10/15 寬頻 v6a／v6", "阻帶 11–19、36–40 GHz ≤ −15／−20；S11/S22 ≤ −10；11–40 GHz", "壓 Ku band 與 n260"]],
    [1.15, 2.0, 6.15, 2.87], "濾波器規格越來越寬頻：從 n257 附近擴到 11–40 GHz",
    "實驗總表 §2.10；七版細節見第四代章節", row_h=0.56)

# ================================================================ 01 第一代
section_slide("01", "第一代　陳廷茂", "2023/07–2024/08　MPGN 生成器＋GEN 代理模型的線上學習")

table_slide(
    "第一代方法演進", "從自建網路、強化學習到合成梯度，最後定案為 MPGN 加 GEN 的線上學習。",
    [["時間", "做法", "結局"],
     ["2023/07", "定題：雙層 PCB、28 GHz；不要資料集，用神經網路直接設計", "與實驗室 RIS 線的「先收資料」做法不同"],
     ["2023/08–10", "numpy 自建全連接網路，加 STE 讓 0/1 二值化也能傳梯度", "轉 PyTorch；STE 保留進論文"],
     ["2023/10", "強化學習：Policy Gradient、DDPG、Dueling DQN、PPO", "4.7 GHz 跑 2096 次 loss 平在約 8.8" + EYE + "，放棄"],
     ["2023/10", "DNI 合成梯度：用另一個網路估計梯度", "演化成 GEN 代理模型"],
     ["2023/11", "MPGN＋GEN 線上學習：每迭代 1 次 HFSS，GEN 用全部累積資料重訓", "論文定案，之後三代都沿用這個骨架"],
     ["2023/12–2024/04", "上下界區間 loss、四層板、GEN 只在後兩層加 BN、試加波束寬條件", "上下界 loss 進論文；波束寬未進論文"],
     ["2024/06/25", "口試；2024/07 定稿", "2024/08/07 交接"]],
    [1.9, 6.0, 4.27], "第一代立下四代共用的骨架：生成器出圖、HFSS 驗證、代理模型借梯度",
    "實驗總表 §3.1；notes/gen1 §2", row_h=0.56)

fig_slide(
    "MPGN＋GEN 線上學習架構", "生成網路 MPGN 輸出 0/1 圖形送 HFSS；GEN 學「圖形→響應」，把梯度借回 MPGN。",
    FIG / "gen1" / "g1_arch_mpgn_gen.png", "HFSS 不能算梯度，所以另訓一個代理模型把梯度借回生成器",
    "陳廷茂碩論圖 3-1",
    "每迭代只跑 1 次 HFSS；GEN 每迭代用全部累積資料重訓到 MSE<0.0001；前 500 次沒往 0 走就提前停。")

table_slide(
    "第一代論文五個實驗", "雙層板與四層板、天線與濾波天線各做一組，另比較饋入固定與激活函數。",
    [["實驗", "設定", "結果"],
     ["一、饋入固定金屬", "雙層板 n257；饋入點周圍 8×8 格強制為金屬", "有固定時「更加平穩快速」收斂，看曲線"],
     ["二、激活函數", "Tanh、LeakyReLU、PReLU", "PReLU 快速收斂到接近 0" + SR + "，定 PReLU"],
     ["三、四層板 n257", "林佑宣四層板；M1 25×25、M2 48×22", "loss 接近 0" + SR + "；26.5 GHz S11 約 −9.3 dB" + EYE],
     ["四、四層板濾波天線", "24–32 GHz 17 點上下界", "loss 趨近 0" + SR + "；帶外兩點 S11 未達 −3 dB 下界" + EYE],
     ["五、雙層板濾波天線", "同濾波規格、雙層板小金屬區", "約 800 次後接近 0" + SR + "；目視一兩點未過" + EYE],
     ["模擬次數比較", "以四層板濾波為例", "本法 1,500 次；GA＋CNN 110 萬次" + SR]],
    [2.65, 4.4, 5.12], "四組設計 loss 都收斂到接近 0；目視讀圖各有一兩個頻點沒過",
    "實驗總表 §3.1 1-18～1-26", row_h=0.6)

fig_slide(
    "雙層板 n257 天線結果", "S11 在 26.5–29.5 GHz 全低於 −10 dB；增益 26.5–29 GHz 約 4–5.3 dBi，29.5 GHz 約 3.5 dBi" + EYE + "。",
    FIG / "gen1" / "g1_exp_2layer_n257.png", "第一代的定案設定在雙層板 n257 天線上收斂",
    "陳廷茂碩論表 4-3，p.44")

fig_slide(
    "四層板濾波天線結果", "四層板 M1、M2 兩層可調金屬；帶內 28、29 GHz 增益約 3.4、3.6 dBi" + EYE + "。",
    FIG / "gen1" / "g1_exp_4layer_filter.png", "濾波天線是論文主打成果；loss 接近 0 不等於逐點全過",
    "陳廷茂碩論表 4-5，p.46")

fig_text_slide(
    "饋入固定金屬：實驗一", "MPGN 輸出疊上 8×8 饋入固定遮罩後才送 HFSS。",
    FIG / "gen1" / "g1_feedfix_mask_and_loss.png",
    ["饋入點周圍 8×8 格，約邊長三分之一，強制為金屬", "沒有固定時 loss 震盪大", "有固定時「更加平穩快速」收斂，只有曲線、沒有數字",
     "之後所有實驗都固定饋入金屬"],
    "固定饋入金屬讓收斂更穩，四代都保留這個做法", "陳廷茂碩論圖 4-2～4-4", img_w=6.6, head="做法與結果")

fig_text_slide(
    "網格解析度：25×25 定案", "28 GHz 同流程比較 50×50 與 25×25 兩種網格。",
    FIG / "gen1" / "g1_mesh_50x50_vs_25x25.png",
    ["50×50 minloss 40.36、25×25 minloss 1.48" + SR, "定 25×25、每格 0.2 mm，理由是格越細模擬越久",
     "50×50 那兩輪沒加饋入固定金屬，比較有混淆", "第四代 8 月在濾波器上試 50×50 探針，9/12 決定不做"],
    "25×25、每格 0.2 mm 從第一代沿用到第四代", "NAS 1_Result/1120、1121；實驗總表 1-09", img_w=5.2, head="比較與限制")

# ================================================================ 02 第二代
section_slide("02", "第二代　錢鵬予", "2024/08–2025/07　BiScaleNorm＋變異與回滾的二階段生成")

table_slide(
    "第二代方法演進", "先解二值化造成的梯度飽和，再用變異加回滾跳出局部解。",
    [["時間", "做法", "結局"],
     ["2024/08", "接手；試 CNN 生成器、預訓練初始圖形", "DCGAN 有棋盤效應，維持 MLP"],
     ["2024/09", "對稱生成、Update Image、Selection Model", "都沒有延續；對稱的圖沒有存到"],
     ["2024/10–11", "優化器比較、MSE 換 Smooth L1、Sign 替代品、GAN 變體", "Smooth L1 進論文，其餘未採用"],
     ["2024/12–2025/03", "非對稱正規化，加 EMA 後命名 BiScaleNorm", "論文實驗一"],
     ["2025/01", "週期變異加回滾，組成二階段生成", "論文實驗二、三"],
     ["2025/02–04", "獨立金屬群處理、代理模型梯度誤差、Over Unite", "孤島決定不刪；0.21 mm 沿用至今"],
     ["2025/05–06", "雙埠濾波、n260 遷移；2025/06/17 口試", "進日月光 6 月簡報"]],
    [1.95, 6.0, 4.22], "第二代把重心放在訓練穩定性與跳出局部解",
    "實驗總表 §3.2；notes/gen2 §2", row_h=0.56)

fig_text_slide(
    "BiScaleNorm：解二值化的梯度飽和", "生成器的 PReLU 換成正負兩側分別縮放的 BiScaleNorm，比較三種學習率。",
    FIG / "gen2" / "g2_biscalenorm_lr_table.png",
    ["學習率 3e-3、3e-4：原設計梯度消失，BiScaleNorm 穩定收斂" + SR,
     "學習率 3e-5：兩者都收斂，BiScaleNorm 約第 500 次、原設計約第 1,000 次" + SR,
     "第三代沿用 BiScaleNorm"],
    "BiScaleNorm 讓訓練不再依賴碰運氣的學習率", "錢鵬予口試投影片 p51；碩論 §4.2.1", head="結果")

fig_text_slide(
    "二階段生成：週期變異", "第一階段找局部解；第二階段每 400 次隨機翻轉 5% 像素，逼生成器跳出。",
    FIG / "gen2" / "g2_twostage_mutation_diff.png",
    ["第一階段約第 400 次出現明顯收斂" + SR, "弱點在 29 GHz：S11 約 −8 dB、增益約 1.7 dB" + EYE,
     "第二階段第 1,200 次後跳出原解空間，29 GHz 兩項都改善" + SR, "右圖黑點是兩階段最佳解之間的差異位置"],
    "變異讓生成器跳出第一階段的局部解", "錢鵬予口試投影片 p55；碩論 §4.2.2", head="結果")

fig_text_slide(
    "回滾機制消融", "變異前若當下不如歷史最佳，就先退回最佳圖再變異；比較有無回滾。",
    FIG / "gen2" / "g2_rollback_ablation.png",
    ["無回滾：「整體收斂品質明顯發散」", "有回滾：「受到歷史最佳解的約束下，穩定收斂」", "週期 400、2,000 次迭代，只有曲線",
     "第四代已移除回滾，探索改交給多候選與代理模型引導"],
    "回滾讓變異不至於把好解丟掉", "錢鵬予口試投影片 p58；碩論 §4.2.3", head="結果")

table_slide(
    "第二代試過但沒進論文的做法", "很多嘗試只有定性結論，這裡列出原因，避免重複踩坑。",
    [["做法", "目的", "結果"],
     ["CNN 生成器", "取代 MLP", "反卷積有棋盤效應，Upsampling 會大片填滿"],
     ["Sign 函數替代品", "讓二值化更平滑", "Approx 最穩，但模擬前仍要硬二值化，判定「雞肋」"],
     ["GAN 變體", "加判別器與分類器", "「很炫但沒用」"],
     ["SWA、餘弦退火", "跳出局部解", "「觀察起來有跳，但是又會回到相同的位置」"],
     ["變異自動觸發", "仿 SWATS 決定何時變異", "「永遠觸發不了這個機制」，維持固定週期 400"],
     ["Selection Mode", "每格從 8 種 2×2 子圖案選一個", "RIS 上「一鳴驚人」，接代理模型就訓不起來"],
     ["代理模型重訓", "修正擬合與梯度誤差", "1,000 筆資料後重訓要 60,000 次迭代" + SR]],
    [2.6, 3.3, 6.27], "代理模型「擬合值對但導數不對」，是第二代留下的核心問題",
    "實驗總表 §3.2", row_h=0.56)

fig_text_slide(
    "日月光工程師的兩項要求", "2025/05/01 的二階段生成報告結論頁，列出合作方需求並逐項回應。",
    FIG / "gen2" / "g2_ase0501_requests.png",
    ["日月光工程師：濾波天線從架構上改動；新增不同截面的響應", "543 老師：碎形天線不符一般認知；能否基於專家知識生成",
     "回應：在另一端新增第二個 port 做雙埠、看 3D 場型", "2025/06 再依「上次要求」換到 n260 頻段"],
    "雙埠濾波與換頻段，都源自日月光 2025/05 的要求", "二階段生成天線.pptx p57；實驗總表 §1", head="要求與回應")

fig_text_slide(
    "雙埠濾波與 n260 遷移", "微帶線從圖形兩側穿過，固定金屬遮罩改成上下各一塊。",
    FIG / "gen2" / "g2_ase06_dualport_mask.png",
    ["雙埠看 S11、S22、S21；兩端微帶線各 7.5 mm", "第一、二階段都只放曲線，沒寫是否達標",
     "n260 37–40 GHz：「測試結果與趨勢和 n257 一致」，沒有數字", "第三代接手後以雙埠為主，用途是濾波"],
    "雙埠濾波器從第二代開始，一路延續到第四代", "日月光 6 月簡報 p1、p3；實驗總表 2-26、2-28", head="做法與結果")

# ================================================================ 03 第三代
section_slide("03", "第三代　吳維文", "2025/07–2026/06　ACP 單階段＋熱啟動＋DLF＋SC 連通 loss")

table_slide(
    "第三代方法演進", "從二階段與多種擴增，收斂到單階段 ACP；孤島抑制依序試了四種 loss。",
    [["時間", "做法", "結局"],
     ["2025/07", "接手；雙埠舊法重跑、變異自動觸發、Dual Mirror、七巧板 v1", "8 月 Future Work 寫「效果不佳」"],
     ["2025/10", "日月光三項要求；MinMax loss、七巧板 v2、Mirror 8、ACP 提出", "MinMax 延續成碩論的邊界 loss"],
     ["2025/11", "熱啟動 Base-1、Base-2；菁英複習、全資料重訓", "熱啟動進碩論"],
     ["2025/12–2026/02", "τ 二值化取代變異；TV→IS→GC→SC 孤島 loss；DMCVAE", "SC 連通 loss 進碩論"],
     ["2026/03", "饋入連通率 R_feed、One-Stop 單階段；3/24 報告 5 個樣品量測", "設計時間約為舊法 50%" + SR],
     ["2026/05", "9×9 饋入、TV／KNS／SC 比較、pixel 顯微量測", "場型納入 loss 列為下一步"],
     ["2026/06", "碩論：ACP、DLF、SC 三機制；6/30 口試", "loss 0.99、R_feed 61.98%" + SR]],
    [1.95, 6.3, 3.92], "第三代的主線是單階段 ACP 與訓練中的孤島抑制",
    "實驗總表 §3.3；notes/gen3 §2", row_h=0.56)

fig_text_slide(
    "七巧板：形狀庫拼圖", "每格不填 0/1，改從形狀庫挑小形狀拼滿 25×25，想用粗形狀避免不導電。",
    FIG / "gen3" / "g3_tangram_shapes_library.png",
    ["v1 2025/07：8 個 5×5 粗形狀；雙埠響應最低 −15、−19、−15 dB" + SR + "，「效果不太好」",
     "v2 2025/10：14、16 個細形狀；11 月擴成 35 種，含 T、L、U 型各種旋轉",
     "2025/11：Gumbel-Softmax 對 Gumbel-Sinkhorn 兩種可微挑選", "三版都只有圖、沒有數值結論，沒進碩論"],
    "七巧板只在第三代正式做過，之後沒有延續", "202511 - ASE s11；實驗總表 §2.2", img_w=6.0, head="三個版本")

fig_text_slide(
    "對稱：Mirror 8 翻轉", "兩軸對稱共 8 種翻轉，先用代理模型預測，只把最好的 1–2 個送 HFSS。",
    FIG / "gen3" / "g3_mirror8_flips_sm.png",
    ["代理模型預測 loss：7 種翻轉 11.02–22.43，最佳 6.43" + SR, "都是代理模型預測值，不是 HFSS 值",
     "回應日月光 2025/10「至少一軸對稱」", "之後的 DMCVAE 也做鏡射；兩者都沒進碩論"],
    "第三代的對稱停在代理模型預測，沒有 HFSS 數值結論", "202511 - ASE s10；實驗總表 3-11", img_w=7.6, head="結果")

fig_text_slide(
    "饋入固定區：單埠與雙埠", "生成圖疊上固定饋入遮罩：單埠下方一塊，雙埠上下各一塊。",
    FIG / "gen3" / "g3_feed_mask_single_dual_slide.png",
    ["2025/07–09 比較有無固定：「無固定的結果比較好」，條件是只在變異後才套上", "2026/05 固定區由 5×5 擴為 9×9",
     "9×9 時 R_feed 約 52.42%" + SR + "，「S11 與 Gain 有初步達標」", "碩論附錄寫 9×9"],
    "第三代把固定饋入區擴大到 9×9", "G07 s20；CCU0528 s15；實驗總表 §2.3", head="做法與結果")

table_slide(
    "孤島抑制：四種 loss", "R_feed＝與饋入點連成一塊的金屬占總金屬的比率，越高代表孤島越少；條件是單埠。",
    [["loss", "做法", "R_feed" + SR],
     ["TV 總變差", "懲罰相鄰像素的差異", "權重 50：18.53%；權重 100：59.62%"],
     ["IS 孤立像素", "懲罰孤立的像素；CCU0528 的 KNS 推測即 IS", "權重 1：33.02%；權重 100：61.00%"],
     ["GC 閉運算", "把形態學閉運算寫進 loss", "權重 100：53.95%，被 SC 取代"],
     ["SC 圖譜連通度", "圖樣當 graph，取 Fiedler 值 λ₂ 的倒數", "低權重 31.13%；高權重 87.80%"],
     ["對照 Base-2", "不加孤島 loss", "13.75%"],
     ["雙埠 TV100", "濾波器加 TV 與區間 loss", "94.03%：「讓兩個饋入點導通」"]],
    [2.4, 5.6, 4.17], "SC 是全域抑制、碩論採用；權重加大要和響應取捨",
    "AI設定 202601–202603、CCU0528 s21；實驗總表 §2.4。各列條件不同，只在同列內比", row_h=0.56)

fig_text_slide(
    "孤島抑制與場型", "Base、TV、SC 三種設定的金屬圖形與輻射場型；綠色是與饋入點連通的金屬。",
    FIG / "gen3" / "g3_island_radpattern_slide.png",
    ["R_feed：Base 14.01%、TV 22.39%、SC 61.98%" + SR, "SC 的「0 度主瓣極為飽滿、勻稱」",
     "SC 權重過高時場型反而變差", "碩論摘要：連通性由 10–20% 提升到 60% 以上"],
    "連通 loss 拉高饋入連通率，也改善了場型", "CCU0528 s22；碩論表 4-3", img_w=7.0, head="結果")

fig_text_slide(
    "ACP 自適應循環排程", "偵測 loss 停滯就重啟學習率與二值化溫度 τ，取代靠運氣的變異。",
    FIG / "gen3" / "g3_acp_lr_tau_cycle.png",
    ["一個週期：warmup 0.2 加餘弦退火，學習率與 τ 同步", "重啟方式試過 reset、peak、linear，Linear 較好",
     "把錢鵬予的兩階段合成一段，日月光也表示不希望分階段訓練", "停滯判定 patience 採程式值 25"],
    "ACP 把探索從隨機變異改成有排程的重啟", "G12 s17；實驗總表 3-14、3-24", img_w=6.8, head="做法")

table_slide(
    "排程比較與三機制消融", "碩論兩張表：五種排程各跑 500 epoch，再看 ACP、DLF、SC 單獨與合用。",
    [["設定", "Response loss" + SR, "R_feed" + SR],
     ["Base", "3.64", "27.22%"],
     ["CyclicLR／ReduceLROnPlateau", "5.4／17.02", "59.05%／42.12%"],
     ["Cyclical Mutation，即第二代方法", "4.72", "34.17%"],
     ["ACP", "2.07", "18.89%"],
     ["DLF 菁英篩選", "1.71", "16.28%"],
     [("ACP＋SC＋DLF", True), ("0.99", True), ("61.98%", True)]],
    [6.0, 3.1, 3.07], "三機制合用同時拿到最低 loss 與最高連通率",
    "碩論表 4-2、表 4-4；G12 s18–19。loss 是第三代定義，不跨代比", row_h=0.58)

fig_text_slide(
    "熱啟動 Base-1 與 Base-2", "用理論公式算出 28 GHz 初始貼片當起點，不從空白開始。",
    FIG / "gen3" / "g3_warmstart_base12.png",
    ["理論公式出自鄭國宏學長", "Base-2「一開始就已在 28 GHz 產生深度諧振」", "碩論寫熱啟動可「大幅改善表現」",
     "2025/12 可重複性兩次實驗，響應形狀相近，只有圖"],
    "熱啟動讓線上學習從已知會共振的形狀出發", "P10 s11；實驗總表 3-19、3-21", img_w=7.0, head="結果")

fig_text_slide(
    "One-Stop 單階段流程", "2026/03/24 給日月光的流程：用 ACP 取代二階段，減少 HFSS 呼叫次數。",
    FIG / "gen3" / "g3_onestop_flow_slide.png",
    ["舊流程兩段各約 3 天、共 6 天，呼叫 HFSS 數百次" + SR, "6 天流程中成功場型比例約 1/400" + SR,
     "One-Stop 設計時間約為舊法 50%" + SR, "流程內含孤島抑制與經驗回放"],
    "第三代給日月光的主訴求：設計時間減半", "202603 - Final s36–37；實驗總表 3-34", img_w=7.0, head="流程指標")

table_slide(
    "5 個實作樣品量測", "2026/03/24 ASE-CCU 報告單元天線實測，無反射室；模擬含 K Connector 與 ENIG 化金層。設計者待確認。",
    [["樣品", "模擬通帶 RL>10 dB" + SR, "量測通帶" + MEAS, "28 GHz 增益 dBi\n模擬／量測", "29.5 GHz 增益 dBi\n模擬／量測"],
     ["S1", "26.1–30.0 GHz", "26.9–31.6 GHz", "2.2／3.0", "3.2／2.2"],
     ["S2", "26.2–30.1 GHz", "26.9–31.6 GHz", "2／2.7", "3.2／5.0"],
     ["S3", "26.2–29.9 GHz", "26.7–31.4 GHz", "1.3／2.7", "4.2／4.0"],
     ["S4", "26.1–30.2 GHz", "26.7–31.6 GHz", "0.9／2.1", "4.4／5.0"],
     ["S5", "26.2–30.1 GHz", "26.7–31.5 GHz", "1.5／2.4", "4.4／4.8"]],
    [1.2, 2.9, 2.9, 2.6, 2.57], "量測通帶都比模擬往高頻偏約 0.5–1.6 GHz，投影片沒有解釋原因",
    "202603 - Final s19–34；實驗總表 §2.11", row_h=0.6,
    notes="設計者待確認：Final 列在量測前的設計流程是兩段各約 3 天，也就是錢鵬予的二階段法；只是推測，報告寫待確認。")

fig_text_slide(
    "Sample 1：實物與量測場型", "實物照、金屬圖形，以及 26.5、28、29.5 GHz 兩個截面的量測場型。",
    FIG / "gen3" / "g3_meas_sample1_slide.png",
    ["S1 量測通帶 26.9–31.6 GHz" + MEAS + "，模擬 26.1–30.0 GHz" + SR, "28 GHz 增益：量測 3.0 dBi" + MEAS + "、模擬含 ENIG 2.2 dBi" + SR,
     "量測 S11 帶內標註 −8.9 dB" + MEAS, "2026/05/28 顯微鏡量 pixel：0.188–0.231 mm" + MEAS + "，模擬 0.21 mm"],
    "四代中唯一的實作量測；製造面的過蝕刻判斷之後沒有結果", "202603 - Final s20–21；CCU0528 s2", img_w=7.2, head="量測")

table_slide(
    "1×4 陣列模擬", "用各樣品單元組成 1×4 陣列，板子 35 mm × 36 mm；數值是峰值增益與 3 dB 波束寬。",
    [["樣品", "26.5 GHz" + SR, "28 GHz" + SR, "29.5 GHz" + SR],
     ["S1", "9.0 dBi／28°", "9.6 dBi／25°", "11.8 dBi／30°"],
     ["S2", "9.3 dBi／26°", "9.2 dBi／22°", "11.6 dBi／26°"],
     ["S3", "9.8 dBi／28°", "9.1 dBi／24°", "11.2 dBi／24°"],
     ["S4", "9.9 dBi／30°", "9.0 dBi／25°", "11.3 dBi／24°"],
     ["S5", "9.8 dBi／28°", "9.2 dBi／23°", "11.4 dBi／25°"]],
    [2.17, 3.33, 3.33, 3.34], "陣列 28 GHz 峰值增益約 9.0–9.6 dBi，都是模擬值",
    "202603 - Final s22、25、28、31、34；數字位置依投影片版面讀出", row_h=0.6)

# ================================================================ 04 第四代
section_slide("04", "第四代　我們", "2026/06–2026/10　批次線上學習＋可製造")

table_slide(
    "第四代方法演進", "六月先在前代架構上做線上學習，七月初改成批次線上學習，之後四條線各自推進。",
    [["期間", "階段", "做了什麼", "關鍵數字"],
     ["6/08–6/25", "接手與提案", "程式精簡、前代資料收進 NAS；加 radiation pattern loss", "前代資料池 34,215 筆"],
     ["6/26–7/10", "線上學習五輪", "逐輪換一個瓶頸假設；建立 WM 對 HFSS 次數的量尺", "線上最佳 −2.89 dB" + ONE],
     ["7/03 起", "轉批次線上學習", "判準寫死、工具產生候選、SM 初篩、三台 HFSS 批次驗證", "一天真值 37 → 約 1,600 筆"],
     ["7/05–7/31", "單埠精修", "部分對稱化、承重圖、資料工廠、爬山、嫁接", "−2.68 → +0.79 dB" + NOT],
     ["8/05–8/10", "可製造化", "3,319 個設計各做菱形橋與挖空", "加橋 WM 中位 −0.07 dB" + ONE],
     ["8/10–8/31", "雙埠與交付", "換可製造幾何、無人值守迴圈；8/31 交付 39 個", "−5.90 → −2.39 dB" + NOT],
     ["9/01–10/23", "平台與十月兩線", "emforge 平台；對稱天線；寬頻濾波器七版規格", "+0.599、+0.4104 dB" + NOT]],
    [1.55, 2.1, 5.25, 3.27], "第四代的轉折在七月初：候選從哪裡來，比怎麼一步步優化更重要",
    "實驗總表 §3.4；notes/gen4 §2", row_h=0.56)

s = content_slide("批次線上學習流程", "研究者與 AI 提出假設，工具產生候選池，SM 篩選後交三台 HFSS 批次模擬，結果全部入庫。",
                  "改變的是工作方式：判準先寫死，HFSS 不停機", "延伸摘要圖 extended-abstract-workflow；實驗總表 4-014、4-038")
picture_fit(s, repo("docs/paper/figures/extended-abstract-workflow.png"), L, BODY_TOP, CW, 2.95, valign="top")
textbox(s, L, 4.9, CW, 1.45, [
    "・判準先寫死：每輪開始前寫下成功與失敗的標準，結果出來再判讀",
    "・一天真值從線上學習約 37 筆，提高到三台 HFSS 約 1,600 筆",
    "・每批收檔後重訓代理模型 SM；SM 只負責初篩，不直接決定誰最好"], size=16, spacing=1.15, after=4)

table_slide(
    "起點：前代資料池", "前三代累積的 34,215 筆模擬是第四代最大的起點資產。",
    [["實驗", "做法", "結果"],
     ["離線期望基準 7/03", "零 HFSS 估算每多一筆模擬的期望最佳", "從前代資料池起步，等效預算領先 200–450 倍" + ONE],
     ["均勻隨機對照 7/05", "每格 p=0.5、饋點強制，10 筆", "最好 −8.38 dB；前代資料池抽樣 −3.47 dB" + ONE],
     ["池頂重驗 7/05", "前代帳面達標 18 筆，用現行設定重測", "存活 8 筆，最佳 +0.44 dB" + ONE],
     ["部分對稱化 7/05", "前代散亂設計做 10-5-10 部分對稱", "−1.42" + ONE + " → −0.29 dB" + NOT + "，粉塵消失"]],
    [2.6, 4.2, 5.37], "之後所有天線冠軍都源自前代資料的錨點", "實驗總表 4-014、4-017、4-019、4-020", row_h=0.75)

fig_text_slide(
    "粉塵在承載共振", "把前代達標設計上 1–3 像素的小碎片拔掉，看響應怎麼變。",
    repo("docs/report/assets/f06_dust_loadbearing.png"),
    ["除塵 5 筆崩 4 筆，−4.7～−16.9 dB" + ONE, "整塊型設計只掉 0.02 dB，成為可製造起點 −2.68 dB" + ONE,
     "把懸浮的翼用 1 像素接回主件，也崩 5.4–5.9 dB" + ONE, "可製造改定義為每塊金屬 ≥4 像素，不強制連通"],
    "粉塵是共振的一部分：乾淨解要用搜的，不能用修的", "round-07、round-11；實驗總表 §2.4", img_w=7.4, head="結果")

fig_text_slide(
    "部分對稱化：零粉塵的起點", "前代散亂設計左右加上部分對稱，碎片消失、性能反而變好。",
    repo("docs/進度報告/07月/對稱化實例.png"),
    ["WM 由 −1.42" + ONE + " 變成 −0.29 dB" + NOT, "粉塵消失，成為可製造設計",
     "可製造紀錄一夜進步 2.39 dB", "後續所有天線冠軍都源自這個前代錨點"],
    "對稱化能救散亂的設計，但會毀掉已經好的設計", "round-09；實驗總表 4-019", img_w=7.6, head="結果")

table_slide(
    "單埠天線紀錄演進", "可製造 WM 紀錄的每一次刷新；S0 網格、無橋。",
    [["日期", "可製造 WM 紀錄", "做法"],
     ["7/03", "−2.68 dB" + ONE, "除塵後的整塊型前代設計，可製造起點"],
     ["7/06", "−0.29 dB" + NOT, "前代散亂設計做部分對稱"],
     ["7/07", "+0.20 dB" + NOT, "精修八個三標冠軍"],
     ["7/09", "+0.35 dB" + NOT, "水平切槽"],
     ["7/12–7/16", "+0.46 → +0.56 dB" + NOT, "預算分散到多種起點分布，資料工廠常駐"],
     ["7/26", "+0.73 dB" + NOT, "隨機生成加 SM 篩選的對照組產出"],
     [("7/31", True), ("+0.79 dB" + NOT, True), ("嫁接後再爬山，至今未破", True)]],
    [2.0, 3.6, 6.57], "可製造 WM 紀錄四週從 −2.68 推到 +0.79 dB", "docs/records.json；實驗總表 §2.7、4-057、4-061",
    row_h=0.56)

fig_text_slide(
    "單埠紀錄 +0.79 dB", "新紀錄對前一紀錄：綠色是加銅、紅色是去銅，右側是 S11、增益與兩切面 radiation pattern。",
    repo("docs/log/assets/round-48/newking_wm079.png"),
    ["把左側引擎件嫁接到主流骨架，再爬 5 包：−0.15 → +0.79 dB、rad +0.20" + NOT,
     "加 0.10 mm 菱形橋後 +0.58 dB" + NOT + "，跨機重現", "較細網格 S1 下為 −0.25 dB" + ONE + "，紀錄只在 S0 下成立"],
    "單埠 WM 紀錄 +0.79 dB，7/31 公證三次一致", "round-48；docs/records.json", img_w=7.8, head="結果")

fig_text_slide(
    "公證：重跑一致才承認", "HFSS 在固定設定下位元級可重現，所以差 0.01 dB 就是真差異。",
    repo("docs/report/assets/f16_phantom_case.png"),
    ["同一設計重跑 41 次全部同值" + NOT, "「首個三標全過」+0.48 重測 8 次全是 −0.06" + NOT,
     "原因是萃取層按索引對位，增益曲線平移 0.5 GHz，已改成按頻率對位", "之後所有紀錄一律公證"],
    "公證制度把單次的好運氣和真進步分開", "round-08～10；實驗總表 4-018、4-023", img_w=7.4, head="做法")

fig_text_slide(
    "可製造：對角接點加菱形橋", "0.21 mm 格的對角重疊只有 0.01 mm，做不出來；出圖在每個對角接點加 45° 菱形橋。",
    repo("docs/log/assets/round-54/diamond_geometry.png"),
    ["3,319 個設計各做加橋與挖空，共 6,638 筆模擬", "加橋 WM 變化中位 −0.07 dB，86.5% 變化不超過 0.3 dB" + ONE,
     "WM 王加橋後 +0.58 dB" + NOT, "天線橋寬 0.10 mm、濾波器 0.075 mm"],
    "出圖一律帶菱形橋，性能幾乎不受影響", "round-54；實驗總表 4-065", img_w=7.4, head="結果")

s = content_slide("可製造天線代表設計", "8/31 交付第一張：金屬圖形、S11、增益與 28 GHz radiation pattern；S0 網格、0.10 mm 橋。",
                  "可製造、帶橋的代表設計仍保有約 +0.7 dB 餘裕", "延伸摘要圖 extended-abstract-antenna；實驗總表 §2.11")
picture_fit(s, repo("docs/paper/figures/extended-abstract-antenna.png"), L, BODY_TOP, CW, 3.35, valign="top")
textbox(s, L, 5.3, CW, 1.05, [
    "・交付重解 +0.77 dB、原批次 +0.69 dB" + ONE + "；與 WM 王只差 1 像素",
    "・radiation pattern 兩切面餘裕 +0.63、+0.50 dB" + ONE + "；8/10 快照無橋合格解約 2,500 個不重複設計" + ONE], size=16, spacing=1.15)

fig_text_slide(
    "對稱天線 +0.599 dB", "生成時就讓金屬精確左右鏡射，再做左右成對翻格爬山；S0 網格、0.10 mm 橋。",
    repo("docs/log/assets/round-80/r80sym_best_card.png"),
    ["自由像素 624 → 324", "5,000 筆只有 1 筆 WM 為正：+0.022，重測 +0.036 dB" + ONE,
     "改成左右成對翻格爬山：+0.129 → +0.465" + ONE + " → +0.599 dB" + NOT, "S11 餘裕 +0.70、增益 +0.60；兩機重測一致"],
    "完全左右對稱、可製造的天線做到 +0.599 dB，回應一軸對稱要求", "round-80；實驗總表 4-098、4-103", img_w=4.7,
    head="做法與結果", size=17)

fig_text_slide(
    "雙埠濾波器：−5.90 到 −2.39 dB", "8/13 換成可製造幾何後重新開帳，記錄每一次刷新；規格 v2、0.075 mm 橋。",
    repo("docs/進度報告/09月/dual_progress_report.png"),
    ["開帳 −5.90 dB，就是前代資料原樣加橋", "8/24 −2.39 dB" + NOT + "，共 17 次刷新", "最後兩次由不含 AI 的無人值守腳本迴圈產生",
     "圖上 −2.30 是當時的估計，之後已被推翻", "仍未通過規格，主要卡 S21 通帶邊與阻帶起點"],
    "雙埠濾波器推進 3.51 dB，但仍未過規格", "九月報告；docs/records_dual.json；實驗總表 §2.7", img_w=7.2,
    head="結果")

fig_text_slide(
    "雙埠：舊模擬幾何的對角接觸", "承接的舊幾何每格剛好 0.2 mm、對角只碰一點；8/13 起改用和天線相同的可製造幾何。",
    repo("docs/進度報告/09月/recap_august_bridge.png"),
    ["純對角鏈 S21 −0.68 dB，被當成導通" + ONE, "補成真連接後同一設計 −3.05 → −23.83 dB" + ONE,
     "同設計切掉橋 −31.03、0.01 mm 微橋 −6.86" + ONE + "、0.075 mm 橋 −3.82 dB" + NOT, "前代雙埠資料同樣帶此問題，舊數字已凍結"],
    "當時的模擬 profile 有問題；換幾何後數字重新起算", "round-59、69、71；實驗總表 4-073、4-082", img_w=7.2,
    head="發現")

table_slide(
    "8/31 送板交付", "由鄭國宏學長安排，交付 39 個 .aedt 與 2 份目錄 PDF；39 筆全部重跑核對。",
    [["類別", "數量", "內容", "橋寬", "狀態"],
     ["天線款式代表", "21", "7 個款式的合格代表", "0.10 mm", "第一張重測 +0.77 dB" + ONE],
     ["天線前代頂尖設計", "4", "前代資料池候選", "0.05／0.075 mm", "橋寬低於 0.1 mm"],
     ["濾波器", "14", "主款式 12 個、第二款式 2 個，取當時最佳 1 dB 內", "0.075 mm", "全部未過規格"],
     [("合計", True), ("39", True), "輕量包約 5 MB；含場解約 7.5 GB 留 NAS", "—", "37 筆位元級一致、2 筆差 ≤0.08 dB" + NOT]],
    [2.3, 0.9, 4.05, 1.75, 3.17], "39 個設計已交付，尚未實測；18 個設計橋寬低於一般 PCB 最小線寬 0.1 mm",
    "九月報告 §2；實驗總表 §2.11", row_h=0.75)

table_slide(
    "寬頻濾波器：規格七版", "10/07 起學長依進度修訂規格；量測不變時舊結果直接用新規格重算，不重跑。",
    [["版本", "日期", "規格重點", "當時最佳"],
     ["v1", "10/07", "通帶 26–30 ≥ −3；阻帶 16–20、36–40 ≤ −20；S11/S22 ≤ −10；16–40 GHz", "八月好設計低頻回升到約 −4 dB" + ONE],
     ["v2", "10/12", "只看 S21；阻帶 22–24.5、36–40 ≤ −15", "−0.24" + ONE],
     ["v3", "10/13", "v2 兩阻帶收緊到 ≤ −20", "−1.06" + ONE],
     ["v4a／v4", "10/13", "S21 三段加 S11/S22 ≤ −10；阻帶 −15／−20", "v4a +0.21" + NOT + "；v4 差 4–5 dB"],
     ["v5a／v5", "10/14", "阻帶 18–24.5、36–40；18–40 GHz", "v5a −2.69" + ONE],
     ["v4x", "10/15", "v4 計分放在 18–40 GHz 量測上", "+0.03" + NOT],
     [("v6a／v6", True), "10/15", "阻帶 11–19、36–40 ≤ −15／−20，壓 Ku band 與 n260；11–40 GHz", ("v6a +0.4104" + NOT + "；v6 −3.05" + ONE, True)]],
    [1.25, 1.0, 6.15, 3.77], "目前規格 v6a 已達標；目標版 v6 的 −20 dB 低阻帶仍差約 3 dB",
    "round-81；實驗總表 §2.10", row_h=0.56)

fig_slide(
    "寬頻濾波器 v6a 性能爬升", "2,252 筆 11–40 GHz 模擬的累計最佳，全部以 −15 dB 版重算；右圖放大最後 1,437 筆，紅線是策略切換點。",
    repo("docs/log/assets/round-81/r81v6a_climb.png"), "最後一段的推進來自區塊移植與雙緣成對翻轉",
    "round-81、89～101；圖中累計最佳為單次值，公證最佳 +0.4104 dB")

fig_text_slide(
    "寬頻濾波器最佳設計", "交付首選設計的 S21、S11、S22 與金屬圖形；上下緣中央是兩埠饋墊，兩埠金屬不相連。",
    FIG / "gen4" / "g4_v6a_best_card_mg44016.png",
    ["10/23 公證 +0.4104 dB，216 與 218 兩機曲線一致" + NOT, "通帶 +0.41、低阻帶 +1.04、高阻帶 +2.95 dB",
     "S11 +1.24、S22 +4.71 dB", "次選 +0.399 dB" + NOT + "，216 與 37 兩機一致",
     "目標版 v6 全體最佳 −3.05 dB" + ONE + "，卡 11–19 GHz 低阻帶"],
    "寬頻濾波器 −15 dB 版已達標並公證，−20 dB 版仍未達標", "round-101；script/figs/r81_v6.py 於 10/23 重畫",
    img_w=5.4, head="結果", size=17,
    notes="與天線同級網格設定，未做網格收斂；10/23 15:05 依 Ricky 指示暫停，不再發新批。")

table_slide(
    "寬頻濾波器的關鍵做法", "十月兩週內試過的做法中，真正推動紀錄的是這六項。",
    [["做法", "日期", "結果"],
     ["兩埠金屬斷開", "10/15–10/16", "11 GHz 兩埠相連約 −3 dB 直通，斷開約 −18 dB" + ONE],
     ["用各項不足量總和挑親代", "10/14", "v4a −1.14 → −0.01 dB，下一批四筆 >0" + ONE + "；+0.21" + NOT],
     ["變化量代理模型", "10/19–10/20", "學「改了哪幾格、曲線變多少」；上線排序 ρ +0.24～+0.50，原 SM −0.27～−0.39" + ONE],
     ["區塊移植", "10/21", "好設計之間互換相異區塊；v6a 首度 ≥0：+0.063 dB" + NOT],
     ["雙緣成對翻轉", "10/22–10/23", "同時抬 26 與 30 GHz 兩緣：+0.312、+0.363 dB" + NOT],
     ["兩款式互移植", "10/23", "+0.391 → +0.4104 dB" + NOT + "，在 +0.39～+0.41 呈平台"]],
    [3.0, 1.75, 7.42], "有效的是在其他好設計出現過的小步", "round-81～101；實驗總表 §2.7、4-104～4-120", row_h=0.62)

table_slide(
    "代理模型：從主角變成粗篩", "第二代起反覆出現的瓶頸是代理模型不夠準；第四代逐步調整它的角色。",
    [["時期", "做法", "結果"],
     ["6–7 月", "線上更新 SM 五輪", "擬合誤差 8–11 → 0.45，最佳仍停在 −3.65 dB" + ONE],
     ["7 月", "SM 當篩子", "WM 排序 ρ 0.493；帶外排序 0.334 → 0.603" + ONE],
     ["7 月", "CNN 排序器、學習率排程修正", "凍結考卷誤差 −56%、−51%" + ONE],
     ["8 月", "濾波器 SM 冷啟動", "整體 ρ +0.4675，但同一盆地內 −0.098" + ONE],
     ["10 月", "全曲線 SM 排寬頻好設計", "好設計之間 ρ 只有 0.15–0.18" + ONE],
     ["10 月", "變化量模型", "上線排序 ρ +0.24～+0.50" + ONE]],
    [1.5, 4.0, 6.67], "SM 只負責粗篩；改學變化量後，好設計之間的排序才轉為正相關",
    "實驗總表 §2.6", row_h=0.6)

fig_text_slide(
    "emforge 平台", "9 月把派工、去重、斷點續跑、公證抽成與天線領域無關的平台。",
    repo("docs/進度報告/09月/emforge_framework.png"),
    ["697 個測試通過", "9/12 三台初驗 10 筆真 HFSS 全部成功，跨機最大差 1.9×10⁻⁶ dB" + NOT,
     "委託方兩條硬約束：沒有 AI 也要能跑一整晚、不綁定特定策略", "受控試用，不是生產系統；十月實驗仍走舊批次線"],
    "emforge 是受控試用的平台，十月實驗尚未使用", "九月報告 §8；實驗總表 4-094、4-095", img_w=7.2, head="現況")

# ================================================================ 05 日月光點名主題
section_slide("05", "日月光點名主題", "對稱、七巧板、饋入強制放金屬，以及孤島與對角導通")

table_slide(
    "對稱：四代做了什麼", "日月光 2025/10 要求「至少一軸對稱」；第四代先證實事後對稱化有害，再做生成時鏡射。",
    [["世代", "做了什麼", "結果", "結論"],
     ["第一代", "沒做", "—", "—"],
     ["第二代", "2024/09 對稱生成，想改善場型凹陷", "圖沒存；帶內 S11 多在 −4～−10 dB" + EYE, "沒有延續"],
     ["第三代", "Dual Mirror 翻轉擴增、對稱式七巧板", "「比較好的是 mirrored2」，沒有數字", "「效果不佳」"],
     ["第三代", "Mirror 8：8 種翻轉先給代理模型預測；DMCVAE", "預測 loss 最佳 6.43" + SR + "，不是 HFSS 值", "回應日月光，沒進碩論"],
     ["第四代", "7–8 月：部分對稱化、事後對稱化探針", "−1.42 → −0.29" + NOT + "；好設計中位 −17.16 dB" + ONE, "救爛解、毀好解"],
     ["第四代", "10 月：生成時精確左右鏡射，再成對翻格爬山", "+0.022" + ONE + " → +0.599 dB" + NOT, "對稱要在生成時內建"]],
    [1.2, 4.2, 4.0, 2.77], "對稱要在生成時就內建；事後把好設計對稱化會毀掉它", "實驗總表 §2.1", row_h=0.6)

table_slide(
    "對稱：第四代的六個實驗", "對稱度是 radiation pattern 的旋鈕，但事後套用會破壞帶內共振。",
    [["實驗", "做法", "結果"],
     ["全對稱化探針 7/07", "八個冠軍全部左右對稱化", "radiation pattern 8/8 變好、平均 +0.19 dB；WM 7/8 變差" + ONE],
     ["事後對稱化 8/06", "13 個好設計做三種鏡射，34 筆", "波束置中 100%；WM 變化中位 −17.16 dB" + ONE],
     ["濾波器上下對稱 8/10", "生成時對稱對照非對稱", "品質相當，中位 −17.87 對 −18.50 dB" + ONE + "；升為預設"],
     ["歷史普查 10/07", "零 HFSS 掃 39,111 張已模擬設計", "1,030 張完全左右對稱" + ONE],
     ["生成時精確鏡射 10/07–10/11", "自由像素 624 → 324，0.10 mm 橋，5,000 筆", "1 筆 WM 為正：+0.022，重測 +0.036 dB" + ONE],
     ["成對翻格爬山 10/15", "左右兩格一起翻，保持對稱", "+0.129 → +0.465" + ONE + " → +0.599 dB" + NOT]],
    [3.2, 4.0, 4.97], "對稱上限不是 +0.022：成對翻格把對稱天線推到 +0.599 dB",
    "round-11、55、57、80；analysis-18；實驗總表 §2.1", row_h=0.6)

table_slide(
    "七巧板：四代做了什麼", "七巧板＝從固定形狀庫挑小形狀拼滿圖形；只有第三代正式做過。",
    [["世代", "做了什麼", "結果", "結論"],
     ["第一代", "沒做", "—", "—"],
     ["第二代", "2024/09 Selection：每格從幾種子圖案選一個", "隨機取樣品質差，先關掉", "2025/05 重啟"],
     ["第二代", "2025/05 Selection：每格選 8 種 2×2 子圖案之一", "RIS「一鳴驚人」，接代理模型訓不起來", "七巧板的前身"],
     ["第三代", "2025/07 v1：8 個 5×5 粗形狀填滿 25×25", "雙埠最低 −15、−19、−15 dB" + SR + "，「效果不太好」", "推測形狀太少"],
     ["第三代", "2025/10–11 v2：14、16 個細形狀，擴成 35 種；對稱式七巧板", "只有圖", "—"],
     ["第三代", "2025/11 Gumbel-Softmax 對 Gumbel-Sinkhorn", "只有圖", "之後沒有再出現"],
     ["第四代", "沒做七巧板", "—", "改在像素上搬整塊金屬"]],
    [1.2, 4.9, 3.85, 2.22], "七巧板三版都只有圖、沒有數值結論，沒進碩論", "實驗總表 §2.2", row_h=0.56)

table_slide(
    "第四代最接近的做法：搬動整塊金屬", "都在既有 25×25 像素設計上搬動或複製整塊金屬，不是從形狀庫挑形狀拼滿，所以不是七巧板。",
    [["做法", "日期", "結果", "結論"],
     ["金屬組級變異：長邊、縮邊、平移、刪除、複製", "7/25", "左側紀錄 −3.46 dBi、WM +0.15" + NOT, "組級優於像素，未達顯著"],
     ["組文法：資料驅動的金屬組拼法", "7/26", "有對角塊鏈時左側增益中位 −3.34，沒有時 +2.27" + ONE, "對角接觸是左側抑制的機制"],
     ["嫁接：把一塊金屬移到另一個設計再爬山", "7/30", "−0.15 → +0.79 dB" + NOT, "單埠紀錄至今未破"],
     ["區塊移植：好設計之間互換相異區塊", "10/21–10/23", "v6a 首度 ≥0，最後 +0.4104 dB" + NOT, "別的好設計出現過的小步最有效"]],
    [3.9, 1.5, 3.9, 2.87], "第四代的「拼裝」是在好設計之間搬區塊，而不是從形狀庫拼",
    "round-41、43、44、48、89～101；實驗總表 §2.2", row_h=0.78)

table_slide(
    "饋入強制放金屬：四代做了什麼", "四代都把饋入點附近固定為金屬，差在大小與目的。",
    [["世代", "做了什麼", "結果", "結論"],
     ["第一代", "論文實驗一：饋入點周圍 8×8 格強制金屬", "有固定「更加平穩快速」收斂，看曲線", "之後全部固定"],
     ["第二代", "底部中央固定遮罩 8 列×7 行；雙埠改上下各一塊", "程式設定，一直沿用", "雙埠沿用到第四代"],
     ["第三代", "比較有無固定饋入源", "「無固定的結果比較好」，沒有數字", "條件：只在變異後才套上"],
     ["第三代", "2026/05 固定區 5×5 擴為 9×9", "R_feed 約 52.42%" + SR, "碩論附錄寫 9×9"],
     ["第四代", "單埠饋點永遠是金屬，必要時補 3×3 貼底墊；雙埠兩塊 5×5 埠墊", "自由像素 624 個", "全程固定"],
     ["第四代", "寬頻濾波器：只留埠墊、兩埠金屬斷開", "11 GHz 相連約 −3 dB 直通，斷開約 −18 dB" + ONE, "低阻帶需要兩埠斷開"]],
    [1.2, 4.6, 3.75, 2.62], "饋入固定四代都保留；到寬頻濾波器，兩埠金屬反而要斷開",
    "實驗總表 §2.3；各代固定區尺寸照各自出處，不統一換算", row_h=0.6)

table_slide(
    "孤島與對角導通：四代解法", "同一個製造問題：對角相接的像素做不做得出來、懸浮金屬能不能刪。",
    [["世代", "發現", "解法", "結果"],
     ["第一代", "口委指出對角相接的像素在 HFSS 算連通，實際洗板可能斷", "構想：內部格子四邊各擴 0.05 mm 讓對角重疊", "只是構想，未實作"],
     ["第二代", "刪除孤島「會劇烈引響天線特性」", "Over Unite：格寬 0.2 → 0.21 mm；決定不刪", "「命中率大幅提升」；0.21 mm 沿用至今"],
     ["第三代", "拿掉孤島響應變差；碎圖樣讓 mesh 出錯", "訓練中加 TV → IS → GC → SC 連通 loss", "R_feed 61.98%" + SR],
     ["第四代", "粉塵與懸浮金屬在承載共振；0.01 mm 對角重疊做不出來", "出圖一律加菱形橋：天線 0.10 mm、濾波器 0.075 mm", "WM 變化中位 −0.07 dB" + ONE]],
    [1.2, 4.0, 4.0, 2.97], "四代一致：直接刪孤島或補洞會傷響應，解法一路移到生成端",
    "實驗總表 §2.4", row_h=0.85)

# ================================================================ 06 其他主題與現況
section_slide("06", "其他主題與現況", "其他主題索引、現況、限制與下一步")

table_slide(
    "其他主題索引", "實驗總表 §2 另有七個主題，各一句結論；細節查總表對應小節。",
    [["主題", "一句結論", "總表"],
     ["損失函數與評分", "從逼一條目標曲線，走向只罰沒達標的部分；第四代改用統一的 dB 尺 WM", "§2.5"],
     ["生成器與代理模型", "骨架三代沿用；代理模型不夠準是第二代起的共同瓶頸", "§2.6"],
     ["搜尋與訓練策略", "變異加回滾 → ACP 單階段 → 從前代好設計出發的爬山、嫁接、區塊移植", "§2.7"],
     ["可製造與結構", "0.2 mm 網格、0.21 mm 金屬；第四代拆成無粉塵、出圖帶橋、留緩衝三件事", "§2.8"],
     ["模擬設定與儀器", "HFSS 固定設定下位元級可重現，據此建立公證；也查出萃取對位錯誤", "§2.9"],
     ["規格演進", "n257 天線規格從第一代沿用；之後加濾波、雙埠、radiation pattern、寬頻阻帶", "§2.10"],
     ["量測與交付", "只有第三代報告過實作量測；第四代 39 個設計尚未實測", "§2.11"]],
    [2.6, 8.4, 1.17], "跳出局部解與代理模型準度，是貫穿四代的兩個共同主題", "實驗總表 §2", row_h=0.58)

table_slide(
    "現況與下一步", "截至 2026/10/23；第四代的性能推進目前都已暫停，等實作驗證。",
    [["項目", "現況", "下一步"],
     ["單埠天線", "+0.79 dB" + NOT + "；交付代表 +0.77 dB" + ONE, "等實作量測對照模擬"],
     ["對稱天線", "+0.599 dB" + NOT, "慢速探索"],
     ["雙埠濾波器 v2", "−2.39 dB" + NOT + "，未過規格", "已暫停"],
     ["寬頻濾波器 v6a", "+0.4104 dB" + NOT + "，達標", "首選、次選兩機一致；另四筆 +0.363～+0.395 三機一致"],
     ["寬頻濾波器 v6", "−3.05 dB" + ONE + "，卡 11–19 GHz 低阻帶", "10/23 暫停"],
     ["8/31 交付 39 個設計", "尚未實測；18 個橋寬低於 0.1 mm", "請板廠確認製程可行性"],
     ["emforge 平台", "受控試用，不是生產系統", "遠端版本待更新"]],
    [3.0, 5.0, 4.17], "下一步的關鍵是實作量測：模擬數字要等板子回來才算驗證",
    "實驗總表 §1、§2.11；ONGOING", row_h=0.56)

bullets_slide(
    "目前的限制", "這些限制會影響數字怎麼解讀，對外引用時一併說明。",
    [("模擬與製造", ["所有第四代數字都是 HFSS 模擬，39 個交付設計尚未實測",
                 "天線紀錄只在 S0 網格成立；較細網格 S1 下 WM 王 +0.79 變 −0.25" + ONE,
                 "寬頻濾波器用同級網格，也沒做網格收斂",
                 "0.075 mm 橋低於一般 PCB 最小線寬 0.1 mm，需板廠確認"]),
     ("方法", ["可製造與深度低頻抑制還不能同時做到：兩個左側紀錄設計加橋後都不合格",
             "三次經驗上界都被後續資料推翻，只能講目前前緣",
             "代理模型只能粗篩，好設計之間排序常失效",
             "七巧板、50×50 網格、連續參數化結構，第四代都沒做"])],
    "限制都寫在明處：哪些數字可以引用、要附什麼條件", "實驗總表 §4.3；notes/gen4 §7", size=16)

# ================================================================ 附錄
section_slide("附錄", "實驗總表精簡版與名詞", "完整 223 項見 實驗總表.md §3")

appendix_table("附錄：第一代實驗精簡表 1/2", "2023/08–2024/03；完整 27 項見總表 §3.1。", [
    ["1-01", "2023/08–10", "numpy 自建網路系列", "只列設定，沒有結果數字；轉 PyTorch，STE 保留"],
    ["1-03", "2023/10", "強化學習 Policy Gradient", "25×25 一個 epoch 跑 200 次模擬，「超慢」；放棄"],
    ["1-04", "2023/10/17", "DNI 合成梯度", "沒有結果紀錄；演化成 GEN"],
    ["1-05", "2023/10", "DDPG、Dueling DQN、PPO", "4.7 GHz 跑 2096 次，loss 平在約 8.8" + EYE],
    ["1-06", "2023/11/12", "LHS 取樣建資料", "未進論文，資料去向不明"],
    ["1-07", "2023/11", "GEN 線上學習成形", "每迭代 1 次 HFSS；論文定案流程"],
    ["1-08", "2023/11", "4.7 GHz 饋入點保證金屬", "饋入固定金屬首次出現"],
    ["1-09", "2023/11", "28 GHz 25×25 對 50×50", "minloss 1.48 對 40.36" + SR + "；定 25×25"],
    ["1-10", "2023/11–12", "25×25 多輪重跑", "minloss 1.48 到 18.58" + SR + "；不同 seed 差很多"],
    ["1-11", "2023/11–12", "上下界區間 loss", "「稍微好一點」，沒有數字；進論文"],
    ["1-12", "2023/12–2024/02", "四層板 M1 網格試探", "論文採 M1 5×5 mm 25×25、M2 48×22"],
    ["1-13", "2024/03", "GEN 加 BN", "全層 BN 梯度消失，改成只放後兩層"],
    ["1-14", "2024/03", "MAE loss", "「似乎對 Loss 的降低沒有顯著影響」，未採用"],
    ["1-16", "2024/04/02", "GEN 準度評估", "MAE 2.28、MSE 11.80" + SR + "，條件不明"]], "實驗總表 §3.1", footer="前半年是方法探索：從自建網路、強化學習走到 MPGN＋GEN 線上學習")

appendix_table("附錄：第一代實驗精簡表 2/2", "2024/04–2024/07 論文實驗與口試補充。", [
    ["1-17", "2024/04", "加 3dB 波束寬條件", "只有 loss 曲線與場型圖，未進論文"],
    ["1-18", "2024 春", "實驗一 饋入固定金屬", "有固定「更加平穩快速」收斂，看曲線"],
    ["1-19", "2024 春", "實驗二 激活函數", "PReLU 快速收斂到接近 0" + SR],
    ["1-20", "2024 春", "實驗三 四層板 n257", "loss 接近 0" + SR + "；26.5 GHz S11 約 −9.3 dB" + EYE],
    ["1-21", "2024 春", "實驗四 四層板濾波天線", "loss 趨近 0" + SR + "；帶外兩點未達下界" + EYE],
    ["1-22", "2024 春", "實驗五 雙層板濾波天線", "約 800 次後接近 0" + SR],
    ["1-23", "2024 春", "雙層板 n257 最終結果", "S11 全低於 −10 dB，最低約 −17 dB" + EYE],
    ["1-24", "2024/07", "重複實驗", "seed 不固定，每次圖形都不同"],
    ["1-25", "2024/07", "場型", "只放圖；優化時沒管場型"],
    ["1-26", "2024/07", "模擬次數比較", "本法 1,500 次；GA＋CNN 110 萬次" + SR],
    ["1-27", "2024/07", "連通性與孤島分析", "對角相接實際可能斷；擴邊 0.05 mm 構想，未實作"]], "實驗總表 §3.1", footer="論文五個實驗都收斂到接近 0；場型與連通性在口試後才補上")

appendix_table("附錄：第二代實驗精簡表 1/2", "2024/08–2024/12；完整 29 項見總表 §3.2。", [
    ["2-01", "全程", "n257 濾波 loss", "指示函數加 Smooth L1，帶內帶外分開算；論文主規格"],
    ["2-02", "2024/08", "CNN 生成器", "反卷積棋盤效應；維持 MLP"],
    ["2-03", "2024/08", "預訓練初始圖形", "能加快收斂，但後面仍梯度消失"],
    ["2-04", "2024/09", "fix_metal_area 饋入固定", "底部中央 8 列×7 行；一直沿用"],
    ["2-05", "2024/09", "對稱生成", "圖沒存；帶內 S11 多在 −4～−10 dB" + EYE],
    ["2-06", "2024/09", "Update Image", "參數太敏感，失敗"],
    ["2-07", "2024/09", "Selection Model", "隨機取樣品質差，先關掉"],
    ["2-08", "2024/10", "優化器比較", "RMSProp 優於 Adam，SGD 直接炸"],
    ["2-09", "2024/10", "MSE 換 Smooth L1", "進論文"],
    ["2-10", "2024/10", "迭代次數觀察", "1,000–1,500 次完成生成" + SR],
    ["2-11", "2024/11–12", "Sign 函數替代品", "Approx 最穩，但模擬前仍要硬二值化，判定雞肋"],
    ["2-12", "2024/11", "Random Sampling", "loss 最低但迭代很多，未進論文"],
    ["2-13", "2024/11", "GAN 變體", "「很炫但沒用」"],
    ["2-14", "2024/12", "非對稱正規化", "RIS case 很穩，即後來的 BiScaleNorm"]], "實驗總表 §3.2", footer="很多嘗試只有定性結論；真正留下的是 Smooth L1 與非對稱正規化")

appendix_table("附錄：第二代實驗精簡表 2/2", "2024/12–2025/06 論文實驗與日月光相關工作。", [
    ["2-15", "論文", "實驗一 BiScaleNorm", "3e-5 約第 500 次收斂，原設計約第 1,000 次" + SR],
    ["2-16", "2024/12", "SWA、餘弦退火", "有跳但回到相同位置，捨棄"],
    ["2-17", "2025/01", "變異機制", "變異率 0.05、週期 400；增益達標 20 幾個、S11 4 個，互不交集" + SR],
    ["2-18", "論文", "實驗三 回滾消融", "無回滾明顯發散，有回滾穩定收斂"],
    ["2-19", "2025/02–04", "獨立金屬群", "刪除會劇烈影響天線特性，決定不刪"],
    ["2-20", "2025/02–06", "代理模型問題", "1,000 筆後重訓要 60,000 次迭代" + SR + "；梯度誤差要當指標"],
    ["2-21", "2025/03", "BiScaleNorm 加 EMA", "「蠻有用」，命名定案"],
    ["2-22", "2025/04", "變異觸發條件", "「永遠觸發不了」，維持固定週期"],
    ["2-23", "2025/04", "Over Unite", "格寬 0.2 → 0.21 mm 保證對角導通"],
    ["2-25", "論文", "實驗二 二階段生成", "第 1,200 次後跳出原解空間" + SR],
    ["2-26", "2025/05–06", "雙埠濾波", "只有曲線，沒寫是否達標；進日月光 6 月簡報"],
    ["2-27", "2025/05", "Selection Mode 重啟", "RIS 一鳴驚人，接代理模型訓不起來"],
    ["2-28", "2025/06", "n260 遷移", "「趨勢和 n257 一致」，沒有數字"],
    ["2-29", "2025/06", "Gradient Doping", "只有構想頁，未驗證"]], "實驗總表 §3.2", footer="BiScaleNorm、變異與回滾進論文；雙埠與 n260 只有初步曲線")

appendix_table("附錄：第三代實驗精簡表 1/2", "2025/07–2025/12；完整 47 項見總表 §3.3。", [
    ["3-01", "2025/07", "雙埠兩階段比較", "「學長做的二階段梯度探索，效果很好」"],
    ["3-02", "2025/07", "雙埠舊法重跑 3 次", "S11、S22 沒有完全壓進規格" + EYE],
    ["3-03", "2025/07", "變異自動觸發", "「只需要一次收斂」；2026/01 改用 τ 取代"],
    ["3-04", "2025/07", "Dual Mirror", "「比較好的是 mirrored2」；之後判定效果不佳"],
    ["3-05", "2025/07", "七巧板 v1", "最低 −15、−19、−15 dB" + SR + "，「效果不太好」"],
    ["3-08", "2025/10", "七巧板 v2", "14、16 個形狀，11 月擴成 35 種；只有圖"],
    ["3-10", "2025/10", "MinMax Loss", "「中間目標區域確實更貼近目標曲線」"],
    ["3-11", "2025/10", "Mirror 8 翻轉", "代理模型預測最佳 6.43" + SR + "；沒進碩論"],
    ["3-13", "2025/10", "預訓練代理模型", "validation 停在約 0.41" + EYE + "，過擬合"],
    ["3-14", "2025/10", "ACS／ACP 提出", "loss 停滯就重啟學習率與 τ；碩論核心"],
    ["3-15", "2025/10", "菁英複習", "演進成 DLF"],
    ["3-18", "2025/11", "Gumbel-Softmax 對 Sinkhorn", "只有圖；七巧板之後沒有再出現"],
    ["3-19", "2025/11", "熱啟動 Base-1、Base-2", "Base-2 一開始就在 28 GHz 深度諧振"],
    ["3-22", "2025/12", "Base 加 TV 1%", "性能保持穩定，圖案更連續"]], "實驗總表 §3.3", footer="前半年試過七巧板、鏡射與變異，最後收斂到 ACP 與熱啟動")

appendix_table("附錄：第三代實驗精簡表 2/2", "2025/12–2026/06 孤島抑制、量測與碩論定案。", [
    ["3-23", "2025/12", "DMCVAE", "TV 1% 時能生成對稱結構；沒有延續"],
    ["3-24", "2025/12", "ACS Peak 對 Linear", "Linear 較好，碩論採用"],
    ["3-25", "2026/01", "TV、IS 權重掃描", "R_feed TV100 59.62%、IS100 61.00%" + SR],
    ["3-27", "2026/02", "GC 閉運算 loss", "R_feed 53.95%" + SR + "；被 SC 取代"],
    ["3-28", "2026/02", "SC 圖譜連通度 loss", "2026/05 定案，碩論採用"],
    ["3-31", "2026/03", "R_feed 指標", "饋入連通金屬占總金屬的比率；不可微，只用來評估"],
    ["3-32", "2026/03", "雙埠 TV100", "R_feed 94.03%" + SR + "；讓兩個饋入點導通"],
    ["3-34", "2026/03", "One-Stop 單階段", "設計時間約為舊法 50%" + SR],
    ["3-35", "2026/03/24", "5 個實作樣品量測", "量測通帶比模擬往高頻偏 0.5–1.6 GHz" + MEAS + "；設計者待確認"],
    ["3-37", "2026/04", "少量 pixel 補洞", "單埠「有補得反而不好」；要在訓練中約束"],
    ["3-41", "2026/05", "固定饋入 9×9", "R_feed 約 52.42%" + SR],
    ["3-44", "2026/05/28", "顯微鏡量 pixel", "0.188–0.231 mm" + MEAS + "，模擬 0.21 mm"],
    ["3-45", "2026/06", "排程比較", "ACP loss 2.07、R_feed 18.89%" + SR],
    ["3-47", "2026/06", "三機制消融", "ACP＋SC＋DLF loss 0.99、R_feed 61.98%" + SR]], "實驗總表 §3.3", footer="後半年聚焦孤島抑制，並報告了四代唯一的實作量測")

appendix_table("附錄：第四代實驗精簡表 1/2", "2026/06–7/24；完整 120 項見總表 §3.4。天線數字皆為 S0 網格、無橋。", [
    ["4-002", "6/08–6/11", "接手與核心精簡", "前代資料池 34,215 筆；6/10 正式機跑通"],
    ["4-008", "6/15–6/25", "radiation pattern loss", "擬合 MSE 1.85 → 0.37" + ONE + "；之後升格為獨立關卡"],
    ["4-013", "6/27–7/03", "線上學習 R1–R4", "線上最佳 −2.89 dB" + ONE],
    ["4-014", "7/03", "離線期望基準", "從前代資料池起步，等效預算領先 200–450 倍" + ONE],
    ["4-015", "7/03", "除塵", "5 筆崩 4 筆，−4.7～−16.9 dB" + ONE],
    ["4-018", "7/05–7/06", "噪聲地板", "同設計 41 次全同值" + NOT],
    ["4-019", "7/05–7/06", "部分對稱化", "−1.42" + ONE + " → −0.29 dB" + NOT],
    ["4-022", "7/06–7/07", "精修八冠軍", "最佳 +0.20 dB" + NOT],
    ["4-023", "7/06–7/07", "+0.48 假象", "重測 8 次全 −0.06" + NOT + "；改按頻率對位"],
    ["4-024", "7/06–7/08", "承重圖", "遮掉左右下段崩 11.1／12.2 dB" + ONE],
    ["4-033", "7/08", "翼消融", "只留主件 −5.79 dB；首翼 +4.4 dB" + ONE],
    ["4-036", "7/09", "水平切槽", "rad +1.0～+1.7 dB" + ONE + "；紀錄 +0.35 dB" + NOT],
    ["4-045", "7/13–7/24", "CNN 架構對決", "凍結考卷誤差 −56%" + ONE],
    ["4-053", "7/23–7/24", "左側戰役", "左側合格首例 WM +0.150" + NOT + "；加橋 −0.10" + ONE]], "實驗總表 §3.4", footer="七月初轉向批次線上學習，一個月內把可製造紀錄推過 0 dB")

appendix_table("附錄：第四代實驗精簡表 2/2", "2026/7/30–10/23；濾波器 8/13 前後幾何不同，數字不可同線比。", [
    ["4-061", "7/30–7/31", "嫁接加爬山", "−0.15 → +0.79 dB" + NOT + "；單埠紀錄"],
    ["4-064", "8/03–8/04", "網格 S0 對 S1", "WM 王 +0.79 → −0.25 dB" + ONE + "；S0 未收斂"],
    ["4-065", "8/05–8/10", "菱形橋全交叉", "加橋 WM 中位 −0.07" + ONE + "；王加橋 +0.58" + NOT],
    ["4-066", "8/06", "事後對稱化", "WM 變化中位 −17.16 dB" + ONE],
    ["4-073", "8/11–8/13", "濾波器零厚度對角接觸", "補成真連接後 −3.05 → −23.83 dB" + ONE],
    ["4-088", "8/16–8/24", "無人值守腳本迴圈", "−2.43 → −2.39 dB" + NOT],
    ["4-092", "8/25–8/31", "送板交付 39 個", "37 筆位元級一致、2 筆差 ≤0.08 dB" + NOT],
    ["4-094", "9/01–9/23", "emforge 平台", "697 個測試；三台 10 筆真 HFSS 全成功"],
    ["4-098", "10/07–10/11", "生成時精確鏡射", "+0.022，重測 +0.036 dB" + ONE],
    ["4-103", "10/15", "對稱成對翻格爬山", "+0.599 dB" + NOT],
    ["4-104", "10/15–10/16", "寬頻兩埠斷開", "11 GHz 相連約 −3、斷開約 −18 dB" + ONE],
    ["4-107", "10/19–10/20", "變化量代理模型", "上線排序 ρ +0.24～+0.50" + ONE],
    ["4-114", "10/21", "區塊移植", "v6a 首度 ≥0：+0.063 dB" + NOT],
    ["4-120", "10/23", "兩款式互移植", "+0.4104 dB" + NOT + "；另四筆三機一致"]], "實驗總表 §3.4", footer="八月起可製造化與雙埠，十月對稱天線與寬頻濾波器達標")

TERMS = [
    ("像素化圖形、pattern", "把金屬區切成網格，每格填金屬或不填；主力是 25×25、每格 0.2 mm、共 5×5 mm"),
    ("HFSS", "業界常用的電磁模擬軟體，四代都拿它當真值；一筆要數分鐘，不能算梯度"),
    ("生成器 G、MPGN", "吐出像素圖形的神經網路；第一代叫 MPGN，金屬圖形生成網路"),
    ("代理模型 SM、GEN", "用神經網路近似 HFSS，輸入圖形、輸出預測響應，把梯度借回生成器"),
    ("線上學習", "生成器與代理模型邊產生、邊模擬、邊學，不先收大資料集"),
    ("STE", "直通估計，讓 0/1 階梯函數也能傳梯度的近似技巧"),
    ("BiScaleNorm", "第二代的雙尺度正規化：正負兩側分別縮放，緩解二值化造成的梯度飽和"),
    ("變異、回滾", "週期性隨機翻轉部分像素以跳出局部解；變異前若不如歷史最佳就退回最佳圖"),
    ("ACP", "第三代的自適應循環排程：loss 停滯就重啟學習率與二值化溫度 τ"),
    ("熱啟動", "用理論公式算出的 28 GHz 初始貼片當起點，不從空白開始"),
    ("DLF", "第三代的動態菁英篩選：用累計平均當門檻，挑好樣本給代理模型複習"),
    ("TV、IS、GC、SC loss", "第三代依序試過的孤島抑制 loss：總變差、孤立像素、閉運算、圖譜連通度"),
    ("R_feed", "饋入可達率：與饋入點連成一塊的金屬占總金屬的比率，越高代表孤島越少"),
    ("孤島、粉塵", "孤島是沒接到饋入點的懸浮金屬；粉塵是第四代用語，指 1–3 像素的小碎片"),
    ("WM", "帶內最差餘裕：每條規格在每個評分頻點離門檻多少 dB，取最差那一點；正值代表全部通過"),
    ("radiation pattern", "方向圖；第四代要求 28 GHz 兩切面在 ±45° 內不低於峰值 −3 dB，簡稱 rad"),
    ("三標、合格", "WM ≥0、rad ≥0、可製造三者同時成立；7/12 起合格另要 WM ≥ +0.15 當製造緩衝"),
    ("可製造", "每塊金屬 ≥4 像素、沒有粉塵；8 月起出圖一律加菱形橋"),
    ("S0／S1 網格", "第四代 HFSS 網格設定；S0 是全程使用的設定，S1 較嚴；天線紀錄只在 S0 下成立"),
    ("帶外、左側", "帶外是規格頻段以外；左側指 24–25.5 GHz 的低頻帶外，是規格之外另加的要求"),
    ("款式", "第四代把結構相近的設計歸成一個款式"),
    ("前代資料池", "第四代承接的前代歷史模擬 34,215 筆，單埠 24,189、雙埠約 10,023"),
    ("公證", "同設定重跑兩次以上、結果一致才承認；HFSS 在固定設定下位元級可重現"),
    ("批次線上學習", "判準先寫死、工具產生一批候選、SM 初篩、三台 HFSS 批次模擬、判讀後重訓 SM"),
    ("爬山、嫁接、區塊移植", "每次只改少數像素留下變好的；把一塊金屬移到另一個設計；好設計之間互換相異區塊"),
]
for k, chunk in enumerate((TERMS[:13], TERMS[13:])):
    table_slide("附錄：名詞白話表 %d/2" % (k + 1), "第一次讀到不懂的詞先查這裡。",
                [["名詞", "白話"]] + [[(a, True), b] for a, b in chunk], [2.9, 9.27],
                footer="術語第一次出現就翻成白話；全場盡量只用少數幾個", source="實驗總表 §0", header_size=13, body_size=12, row_h=0.32, header_h=0.4)

table_slide(
    "附錄：資料位置", "本簡報的數字與圖都可以追回原始檔。",
    [["資料", "位置"],
     ["實驗總表 223 項、主題索引、待確認清單", "docs/進度報告/10月_歷來實驗總整理/實驗總表.md"],
     ["四代筆記與可用圖說明", "同夾 notes/gen1～gen4；圖在同夾 figs/"],
     ["第四代研究日誌", "docs/log/round-NN-*.md，圖在 docs/log/assets/"],
     ["紀錄真相源", "docs/records.json、docs/records_dual.json"],
     ["量測原始資料", "NAS 工作區；本機有增量備份"],
     ["8/31 交付包", "輕量包約 5 MB：39 個 .aedt 加 2 份目錄 PDF；含場解約 7.5 GB 留 NAS"],
     ["本簡報與時間軸圖", "同夾 make_pptx.py；script/figs/report_generations.py"]],
    [4.4, 7.77], "所有數字都以實驗總表為準", None, header_size=14, body_size=13, row_h=0.52)

prs.save(OUT)
print(f"saved {OUT} ({len(prs.slides)} slides)")
