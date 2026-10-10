"""Build the Traditional Chinese extended abstract (one page) and its attachments.

Usage: python docs/paper/build_extended_abstract.py [source.json ...]
(default: the abstract; e.g. also symmetry-difficulty-zh-2026-10.json).

Dependencies: reportlab (PDF), pymupdf (render/QA), pypdf (text QA), matplotlib (figure).
Run from any working directory. Local temporary dependencies may live in
tmp/pdfs/deps; final PDF is written to output/pdf/. The figure is rebuilt from
the frozen evidence JSONs next to this file; nothing here runs HFSS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / "tmp/pdfs/deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from matplotlib.colors import ListedColormap
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from scipy import ndimage

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "extended-abstract-zh-2026-10.json"   # default source
FREE = BASE / "single-port-evidence.json"
SYM = BASE / "symmetric-best-evidence.json"
FIG = BASE / "figures/extended-abstract-symmetry.png"
FIG_SIZE_IN = (7.0, 1.45)


def rich(text: str) -> str:
    """Escape for reportlab, keeping only the <b>…</b> run-in labels used in the JSON."""
    return escape(text).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")


def components(bits: np.ndarray) -> int:
    """Metal pieces counted with diagonal contacts joined (the 0.1 mm bridges)."""
    return int(ndimage.label(bits, structure=np.ones((3, 3)))[1])


def build_figure() -> dict:
    """Free-form delivery antenna vs the R80 mirror-symmetric best, same spec ruler."""
    free = json.loads(FREE.read_text(encoding="utf-8"))
    sym = json.loads(SYM.read_text(encoding="utf-8"))
    a = np.asarray(free["candidate"]["bits"], dtype=int)
    b = np.array([[int(c) for c in row] for row in sym["pattern_rows"]])
    assert a.shape == b.shape == (25, 25) and a[24, 12] == b[24, 12] == 1   # feed at bottom centre
    assert (b[:, :12] == b[:, 13:][:, ::-1]).all()
    f = np.asarray(free["frequency_ghz"])
    assert np.array_equal(f, np.asarray(sym["frequency_ghz"]))
    band = (f >= 26.5) & (f <= 29.5)
    curves = {
        "free": (np.asarray(free["response_db"]["S11"]), np.asarray(free["response_db"]["RealizedGainTotal"])),
        "sym": (np.asarray(sym["S11_db"]), np.asarray(sym["RealizedGainTotal_dbi"])),
    }
    wm = {k: min(-10 - s[band].max(), g[band].min() - 4) for k, (s, g) in curves.items()}
    assert round(wm["free"], 2) == 0.77 and abs(wm["sym"] - sym["margins_db"]["wm"]) < 1e-9
    pieces = {"free": components(a), "sym": components(b)}

    plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 7,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .6})
    blue, amber, ink, grid = "#234c68", "#bd6532", "#18212b", "#e3e8ec"
    fig = plt.figure(figsize=FIG_SIZE_IN, dpi=300)
    spec = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.45, 1.45], wspace=.32,
                            left=.005, right=.985, bottom=.22, top=.8)
    for k, (bits, label, color) in enumerate(((a, "(a) 自由圖形", blue), (b, "(b) 左右鏡射", amber))):
        ax = fig.add_subplot(spec[0, k])
        ax.imshow(bits, cmap=ListedColormap(["#f7f8f9", color]), interpolation="nearest")
        if k == 1:
            ax.axvline(12, color=ink, lw=.6, ls=(0, (2, 2)))
        ax.scatter([12], [25.3], marker="^", s=14, color=ink, zorder=3, clip_on=False)
        ax.set(xticks=[], yticks=[], xlim=(-.5, 24.5), ylim=(24.5, -.5))
        for side in ax.spines.values():
            side.set(visible=True, linewidth=.5, color="#9eabb5")
        ax.set_title(f"{label}\n{pieces['free' if k == 0 else 'sym']} 個金屬連通塊", fontsize=7, pad=3)
    for k, (idx, name, limit) in enumerate(((0, "S11（dB）", -10), (1, "正向實現增益（dBi）", 4))):
        ax = fig.add_subplot(spec[0, 2 + k])
        ax.axvspan(26.5, 29.5, color=grid, lw=0)
        ax.plot([26.5, 29.5], [limit, limit], color=ink, lw=.7, ls=(0, (3, 2)))
        for key, color, text in (("free", blue, "自由"), ("sym", amber, "鏡射")):
            ax.plot(f, curves[key][idx], color=color, lw=1.1, marker="o", ms=1.6,
                    label=f"{text} WM {wm[key]:+.2f}")
        ax.set_xlim(24, 32)
        ax.set_xticks(range(24, 33, 2))
        ax.set_xlabel("頻率（GHz）", labelpad=1)
        ax.set_title(f"({'c' if k == 0 else 'd'}) {name}", fontsize=7, pad=3)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(color=grid, lw=.4)
        if k == 1:
            ax.legend(loc="lower left", fontsize=6, frameon=False, handlelength=1.4)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300, metadata={"Software": None})
    plt.close(fig)
    return {"wm_db": {k: round(float(v), 4) for k, v in wm.items()}, "metal_components_8conn": pieces,
            "metal_fraction": {"free": round(float(a.mean()), 4), "sym": round(float(b.mean()), 4)}}


def build(source: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    out = ROOT / "output/pdf" / f"{source.stem}.pdf"
    qa = ROOT / "tmp/pdfs" / source.stem
    has_results = any(s.get("with_results") for s in data["sections"])
    figure_facts = build_figure() if has_results else None
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("JhengHei", str(font_dir / "msjh.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("JhengHeiBold", str(font_dir / "msjhbd.ttc"), subfontIndex=0))
    pdfmetrics.registerFontFamily("JhengHei", normal="JhengHei", bold="JhengHeiBold")

    out.parent.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    ink = colors.HexColor("#18212b")
    accent = colors.HexColor("#234c68")
    base = dict(fontName="JhengHei", textColor=ink, wordWrap="CJK")
    title = ParagraphStyle("Title", **base, fontSize=15.6, leading=20,
                           alignment=TA_CENTER, spaceAfter=3)
    subtitle = ParagraphStyle("Subtitle", **base, fontSize=7.8, leading=10,
                              alignment=TA_CENTER, spaceAfter=6)
    keywords = ParagraphStyle("Keywords", **base, fontSize=8.4, leading=11,
                              spaceAfter=3)
    body = ParagraphStyle("Body", **base, fontSize=9.5, leading=13.2,
                          alignment=TA_JUSTIFY, spaceAfter=2.5)
    heading = ParagraphStyle("Heading", fontName="JhengHeiBold", textColor=accent,
                             fontSize=9.9, leading=13.4, wordWrap="CJK",
                             spaceBefore=2.8, spaceAfter=1.2, keepWithNext=True)
    cell = ParagraphStyle("Cell", **base, fontSize=8.4, leading=11.2,
                          alignment=TA_LEFT)
    caption = ParagraphStyle("Caption", **base, fontSize=8.1, leading=11,
                             alignment=TA_JUSTIFY, spaceBefore=1.5, spaceAfter=2)
    refs = ParagraphStyle("Reference", fontName="Times-Roman", textColor=ink,
                          fontSize=7.6, leading=9, spaceAfter=1.2,
                          leftIndent=13, firstLineIndent=-13)
    refs_cjk = ParagraphStyle("ReferenceCJK", parent=refs, fontName="JhengHei")
    refheading = ParagraphStyle("ReferenceHeading", parent=heading, fontSize=8.8,
                                leading=11.5, spaceBefore=3)

    margin = 16 * mm
    width = A4[0] - 2 * margin
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=margin,
                          rightMargin=margin, topMargin=12 * mm, bottomMargin=11 * mm,
                          title=data["title"], author="", subject=data.get("footer", "中文延伸摘要"),
                          pageCompression=1)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("JhengHei", 7.3)
        canvas.setFillColor(colors.HexColor("#65717d"))
        canvas.drawString(margin, 6.5 * mm, data.get("footer", "中文延伸摘要"))
        canvas.drawRightString(A4[0] - margin, 6.5 * mm, str(_doc.page))
        canvas.restoreState()

    frame = Frame(margin, 11 * mm, width, A4[1] - 23 * mm,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Abstract", frames=frame, onPage=footer))
    story = [Paragraph(escape(data["title"]), title),
             Paragraph(escape(data["subtitle"]), subtitle)]
    if data.get("keywords"):
        story.append(Paragraph("<b>關鍵詞：</b>" + escape(data["keywords"]), keywords))
    for section in data["sections"]:
        story.append(Paragraph(escape(section["heading"]), heading))
        story.extend(Paragraph(rich(p), body) for p in section["paragraphs"])
        if section.get("with_results"):
            t = data["table"]
            rows = [[Paragraph("<b>" + escape(x) + "</b>", cell) for x in t["headers"]]]
            rows.extend([Paragraph(escape(x), cell) for x in row] for row in t["rows"])
            table = Table(rows, colWidths=[80, width - 80], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
                ("LINEABOVE", (0, 0), (-1, 0), .7, accent),
                ("LINEBELOW", (0, 0), (-1, 0), .4, colors.HexColor("#9eabb5")),
                ("LINEBELOW", (0, -1), (-1, -1), .7, accent),
                ("LINEBELOW", (0, 1), (-1, -2), .25, colors.HexColor("#d3dbe1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ]))
            story.append(KeepTogether([Paragraph(escape(t["caption"]), caption), table]))
            story.append(Spacer(1, 3))
            fig_h = width * FIG_SIZE_IN[1] / FIG_SIZE_IN[0]
            story.append(KeepTogether([Image(str(FIG), width=width, height=fig_h),
                                       Paragraph(escape(data["figure_caption"]), caption)]))
    if data.get("references"):
        story.append(Paragraph("參考文獻", refheading))
    for ref in data.get("references", []):
        # Each title is a clickable link to the precise previously reviewed version.
        text = f'[{ref["id"]}] ' + (f'<link href="{escape(ref["url"])}">{escape(ref["text"])}</link>'
                                    if ref.get("url") else escape(ref["text"]))
        story.append(Paragraph(text, refs_cjk if re.search(r"[一-鿿]", ref["text"]) else refs))
    doc.build(story)

    # Reopen the produced PDF, independently extract its text, and render every page.
    reader = PdfReader(str(out))
    extracted = "\n".join(p.extract_text() for p in reader.pages)
    (qa / "extracted.txt").write_text(extracted, encoding="utf-8")
    document = pymupdf.open(out)
    for n, page in enumerate(document):
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(qa / f"page-{n+1}.png")
    max_pages = data.get("max_pages", 1)
    if len(reader.pages) > max_pages:
        raise RuntimeError(f"Expected at most {max_pages} page(s), got {len(reader.pages)}; inspect render and edit layout.")
    normalized = re.sub(r"\s+", "", extracted)
    required = [data["title"], *[s["heading"] for s in data["sections"]], *data["qa_required"]] + (["參考文獻"] if data.get("references") else [])
    for phrase in required:
        assert re.sub(r"\s+", "", phrase) in normalized, phrase
    assert "\ufffd" not in extracted and "\x00" not in extracted
    p = document[0]
    bounds = []
    for page in document:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    box = pymupdf.Rect(span["bbox"])
                    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= page.rect.width and box.y1 <= page.rect.height, span
                    bounds.append(list(box))
    links = sum(len(list(page.get_links())) for page in document)
    assert links >= sum(1 for r in data.get("references", []) if r.get("url"))
    fonts = p.get_fonts(full=True)
    embedded_chinese = [f for f in fonts if "MicrosoftJhengHei" in f[3]]
    assert len(embedded_chinese) >= 2 and all(f[1] == "ttf" for f in embedded_chinese), fonts
    lowest_text = max(b[3] for b in bounds)
    report = {
        "pdf": out.relative_to(ROOT).as_posix(), "pages": len(document),
        "page_size_pt": list(p.rect), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **({"figure": FIG.relative_to(ROOT).as_posix(), "figure_sha256": hashlib.sha256(FIG.read_bytes()).hexdigest(),
            "figure_facts": figure_facts} if has_results else {}),
        "characters": len(normalized), "required_text_pass": True,
        "embedded_chinese_fonts": [f[3] for f in embedded_chinese],
        "reference_links": links, "text_within_page": True,
        "lowest_text_pt": round(lowest_text, 1),
        "renderer": "PyMuPDF 2x; Poppler was unavailable locally",
        "visual_review": "Pending conductor inspection of page-1.png",
    }
    (qa / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", type=Path, default=[SOURCE])
    for source in parser.parse_args().sources:
        build(source if source.is_absolute() else Path.cwd() / source)


if __name__ == "__main__":
    main()
