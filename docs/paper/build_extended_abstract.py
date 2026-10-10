"""Build the Traditional Chinese extended abstract and its attachments as PDFs.

Usage: python docs/paper/build_extended_abstract.py [source.json ...]
(default: the abstract; e.g. also symmetry-difficulty-zh-2026-10.json).

A source JSON has a title, subtitle and sections. A section holds either plain
"paragraphs" or ordered "blocks": {"p"}, {"bullets"}, {"sub"}, {"table"} or
{"figure", "caption"}; "new_page": true starts it on a fresh page. An optional
"layout" object tightens spacing (used by the one-page handout). Figures are drawn by
extended_abstract_figures.py from frozen evidence JSON; nothing here runs HFSS.

Dependencies: reportlab (PDF), pymupdf (render/QA), pypdf (text QA), matplotlib.
Local temporary dependencies may live in tmp/pdfs/deps; the PDF is written to
output/pdf/<stem>.pdf and QA renders to tmp/pdfs/<stem>/.
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

import pymupdf
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)
from reportlab.lib import textsplit as rl_textsplit
from reportlab.platypus import paragraph as rl_paragraph

import extended_abstract_figures as figures

# reportlab's line-start table is Japanese-oriented; let fullwidth Chinese punctuation hang in the margin too.
# Single-fragment paragraphs wrap through textsplit, multi-fragment ones through paragraph; each holds a copy.
for module in (rl_textsplit, rl_paragraph):
    module.ALL_CANNOT_START += "，；：！？"

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "extended-abstract-zh-2026-10.json"   # default source
CJK = re.compile(r"[\u4e00-\u9fff]")


def rich(text: str) -> str:
    """Escape for reportlab, keeping only <b>…</b> run-in labels and <sup>…</sup> exponents."""
    out = escape(text)
    for tag, rl in (("b", "b"), ("sup", "super")):
        out = out.replace(f"&lt;{tag}&gt;", f"<{rl}>").replace(f"&lt;/{tag}&gt;", f"</{rl}>")
    return out


def texts(data: dict) -> list[str]:
    """Every string that is set in the CJK font (references in Times are excluded)."""
    out = [data["title"], data["subtitle"], data.get("keywords", ""), data.get("footer", "")]
    for section in data["sections"]:
        out.append(section["heading"])
        out.extend(section.get("paragraphs", []))
        for block in section.get("blocks", []):
            out.extend(v for k, v in block.items() if k in ("p", "sub", "caption"))
            out.extend(block.get("bullets", []))
            if "table" in block:
                t = block["table"]
                out.extend([t["caption"], *t["headers"], *(x for row in t["rows"] for x in row)])
    out.extend(r["text"] for r in data.get("references", []) if CJK.search(r["text"]))
    return out


def build(source: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    out = ROOT / "output/pdf" / f"{source.stem}.pdf"
    qa = ROOT / "tmp/pdfs" / source.stem
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("JhengHei", str(font_dir / "msjh.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("JhengHeiBold", str(font_dir / "msjhbd.ttc"), subfontIndex=0))
    pdfmetrics.registerFontFamily("JhengHei", normal="JhengHei", bold="JhengHeiBold")
    # Microsoft JhengHei lacks e.g. U+2212, <=, >=; fail here instead of printing boxes.
    for name in ("JhengHei", "JhengHeiBold"):
        cmap = pdfmetrics.getFont(name).face.charToGlyph
        missing = sorted({c for t in texts(data) for c in re.sub(r"<[^>]+>", "", t) if ord(c) > 126 and ord(c) not in cmap})
        assert not missing, (name, missing)

    out.parent.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    ink = colors.HexColor("#18212b")
    accent = colors.HexColor("#234c68")
    muted = colors.HexColor("#65717d")
    rule = colors.HexColor("#d3dbe1")
    base = dict(fontName="JhengHei", textColor=ink, wordWrap="CJK")
    # Defaults leave room to breathe; a one-page handout overrides them via "layout".
    lay = {"font": 10, "leading": 17.5, "para_after": 7, "margin_mm": 20, "top_mm": 18, "bottom_mm": 17,
           "heading_before": 16, "heading_after": 6, "sub_before": 10, "fig_gap": 8, "keep_mm": 45,
           **data.get("layout", {})}
    title = ParagraphStyle("Title", **base, fontSize=16, leading=21, alignment=TA_CENTER, spaceAfter=4)
    subtitle = ParagraphStyle("Subtitle", **base, fontSize=8, leading=11, alignment=TA_CENTER, spaceAfter=8)
    keywords = ParagraphStyle("Keywords", **base, fontSize=8.6, leading=12, spaceAfter=4)
    body = ParagraphStyle("Body", **base, fontSize=lay["font"], leading=lay["leading"], alignment=TA_JUSTIFY,
                          spaceAfter=lay["para_after"], allowWidows=0, allowOrphans=0)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=11, bulletIndent=1.5,
                            bulletFontName="JhengHei", bulletFontSize=9, spaceAfter=lay["para_after"] * .6)
    heading = ParagraphStyle("Heading", fontName="JhengHeiBold", textColor=accent, fontSize=11.2,
                             leading=15, wordWrap="CJK", spaceBefore=lay["heading_before"],
                             spaceAfter=lay["heading_after"], keepWithNext=True)
    sub = ParagraphStyle("Sub", parent=heading, fontSize=10, leading=14, textColor=ink,
                         spaceBefore=lay["sub_before"], spaceAfter=lay["heading_after"] * .6)
    cell = ParagraphStyle("Cell", **base, fontSize=8.8, leading=13, alignment=TA_LEFT)
    caption = ParagraphStyle("Caption", **base, fontSize=8.4, leading=12.4, alignment=TA_JUSTIFY,
                             spaceBefore=3, spaceAfter=lay["fig_gap"] + 2)
    refs = ParagraphStyle("Reference", fontName="Times-Roman", textColor=ink, fontSize=7.6, leading=9,
                          spaceAfter=1.2, leftIndent=13, firstLineIndent=-13)
    refs_cjk = ParagraphStyle("ReferenceCJK", parent=refs, fontName="JhengHei")
    refheading = ParagraphStyle("ReferenceHeading", parent=heading, fontSize=8.8, leading=11.5, spaceBefore=4)

    margin = lay["margin_mm"] * mm
    width = A4[0] - 2 * margin
    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=lay["top_mm"] * mm, bottomMargin=lay["bottom_mm"] * mm, title=data["title"], author="",
                          subject=data.get("footer", "中文延伸摘要"), pageCompression=1)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("JhengHei", 7.3)
        canvas.setFillColor(muted)
        canvas.drawString(margin, 6.5 * mm, data.get("footer", "中文延伸摘要"))
        canvas.drawRightString(A4[0] - margin, 6.5 * mm, str(_doc.page))
        canvas.restoreState()

    frame = Frame(margin, lay["bottom_mm"] * mm, width, A4[1] - (lay["top_mm"] + lay["bottom_mm"]) * mm,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Abstract", frames=frame, onPage=footer))

    def table(t: dict):
        rows = [[Paragraph("<b>" + escape(x) + "</b>", cell) for x in t["headers"]]]
        rows.extend([Paragraph(rich(x), cell) for x in row] for row in t["rows"])
        first = t.get("first_col_mm", 28) * mm
        grid = Table(rows, colWidths=[first, width - first], hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
            ("LINEABOVE", (0, 0), (-1, 0), .7, accent),
            ("LINEBELOW", (0, 0), (-1, 0), .4, colors.HexColor("#9eabb5")),
            ("LINEBELOW", (0, -1), (-1, -1), .7, accent),
            ("LINEBELOW", (0, 1), (-1, -2), .25, rule),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ]))
        return [KeepTogether([Paragraph(escape(t["caption"]), caption), grid]), Spacer(1, lay["fig_gap"])]

    built = {}

    def figure(name: str, text: str):
        if name not in built:
            built[name] = figures.FIGURES[name]()
        path = built[name]["path"]
        w, h = ImageReader(str(path)).getSize()
        return [KeepTogether([Spacer(1, lay["fig_gap"]), Image(str(path), width=width, height=width * h / w),
                              Paragraph(rich(text), caption)])]

    story = [Paragraph(escape(data["title"]), title), Paragraph(escape(data["subtitle"]), subtitle)]
    if data.get("keywords"):
        story.append(Paragraph("<b>關鍵詞：</b>" + escape(data["keywords"]), keywords))
    for section in data["sections"]:
        story.append(PageBreak() if section.get("new_page") else CondPageBreak(lay["keep_mm"] * mm))
        story.append(Paragraph(escape(section["heading"]), heading))
        story.extend(Paragraph(rich(p), body) for p in section.get("paragraphs", []))
        for block in section.get("blocks", []):
            if "p" in block:
                story.append(Paragraph(rich(block["p"]), body))
            elif "sub" in block:
                story.append(Paragraph(escape(block["sub"]), sub))
            elif "bullets" in block:
                story.extend(Paragraph(rich(b), bullet, bulletText="•") for b in block["bullets"])
            elif "table" in block:
                story.extend(table(block["table"]))
            elif "figure" in block:
                story.extend(figure(block["figure"], block["caption"]))
            else:
                raise ValueError(f"unknown block {sorted(block)}")
    if data.get("references"):
        story.append(Paragraph("參考文獻", refheading))
    for ref in data.get("references", []):
        # Each title is a clickable link to the precise previously reviewed version.
        text = f'[{ref["id"]}] ' + (f'<link href="{escape(ref["url"])}">{escape(ref["text"])}</link>'
                                    if ref.get("url") else escape(ref["text"]))
        story.append(Paragraph(text, refs_cjk if CJK.search(ref["text"]) else refs))
    doc.build(story)

    # Reopen the produced PDF, independently extract its text, and render every page.
    reader = PdfReader(str(out))
    extracted = "\n".join(p.extract_text() for p in reader.pages)
    (qa / "extracted.txt").write_text(extracted, encoding="utf-8")
    document = pymupdf.open(out)
    for old in qa.glob("page-*.png"):
        old.unlink()
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
    fonts = document[0].get_fonts(full=True)
    embedded_chinese = [f for f in fonts if "MicrosoftJhengHei" in f[3]]
    assert len(embedded_chinese) >= 2 and all(f[1] == "ttf" for f in embedded_chinese), fonts
    report = {
        "pdf": out.relative_to(ROOT).as_posix(), "pages": len(document),
        "page_size_pt": list(document[0].rect), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "figures": {name: {"path": f["path"].relative_to(ROOT).as_posix(),
                           "sha256": hashlib.sha256(f["path"].read_bytes()).hexdigest(), "facts": f["facts"]}
                    for name, f in built.items()},
        "characters": len(normalized), "required_text_pass": True, "glyph_coverage_pass": True,
        "embedded_chinese_fonts": [f[3] for f in embedded_chinese],
        "reference_links": links, "text_within_page": True,
        "renderer": "PyMuPDF 2x; Poppler was unavailable locally",
        "visual_review": "Pending conductor inspection of page-*.png",
    }
    (qa / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "figures"}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", type=Path, default=[SOURCE])
    for source in parser.parse_args().sources:
        build(source if source.is_absolute() else Path.cwd() / source)


if __name__ == "__main__":
    main()
