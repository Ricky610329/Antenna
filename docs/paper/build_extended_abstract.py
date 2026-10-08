"""Build the separate one-page Traditional Chinese extended abstract.

Dependencies: reportlab (PDF), pymupdf (render/QA), pypdf (text QA).
Run from any working directory. Local temporary dependencies may live in
tmp/pdfs/deps; final PDF is written to output/pdf/.
"""

from __future__ import annotations

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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

SOURCE = Path(__file__).with_name("extended-abstract-zh-2026-10.json")
OUT = ROOT / "output/pdf/extended-abstract-zh-2026-10.pdf"
QA = ROOT / "tmp/pdfs/extended-abstract-zh-2026-10"


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("JhengHei", str(font_dir / "msjh.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("JhengHeiBold", str(font_dir / "msjhbd.ttc"), subfontIndex=0))
    pdfmetrics.registerFontFamily("JhengHei", normal="JhengHei", bold="JhengHeiBold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    ink = colors.HexColor("#18212b")
    accent = colors.HexColor("#234c68")
    base = dict(fontName="JhengHei", textColor=ink, wordWrap="CJK")
    title = ParagraphStyle("Title", **base, fontSize=16.2, leading=22,
                           alignment=TA_CENTER, spaceAfter=5)
    subtitle = ParagraphStyle("Subtitle", **base, fontSize=8, leading=11,
                              alignment=TA_CENTER, spaceAfter=10)
    keywords = ParagraphStyle("Keywords", **base, fontSize=8.7, leading=12,
                              spaceAfter=8)
    body = ParagraphStyle("Body", **base, fontSize=10.5, leading=15.6,
                          alignment=TA_JUSTIFY, spaceAfter=5.2)
    heading = ParagraphStyle("Heading", fontName="JhengHeiBold", textColor=accent,
                             fontSize=10.3, leading=14.6, wordWrap="CJK",
                             spaceBefore=5.3, spaceAfter=3, keepWithNext=True)
    cell = ParagraphStyle("Cell", **base, fontSize=9, leading=13,
                          alignment=TA_LEFT)
    caption = ParagraphStyle("Caption", **base, fontSize=8.8, leading=12,
                             alignment=TA_CENTER, spaceBefore=2, spaceAfter=4)
    refs = ParagraphStyle("Reference", fontName="Times-Roman", textColor=ink,
                          fontSize=8, leading=9.6, spaceAfter=2,
                          leftIndent=13, firstLineIndent=-13)
    refheading = ParagraphStyle("ReferenceHeading", parent=heading, fontSize=9,
                                leading=12, spaceBefore=5)

    margin = 17 * mm
    width = A4[0] - 2 * margin
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=margin,
                          rightMargin=margin, topMargin=15 * mm, bottomMargin=14 * mm,
                          title=data["title"], author="", subject="中文一頁延伸摘要，內部討論稿",
                          pageCompression=1)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("JhengHei", 7.3)
        canvas.setFillColor(colors.HexColor("#65717d"))
        canvas.drawString(margin, 9 * mm, "中文延伸摘要 · 討論稿")
        canvas.drawRightString(A4[0] - margin, 9 * mm, str(_doc.page))
        canvas.restoreState()

    frame = Frame(margin, 14 * mm, width, A4[1] - 29 * mm,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Abstract", frames=frame, onPage=footer))
    story = [Paragraph(escape(data["title"]), title),
             Paragraph(escape(data["subtitle"]), subtitle),
             Paragraph("<b>關鍵詞：</b>" + escape(data["keywords"]), keywords)]
    for i, section in enumerate(data["sections"]):
        story.append(Paragraph(escape(section["heading"]), heading))
        story.extend(Paragraph(escape(p), body) for p in section["paragraphs"])
        if i == 2:
            t = data["table"]
            rows = [[Paragraph("<b>" + escape(x) + "</b>", cell) for x in t["headers"]]]
            rows.extend([Paragraph(escape(x), cell) for x in row] for row in t["rows"])
            table = Table(rows, colWidths=[44, width - 44], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
                ("LINEABOVE", (0, 0), (-1, 0), .7, accent),
                ("LINEBELOW", (0, 0), (-1, 0), .4, colors.HexColor("#9eabb5")),
                ("LINEBELOW", (0, -1), (-1, -1), .7, accent),
                ("LINEBELOW", (0, 1), (-1, 1), .25, colors.HexColor("#d3dbe1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(KeepTogether([Paragraph(escape(t["caption"]), caption), table]))
            story.append(Spacer(1, 4))
    story.append(Paragraph("參考文獻", refheading))
    for ref in data["references"]:
        # Each title is a clickable link to the precise previously reviewed version.
        text = f'[{ref["id"]}] <link href="{escape(ref["url"])}">{escape(ref["text"])}</link>'
        story.append(Paragraph(text, refs))
    doc.build(story)

    # Reopen the produced PDF, independently extract its text, and render every page.
    reader = PdfReader(str(OUT))
    extracted = "\n".join(p.extract_text() for p in reader.pages)
    (QA / "extracted.txt").write_text(extracted, encoding="utf-8")
    document = pymupdf.open(OUT)
    for n, page in enumerate(document):
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(QA / f"page-{n+1}.png")
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected one page, got {len(reader.pages)}; inspect render and edit layout.")
    normalized = re.sub(r"\s+", "", extracted)
    required = [data["title"], *[s["heading"] for s in data["sections"]],
                "+0.77", "+0.50", "-5.90", "-2.39", "3.51", "6個傳輸採樣點", "參考文獻"]
    for phrase in required:
        assert re.sub(r"\s+", "", phrase) in normalized, phrase
    assert "\ufffd" not in extracted and "\x00" not in extracted
    p = document[0]
    bounds = []
    for block in p.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                box = pymupdf.Rect(span["bbox"])
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= p.rect.width and box.y1 <= p.rect.height, span
                bounds.append(list(box))
    assert len(list(p.get_links())) >= 3
    fonts = p.get_fonts(full=True)
    embedded_chinese = [f for f in fonts if "MicrosoftJhengHei" in f[3]]
    assert len(embedded_chinese) >= 2 and all(f[1] == "ttf" for f in embedded_chinese), fonts
    report = {
        "pdf": str(OUT.relative_to(ROOT)), "pages": len(document),
        "page_size_pt": list(p.rect), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "characters": len(normalized), "required_text_pass": True,
        "embedded_chinese_fonts": [f[3] for f in embedded_chinese],
        "reference_links": len(list(p.get_links())), "text_within_page": True,
        "renderer": "PyMuPDF 2x; Poppler was unavailable locally",
        "visual_review": "Pending conductor inspection of page-1.png",
    }
    (QA / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
