"""Convert REPORT_R94_to_R156.md to PDF — CJK-aware, improved typography.

Uses Microsoft JhengHei (msjh.ttc) for Traditional Chinese rendering.
Pipeline: markdown -> HTML (with embedded base64 images + CJK font) -> PDF.
"""
import base64
import re
from pathlib import Path

import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MD_PATH = OUTPUTS / "REPORT_R94_to_R156.md"
PDF_PATH = OUTPUTS / "REPORT_R94_to_R156.pdf"

# Register CJK font — kaiu.ttf (DFKai-SB 標楷體) is pure TTF, avoids
# reportlab's TTC extraction bug that plagues msjh.ttc on this version.
FONT_REG = "C:/Windows/Fonts/kaiu.ttf"
FONT_BOLD = "C:/Windows/Fonts/kaiu.ttf"  # no separate bold; rely on synthesized bold
pdfmetrics.registerFont(TTFont("CJK", FONT_REG))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("CJK", normal="CJK", bold="CJK",
                   italic="CJK", boldItalic="CJK")


# Read markdown
md_text = MD_PATH.read_text(encoding="utf-8")

# Convert markdown to HTML
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "nl2br"],
)


# Inline images as base64
def inline_image(match):
    src = match.group(1)
    img_path = OUTPUTS / src
    if not img_path.exists():
        return match.group(0)
    ext = img_path.suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, f"image/{ext}")
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f'src="data:{mime};base64,{data}"'


html_body = re.sub(r'src="([^"]+\.(?:png|jpg|jpeg))"', inline_image, html_body)

# Wrap each top-level h2 with a page-break-before for clean section starts
# (skip the first one)
h2_seen = [False]
def add_pagebreak(match):
    if not h2_seen[0]:
        h2_seen[0] = True
        return match.group(0)
    return '<div style="page-break-before: always;"></div>' + match.group(0)
html_body = re.sub(r"<h2[^>]*>", add_pagebreak, html_body)

# Add figure caption styling: <p><img></p> followed by emphasized text -> caption
# (markdown converts ![alt](src) to <img alt="alt" src="src">; alt becomes caption)
def add_caption(match):
    img_tag = match.group(0)
    alt_match = re.search(r'alt="([^"]+)"', img_tag)
    if alt_match:
        alt = alt_match.group(1)
        return img_tag + f'<div class="figcaption">Figure: {alt}</div>'
    return img_tag
html_body = re.sub(r'<img[^>]+>', add_caption, html_body)


# Improved CSS — CJK font, better margins, page breaks, captions
html_doc = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
@page {
  size: A4 portrait;
  margin-top: 1.6cm;
  margin-bottom: 1.6cm;
  margin-left: 1.8cm;
  margin-right: 1.8cm;
  @frame footer {
    -pdf-frame-content: footerContent;
    bottom: 0.7cm; left: 1.8cm; right: 1.8cm; height: 0.5cm;
  }
}
body {
  font-family: 'CJK', sans-serif;
  font-size: 9.5pt;
  line-height: 1.55;
  color: #1f1f1f;
  word-wrap: break-word;
}
h1 {
  font-family: 'CJK', sans-serif;
  font-size: 18pt;
  color: #1a3a5e;
  border-bottom: 2.5pt solid #2c5282;
  padding-bottom: 6pt;
  margin-top: 14pt;
  margin-bottom: 12pt;
  page-break-after: avoid;
}
h2 {
  font-family: 'CJK', sans-serif;
  font-size: 14pt;
  color: #2c5282;
  border-bottom: 1.2pt solid #88a3c4;
  padding-bottom: 4pt;
  margin-top: 16pt;
  margin-bottom: 10pt;
  page-break-after: avoid;
}
h3 {
  font-family: 'CJK', sans-serif;
  font-size: 11.5pt;
  color: #345a8a;
  margin-top: 12pt;
  margin-bottom: 6pt;
  page-break-after: avoid;
}
h4 {
  font-family: 'CJK', sans-serif;
  font-size: 10.5pt;
  color: #4a6da0;
  margin-top: 10pt;
  margin-bottom: 5pt;
  page-break-after: avoid;
}
p {
  margin: 5pt 0;
  text-align: justify;
}
li {
  margin: 3pt 0;
}
strong, b {
  font-family: 'CJK', sans-serif;
  color: #c0392b;
}
em, i {
  color: #555;
}
code {
  background: #f4f4f4;
  padding: 1pt 4pt;
  font-family: 'Courier New', 'CJK', monospace;
  font-size: 8.5pt;
  color: #c0392b;
  border-radius: 2pt;
}
pre {
  background: #f6f8fa;
  border-left: 3pt solid #4a6da0;
  padding: 8pt 12pt;
  font-family: 'Courier New', 'CJK', monospace;
  font-size: 8pt;
  line-height: 1.4;
  page-break-inside: avoid;
  color: #2c3e50;
  margin: 8pt 0;
  word-wrap: break-word;
  white-space: pre-wrap;
}
pre code {
  background: transparent;
  color: #2c3e50;
  padding: 0;
  font-size: 8pt;
}
table {
  border-collapse: collapse;
  margin: 8pt 0;
  width: 100%;
  font-size: 8.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 0.5pt solid #99a;
  padding: 4pt 6pt;
  text-align: left;
  vertical-align: top;
}
th {
  background: #d6e3f2;
  font-family: 'CJK', sans-serif;
  color: #1a3a5e;
}
tr:nth-child(even) td {
  background: #f6f8fa;
}
img {
  max-width: 100%;
  margin-top: 8pt;
  margin-bottom: 2pt;
  page-break-inside: avoid;
}
.figcaption {
  font-size: 8pt;
  color: #555;
  font-style: italic;
  text-align: center;
  margin-bottom: 12pt;
  margin-top: 0;
}
blockquote {
  border-left: 3pt solid #6c8eaf;
  padding: 6pt 12pt;
  color: #444;
  background: #f7f9fc;
  margin: 8pt 0;
  font-style: italic;
}
hr {
  border: none;
  border-top: 0.6pt solid #ccc;
  margin: 14pt 0;
}
ul, ol { margin: 6pt 0; padding-left: 20pt; }
.cover {
  text-align: center;
  margin-top: 4cm;
  page-break-after: always;
}
.cover h1 {
  font-size: 26pt;
  color: #1a3a5e;
  border: none;
}
.cover .subtitle {
  font-size: 14pt;
  color: #4a6da0;
  margin: 1cm 0;
}
.cover .meta {
  font-size: 11pt;
  color: #666;
  margin-top: 2cm;
}
</style>
</head>
<body>
<div class="cover">
  <h1>Binary RIS 優化方法論</h1>
  <h1 style="font-size:18pt; color:#4a6da0; margin-top:6pt;">R94 → R156 完整報告</h1>
  <div class="subtitle">為 Patch Antenna Surrogate-in-the-loop Transition<br/>建立可信賴方法論</div>
  <div class="meta">
    報告日期：2026-05-03<br/>
    Branch: ricky/modernize<br/>
    累計：156 rounds, 200+ commits<br/>
    Author: Ricky × Claude (Opus 4.7, /loop session)
  </div>
</div>
""" + html_body + """
<div id="footerContent" style="text-align: right; font-size: 7pt; color: #888;">
  RIS Optimization Methodology Report — page <pdf:pagenumber/> of <pdf:pagecount/>
</div>
</body>
</html>"""


# Save HTML for debugging
html_path = OUTPUTS / "REPORT_R94_to_R156.html"
html_path.write_text(html_doc, encoding="utf-8")
print(f"Wrote intermediate HTML: {html_path}")

# Generate PDF
print(f"Generating PDF: {PDF_PATH}")


def link_callback(uri, rel):
    """Resolve src URIs (used for @font-face). Pass through data: URIs and abs paths."""
    if uri.startswith("data:"):
        return uri
    if uri.startswith(("http://", "https://", "file://")):
        return uri
    # Absolute Windows path
    if Path(uri).exists():
        return uri
    return uri


with open(PDF_PATH, "wb") as out:
    pisa_status = pisa.CreatePDF(html_doc, dest=out, encoding="utf-8",
                                 link_callback=link_callback)

if pisa_status.err:
    print(f"PDF generation had {pisa_status.err} errors")
else:
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"PDF generated: {PDF_PATH} ({size_kb:.1f} KB)")
