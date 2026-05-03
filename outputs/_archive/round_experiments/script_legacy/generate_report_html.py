"""Generate a standalone, print-friendly HTML report from REPORT_R94_to_R156.md.

Browser handles CJK font fallback natively (uses system fonts), so we don't
need to bundle a font file. User can open in browser and print to A4 PDF.

Pipeline: markdown -> HTML with base64-inlined images + clean print CSS.
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MD_PATH = OUTPUTS / "REPORT_R94_to_R156.md"
HTML_PATH = OUTPUTS / "REPORT_R94_to_R156.html"


md_text = MD_PATH.read_text(encoding="utf-8")
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc", "nl2br"],
)


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


# Wrap each h2 (except first) with page-break-before for clean print sections
h2_seen = [False]
def add_pagebreak(match):
    if not h2_seen[0]:
        h2_seen[0] = True
        return match.group(0)
    return '<div style="page-break-before: always;"></div>' + match.group(0)
html_body = re.sub(r"<h2[^>]*>", add_pagebreak, html_body)


# Add Figure caption
def add_caption(match):
    img_tag = match.group(0)
    alt_match = re.search(r'alt="([^"]+)"', img_tag)
    if alt_match:
        alt = alt_match.group(1).replace('"', '&quot;')
        return img_tag + f'<div class="figcaption">Figure: {alt}</div>'
    return img_tag
html_body = re.sub(r'<img[^>]+>', add_caption, html_body)


CSS = """
:root {
  --color-text: #1f1f1f;
  --color-h1: #1a3a5e;
  --color-h2: #2c5282;
  --color-h3: #345a8a;
  --color-h4: #4a6da0;
  --color-accent: #c0392b;
  --color-codebg: #f6f8fa;
  --color-tableborder: #99a;
  --color-tableheader: #d6e3f2;
  --color-blockquote: #6c8eaf;
}
@media print {
  @page {
    size: A4;
    margin: 1.6cm 1.8cm;
  }
  body { font-size: 10.5pt; }
  h2 { page-break-before: always; }
  h2:first-of-type { page-break-before: auto; }
  pre, table, img { page-break-inside: avoid; }
}
html { color-scheme: light; }
body {
  font-family: "Microsoft JhengHei", "微軟正黑體", "PingFang TC",
    "Heiti TC", "Hiragino Sans", "Noto Sans CJK TC", "Helvetica Neue",
    Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.65;
  color: var(--color-text);
  max-width: 880px;
  margin: 1em auto;
  padding: 0 1.5em 4em 1.5em;
  background: #fdfdfd;
}
h1 {
  font-size: 22pt;
  color: var(--color-h1);
  border-bottom: 3px solid var(--color-h2);
  padding-bottom: 6px;
  margin-top: 28pt;
}
h2 {
  font-size: 16pt;
  color: var(--color-h2);
  border-bottom: 1.5px solid #88a3c4;
  padding-bottom: 4px;
  margin-top: 24pt;
}
h3 {
  font-size: 13pt;
  color: var(--color-h3);
  margin-top: 18pt;
}
h4 {
  font-size: 11.5pt;
  color: var(--color-h4);
  margin-top: 14pt;
}
p { text-align: justify; margin: 0.6em 0; }
li { margin: 0.25em 0; }
strong, b { color: var(--color-accent); font-weight: 700; }
em, i { color: #555; }
code {
  background: var(--color-codebg);
  padding: 1px 5px;
  font-family: "JetBrains Mono", "Cascadia Code", "Source Code Pro",
    "Consolas", "Courier New", monospace;
  font-size: 0.9em;
  color: var(--color-accent);
  border-radius: 3px;
}
pre {
  background: var(--color-codebg);
  border-left: 4px solid var(--color-h4);
  padding: 10pt 14pt;
  font-family: "JetBrains Mono", "Cascadia Code", "Consolas",
    "Courier New", monospace;
  font-size: 9.5pt;
  line-height: 1.45;
  color: #2c3e50;
  margin: 8pt 0;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  border-radius: 3px;
}
pre code {
  background: transparent;
  color: inherit;
  padding: 0;
  font-size: inherit;
}
table {
  border-collapse: collapse;
  margin: 8pt 0;
  width: 100%;
  font-size: 10pt;
}
th, td {
  border: 1px solid var(--color-tableborder);
  padding: 5pt 8pt;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--color-tableheader);
  font-weight: 700;
  color: var(--color-h1);
}
tr:nth-child(even) td { background: #f6f8fa; }
img {
  max-width: 100%;
  margin-top: 10pt;
  margin-bottom: 4pt;
  display: block;
  border: 1px solid #e2e2e2;
  border-radius: 3px;
}
.figcaption {
  font-size: 9pt;
  color: #555;
  font-style: italic;
  text-align: center;
  margin: 4pt 0 16pt 0;
}
blockquote {
  border-left: 4px solid var(--color-blockquote);
  padding: 6pt 14pt;
  color: #444;
  background: #f7f9fc;
  margin: 10pt 0;
  font-style: italic;
}
hr { border: none; border-top: 1px solid #ccc; margin: 18pt 0; }
ul, ol { margin: 8pt 0; padding-left: 24pt; }

/* Cover page */
.cover {
  text-align: center;
  margin: 4em 0 6em 0;
  padding: 4em 1em;
  border-bottom: 2px solid #e2e2e2;
}
.cover h1 {
  font-size: 32pt;
  color: var(--color-h1);
  border: none;
  margin-bottom: 8pt;
}
.cover .subtitle {
  font-size: 14pt;
  color: var(--color-h3);
  margin: 1em 0 2em 0;
}
.cover .meta {
  font-size: 11pt;
  color: #666;
  margin-top: 1.5em;
  line-height: 1.9;
}

/* Toolbar at top for browser users */
.toolbar {
  position: sticky;
  top: 0;
  background: rgba(253, 253, 253, 0.97);
  border-bottom: 1px solid #e2e2e2;
  padding: 6pt 0;
  margin: -1em -1.5em 1.5em -1.5em;
  padding-left: 1.5em;
  padding-right: 1.5em;
  font-size: 10pt;
  color: #555;
  text-align: center;
  z-index: 100;
}
.toolbar kbd {
  background: #eee;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 1px 5px;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 0.9em;
}
@media print { .toolbar { display: none; } }
"""


html_doc = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RIS 優化方法論：R94 → R156 完整報告</title>
<style>{CSS}</style>
</head>
<body>
<div class="toolbar">
  瀏覽用：直接閱讀。列印 A4 PDF：按 <kbd>Ctrl</kbd>+<kbd>P</kbd>，目的地選「另存為 PDF」，紙張 A4，邊界 default。
</div>
<div class="cover">
  <h1>Binary RIS 優化方法論</h1>
  <h1 style="font-size:20pt; color:var(--color-h3); margin-top:6pt;">R94 → R156 完整報告</h1>
  <div class="subtitle">為 Patch Antenna Surrogate-in-the-loop Transition<br>建立可信賴方法論</div>
  <div class="meta">
    報告日期：2026-05-03<br>
    Branch: <code>ricky/modernize</code><br>
    累計：156 rounds, 200+ commits<br>
    Author: Ricky × Claude (Opus 4.7, /loop session)
  </div>
</div>
{html_body}
</body>
</html>"""


HTML_PATH.write_text(html_doc, encoding="utf-8")
size_kb = HTML_PATH.stat().st_size / 1024
print(f"HTML written: {HTML_PATH} ({size_kb:.1f} KB)")
print(f"Open in browser. Print to PDF via Ctrl+P -> 'Save as PDF', A4 size.")
