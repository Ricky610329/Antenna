# -*- coding: utf-8 -*-
"""build_pdf.py — 把報告 md 組成 PDF（繁中）。

產線（此機無 pandoc/LaTeX,走瀏覽器印製）:
    md --(markdown 套件)--> html（內嵌 CSS,微軟正黑）--(headless Edge)--> pdf --(pymupdf)--> 蓋頁碼
排版守則：流式（不強制章起新頁）；h2/h3 與後續 1-2 個區塊包 .keep 防標題孤懸頁尾；
orphans/widows=3；Chromium 不支援 CSS @page 頁碼 → pymupdf 後製「X / N」底部置中。

用法:
    <ant env python> docs/report/build_pdf.py                    # 預設 stem=progress-r1-r10
    <ant env python> docs/report/build_pdf.py <stem|md路徑>
        [--out-name 名稱]                  # 輸出 pdf 檔名（不含副檔,預設=stem）
        [--scale "圖.png=56%,另一圖.png=84%"]   # 特定圖縮尺（方形/手繪圖不需滿版寬）
md 在 docs/report 之外也可（給桌面交付報告用）；HTML 中間檔生在 md 同夾（相對圖引用）、印完即刪。
"""
import argparse
import os
import re
import subprocess

import fitz
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft JhengHei", sans-serif; color: #0b0b0b; background: #fff;
       font-size: 10.5pt; line-height: 1.65; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 17pt; border-bottom: 3px solid #1c5cab; padding-bottom: 6px; }
h2 { font-size: 13.5pt; color: #1c5cab; border-bottom: 1.5px solid #e1e0d9;
     padding-bottom: 4px; margin-top: 22px; page-break-after: avoid; break-after: avoid-page; }
h3 { font-size: 11.5pt; color: #0b0b0b; margin-top: 18px; page-break-after: avoid; break-after: avoid-page; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 10px 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #d5d3cb; padding: 4px 7px; text-align: left; }
th { background: #eef2f8; }
tr:nth-child(even) td { background: #fafaf8; }
img { max-width: 100%; display: block; margin: 12px auto 4px; page-break-inside: avoid; }
blockquote { background: #eef2f8; border-left: 4px solid #1c5cab; margin: 12px 0;
             padding: 8px 14px; page-break-inside: avoid; }
blockquote p { margin: 0; }
code { font-family: Consolas, monospace; background: #f2f1ec; padding: 1px 4px;
       border-radius: 3px; font-size: 9pt; }
hr { border: none; border-top: 1px solid #e1e0d9; margin: 18px 0; }
em { color: #52514e; font-size: 9.5pt; }
strong { color: #0b0b0b; }
.keep { page-break-inside: avoid; break-inside: avoid; }
"""

BLOCK = r"(?:<(?:p|ul|ol|blockquote|table)>[\s\S]*?</(?:p|ul|ol|blockquote|table)>\s*)"


def _keep_together(body):
    """小節標題與其後 1-2 個區塊包成不可分頁 div，避免標題孤懸頁尾。"""
    body = re.sub(rf"(<h3>[\s\S]*?</h3>\s*{BLOCK}{{1,2}})", r'<div class="keep">\1</div>', body)
    body = re.sub(rf"(<h2>[\s\S]*?</h2>\s*{BLOCK}{{1}})", r'<div class="keep">\1</div>', body)
    return body


def _stamp_pages(pdf):
    """底部置中蓋「X / N」灰字（內文邊界外）。"""
    doc = fitz.open(pdf)
    n = len(doc)
    for i, page in enumerate(doc):
        label = f"{i + 1} / {n}"
        w = fitz.get_text_length(label, fontname="helv", fontsize=9)
        page.insert_text(((page.rect.width - w) / 2, page.rect.height - 22),
                         label, fontname="helv", fontsize=9, color=(0.32, 0.32, 0.31))
    doc.save(pdf + ".tmp")
    doc.close()
    os.replace(pdf + ".tmp", pdf)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="progress-r1-r10",
                    help="docs/report 內的 stem,或任意 md 路徑")
    ap.add_argument("--out-name", default=None, help="輸出 pdf 檔名（不含副檔,預設=stem）")
    ap.add_argument("--scale", default="", help='特定圖縮尺,如 "承重圖.png=56%%,新架構.png=84%%"')
    args = ap.parse_args()

    md = args.src if args.src.lower().endswith(".md") else os.path.join(HERE, f"{args.src}.md")
    md = os.path.abspath(md)
    folder = os.path.dirname(md)
    stem = args.out_name or os.path.splitext(os.path.basename(md))[0]
    html_p = os.path.join(folder, f"_build_{stem}.html")
    pdf = os.path.join(folder, f"{stem}.pdf")

    body = markdown.markdown(open(md, encoding="utf-8").read(),
                             extensions=["tables", "fenced_code"])
    for pair in filter(None, (s.strip() for s in args.scale.split(","))):
        name, w = pair.split("=")
        body = body.replace(f'src="{name}"', f'src="{name}" style="max-width:{w}"')
    body = _keep_together(body)
    html = ("<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<title>{stem}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    open(html_p, "w", encoding="utf-8").write(html)
    if os.path.exists(pdf):
        os.remove(pdf)
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file:///" + html_p.replace("\\", "/")],
                   check=True, timeout=120)
    os.remove(html_p)
    n = _stamp_pages(pdf)
    print(f"→ {pdf}  ({os.path.getsize(pdf) // 1024} KB, {n} 頁, 頁碼已蓋)")


if __name__ == "__main__":
    main()
