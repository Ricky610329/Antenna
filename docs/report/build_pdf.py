# -*- coding: utf-8 -*-
"""build_pdf.py — 把 progress-r1-r10.md 組成 PDF（繁中）。

產線（此機無 pandoc/LaTeX,走瀏覽器印製）:
    md --(markdown 套件)--> html（內嵌 CSS,微軟正黑）--(headless Edge)--> pdf
用法: <ant env python> docs/report/build_pdf.py
"""
import os
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = sys.argv[1] if len(sys.argv) > 1 else "progress-r1-r10"   # 檔名（不含副檔）,預設 R1-R10 主報告
MD = os.path.join(HERE, f"{STEM}.md")
HTML = os.path.join(HERE, f"{STEM}.html")
PDF = os.path.join(HERE, f"{STEM}.pdf")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft JhengHei", sans-serif; color: #0b0b0b; background: #fff;
       font-size: 10.5pt; line-height: 1.65; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 17pt; border-bottom: 3px solid #1c5cab; padding-bottom: 6px; }
h2 { font-size: 13.5pt; color: #1c5cab; border-bottom: 1.5px solid #e1e0d9;
     padding-bottom: 4px; margin-top: 22px; page-break-after: avoid; }
h2.pb { page-break-before: always; }
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
strong { color: #0b0b0b; }
"""


def main():
    text = open(MD, encoding="utf-8").read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    # 流式排版:不強制章起新頁(避免孤兒頁/半空頁);h2 靠 page-break-after:avoid 黏住後文
    html = ("<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<title></title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    open(HTML, "w", encoding="utf-8").write(html)
    if os.path.exists(PDF):
        os.remove(PDF)
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", "file:///" + HTML.replace("\\", "/")],
                   check=True, timeout=120)
    print(f"→ {PDF}  ({os.path.getsize(PDF) // 1024} KB)")


if __name__ == "__main__":
    main()
