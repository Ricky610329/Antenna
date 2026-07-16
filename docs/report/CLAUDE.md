# docs/report/ 撰寫規範（成果報告產線）

> 交付給使用者/實驗室看的報告。命名：`progress-*.md`（成果報告）／`status-*.md`（現況快照）。

## 產線

1. 圖：一律由 `script/figs/report_*.py` 產生（決定性可重跑）→ 落 `assets/`；
   共用 `script/figs/report_r1r10_style.py`（色票/Microsoft JhengHei/`save()`）；
   新腳本在 `script/figs/README.md` 索引 +1 行。
2. 文：本資料夾 `<stem>.md`，圖用相對路徑 `assets/x.png`。
3. PDF：`python docs/report/build_pdf.py <stem|md路徑> [--out-name 名] [--scale "圖.png=56%,…"]`
   （md → html → headless Edge → PDF → pymupdf 蓋頁碼；流式排版＋h2/h3 keep-together 防標題孤懸；
   md 可在 repo 外——對外交付報告如桌面「進度報告」也走這條線）。

## 硬驗收（不可跳過）

- **每張 PNG 產出後用 Read 目檢**：中文字重疊、裁切、豆腐字——修到全過才組 PDF。
- **PDF 逐頁渲染檢查**（pymupdf 出 PNG 再 Read）：分頁切圖、字型、表格溢出。

## 內容紀律

- 開頭必有「Scope 與 Limitations」（模擬≠量測、spec 範圍、族群侷限）——不把工具當 production system。
- 「▍一句話重點」框：每大節開頭一句可抽成投影片的總結。
- 數字全數對 `docs/log/round-*.md` 與公證紀錄核對；未公證值一律標「單次」。

## git 規則

- `*.html`（build 中間產物）**不進 git**（.gitignore pattern）；`*.pdf`（交付物）與 `assets/*.png` **進 git**。
