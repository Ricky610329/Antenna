---
name: project-ase-report-v2
description: ASE 進度報告 v2（2026-07 書面版）完成——逐節共筆流程、核心 framing、圖產線位置
metadata: 
  node_type: memory
  type: project
  originSessionId: 13490732-42ae-445f-bea0-32ca5037c0cc
---

ASE 書面進度報告 v2 = `C:\Users\Ricky\Desktop\進度報告\進度報告_v2.md`（§1–§8 完整，2026-07-16 定稿；圖與 md 同資料夾、裸檔名引用）。取代同夾舊版 `進度報告_2026-07.md`（§0–§11，Ricky 嫌掌控度低而重啟）。

**協作流程（Ricky 定的，之後照做）**：逐節討論→我起草→他改→我修順；先高層次後細節；短、少括號、每節結論先行；圖扛細節。對外用語：radiation pattern（不用「方向圖」）、款式/變體（不用「家族」）、帶外洩漏（不用「惡度」）；粗體小標接「 :」不接「。」。

**核心 framing（下次報告沿用）**：§4 隨機性=「迴圈飽和、拚初始化」（41 條學長軌跡、跨軌跡 σ、檢定力算帳；盲抽對照被 Ricky 否決過——池=學長軌跡聯集有循環嫌疑）；§7.1=「最後一哩路」（自首不公平→翻轉成能力主張；學長 18 筆達標 rad 雙過 0 vs 我們 1,547）；§7.2 款式目錄（≤10px 連通=款式，64 款，前三大近九成）；承重圖=工具非結論（per-pattern，不可推全域）。

**圖產線**：`script/figs/report_*.py` 7 支對外版＋champ_compare `--old-left`，commit 94c3fed（GAN）。數字快照會過期（工廠常駐跑），重跑腳本即更新。

**Why**: 下份報告 Ricky 期待直接更新同一組表/圖（結尾原有「儀表板宣告」段被他刪，但意圖保留）。
**How to apply**: 寫 ASE 報告先讀 v2 檔學語氣與用語表；紀錄數字一律先對 [[docs/records.json]]；分布/多樣性數字用 report_wm_dist / report_diversity 重算。
