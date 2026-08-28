---
name: project_research_log
description: "研究時間軸在 docs/log/(round=燒HFSS);live=configs/ONGOING.md;R1-R14 全歸檔(2026-07-08),現階段=論文寫作衝刺(P0=算法+分數)+等實作量測;接手先讀 log/README+ONGOING+scratch 最新塊"
metadata: 
  node_type: memory
  type: project
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

研究進度追蹤＝四層分工：**`docs/log/`**（round 一檔，append-only 時間軸）／**`configs/ONGOING.md`**
（live 操作板）／**`configs/README.md`**（config 全集，硬規則同步）／`docs/` 設計文件（為什麼，固定）。
round 檔只連結不複製。工具：`script/status.py`（掃 NAS 真相，更新 ONGOING 前必跑）、
`script/round_report.py`（歸檔圖+數字）、`script/analyze.py`（重現診斷）。

**現況（2026-07-08）：R1–R14 全數歸檔**，HFSS 探索線到報酬遞減點、主動收束（decisions「戰略轉向」）。
- 線上線 R1–R5＝診斷史：天花板 −2.89@R4（誠實歸因＝探索非利用；R4 三件事寫法見 thesis_outline Ch.6.4）；
  R5 把 SM 訓起來（fit_loss→0.45）但 wm 未跟上＝範式天花板。
- 批次線 R6–R14：紀錄 −2.68→+0.22；製造冠軍 x00；w17 特殊性；組件設計語言（詳 [[project_w17_champion]]）。
- **現階段＝論文寫作衝刺**（Ricky 定調 2026-07-08：**算法與分數先行**＝`docs/thesis_outline.md` v2 的
  P0＝Ch.4 方法論＋Ch.5 定量結果；背景/相關工作 P2 後補）＋等實作量測（x00）＋規則→generator 工程（候選）。
- 報告成品：`docs/report/progress-r1-r10.{md,pdf}`（15 頁）＋`progress-r11-r14.{md,pdf}`（5 頁，分數先行版）；
  重建＝`build_pdf.py <stem>`（headless Edge 產線）。
- 敘事定調見 [[project_narrative_pivot]]（agent＋human-in-the-loop 調度工具箱；Sengupta 三篇在 docs/reference/）。

**How to apply**：跨 session 接手先讀 `docs/log/README.md`＋`configs/ONGOING.md`＋scratch 最新塊。
新工作優先序：論文 P0 章節 > 量測支援 > generator 工程 > 新 HFSS 批（除非有新假設，別再刷探索）。
關聯 [[project_experiment_catalog]] [[project_benchmark_vs_random]] [[project_sm_training_redesign]]

**2026-08-07 相位更新**:專案獲學長點頭;**single-port 收尾至 8/15**(消融+方法論收攏+合格 pattern 交付),之後不開新戰線;帶外壓制可外包後級濾波器(pattern 職責=帶內+rad+可製造);下一線=dual port 濾波器(日月光框架,未開工)。詳 decisions「Single-port 收尾期程」。
