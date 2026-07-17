---
name: project_experiment_catalog
description: 實驗 config 都登記在 configs/README.md；新增/改 config 或訓練腳本必須同步更新該表
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

訓練實驗的索引在 **`configs/README.md`**（對照表：每個 `configs/*.yaml` 對標哪個 baseline、改了什麼、想看什麼、舊 MultiConfig 編號）。已寫進 CLAUDE.md 慣例成為**硬規則**。

**Why**：使用者（2026-06-19）要求「把記錄這件事做成一個機制，讓你每次執行時都記得記錄」，避免實驗散落、重複造輪子。選 CLAUDE.md 慣例＋in-repo 文件而非 skill，因為 CLAUDE.md 每 session 自動載入且覆寫預設行為（skill 要主動叫才會跑，達不到「每次都記得」）。

**How to apply**：
- 新增/修改任何 `configs/*.yaml` 或訓練腳本 → **同步在 `configs/README.md` 加/改一行**（測什麼、與 base 差在哪）。這是硬規則，不是順手做。
- 產生新實驗 config **前**先掃 `configs/README.md`，避免重複。
- baseline：single 對標 `single_base.yaml`、dual 對標 `dual_base.yaml`；新實驗只改一個變因（A/B 才乾淨）。
- 已知缺口（截至 2026-06-19）：單埠沒有 `spectral_connectivity`（論文主方法）config；方向圖 loss config 待 [[project_radiation_pattern]] 實作後才能加。

相關 [[feedback_prefer_simplicity]] [[feedback_audit_existing_first]]。
