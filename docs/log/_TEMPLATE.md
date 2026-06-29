# Round NN — <主題>

- **狀態**: proposed  <!-- proposed | running | analyzing | concluded | archived -->
- **提出 / 開跑 / 結論**: YYYY-MM-DD / — / —
- **一句話問題**: <hypothesis 的核心問句>
- **一句話結論 (TL;DR)**: <結論出來後補;沒結論寫「待分析」>
- **指向**: configs/README「<段>」· 結果夾 §3 · memory [[project_xxx]] · 設計文件 docs/xxx.md

> 本檔只放**連結指向**其他層(configs/README=config 全集、docs/設計文件=為什麼、結果夾=原始數字),不複製內容。

## 1. 假設 (Propose)
- **問題 / 假設**:
- **為何現在做**(承接哪個 round / 哪個候選):
- **預期結果與判準**(怎樣算對、怎樣算錯):
- **依據**: 設計文件 `docs/xxx.md` · memory `[[project_xxx]]` · ONGOING 候選項

## 2. 實驗設計 (Design)
| 臂 | config | 機器 | 唯一變因 | 對照 baseline |
|---|---|---|---|---|
| A | `single_xxx` | — | … | … |
- **判準**(用哪把尺): worst-margin(dB) vs HFSS-call 曲線 + 對比 random best-of-N;A/B 看 `<診斷量>`。
- **HFSS 預算**: 跑到 ~500 epoch。

## 3. 執行紀錄 (Run)
| 臂 | 機器 | 狀態 / 進度 | 結果夾 (NAS) |
|---|---|---|---|
| A | — | — | `T:\…\result\[Patch-single-…]<name>\` |
- **事件 / crash / 全域變更**:

## 4. 分析 (Analyze)
<!-- 跑 `python -m script.round_report --round NN --runs … --labels …` → 貼下方數字表;圖在 assets/round-NN/ -->
| 臂 | 最佳 worst_margin | 達到 epoch |
|---|---|---|
- 最佳 pattern + S11/Gain: `assets/round-NN/<arm>_best.png`
- worst-margin vs HFSS-call 疊圖: `assets/round-NN/benchmark.png`
- **觀察**:

## 5. 結論 (Conclude)
- **學到什麼**:
- **決策**:
- **促成 / 排除了哪個候選**(回寫 ONGOING 🔜):

## 6. 後續決策 (Next)
- **解鎖**: → Round NN+1 / 哪個候選升主線
- **新產生的待辦**(同步進 ONGOING 🔜 或 configs/README 已知缺口):

## 7. 歸檔指向 (Archive)
- configs/README 列:
- 結果夾:
- memory: `[[...]]`
- 設計文件:
- ONGOING 動作: 已從 🔵 移除 / ✅ 區改成指向本檔
