# Round 15 — 對照組實驗：push-button（GA）vs 工具箱（知情）@ 同一組件空間

- **狀態**: running（2026-07-09 發車;SM v4 已先出爐）
- **提出 / 開跑 / 結論**: 2026-07-08 / 2026-07-09 / —
- **一句話問題**: 同一組件空間、同 SM、同 HFSS 驗證預算下——文獻式 push-button（GA 全代理盲搜,MWSCAS 2024 立場）跟我們的知情調度,誰交出更多/更好的三標解？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 文獻立靶=docs/reference/（MWSCAS 2024 全自動）;敘事=memory project_narrative_pivot;
  SM v4=sm_reanchor4.pth;前作 [round-14](round-14-component-axis.md)

## 1. 假設 (Propose)
- 背景：敘事主張「agent+human 調度工具箱 ≫ push-button」需要自家定量數據;R13/R14 給了共同的組件空間。
- 賦能器（已完成,2026-07-09）：**SM v4**（訓練集 600→1137 現行真值,雙保真=harvest 預訓+dedust 重錨）——
  held-out 中位 0.80→0.61、爛區 3.45→2.97;**作戰區 0.45→0.44=飽和**（資料翻倍未解鎖→已非資料受限,
  誠實記錄=論文可用發現）。三臂共用 v4。
- 判準（發車前寫死）：同額預算（每臂 30 HFSS）比三項——①三標過數 ②best（字典序:硬約束→帶內 margin→oob）
  ③oob 分布。N 臂（空間隨機 20）隔離「空間 vs 搜尋」貢獻。
  **兩頭都是結論**：知情勝=「工具箱>push-button」自家數據;GA 勝=誠實收下+收割其解。

## 2. 實驗設計 (Design)
組件空間（三臂同一個）：錨點∈{x00,c21,a15,c25} × 0-3 塊（合法位置表 38-53 個/錨點,尺寸 2-4）→ 10-5-10 對稱化。
| 臂 | 選擇機制 | 筆數 | 機器 |
|---|---|---|---|
| G push-button | GA pop64×40 代,fitness=SM 預測 wm−0.02·oob（全代理盲搜,MWSCAS 式） | 30 | 37 |
| N 空間隨機 | 均勻抽樣（對照） | 20 | 37 |
| I 工具箱 | 知識分層（低成本帶優先）→SM 字典序頂帶按 oob | 30 | 218 |
生成細節：合法位置索引基因（避免無效放塊塌回錨點）;歷史已跑排除;check-dup 全過;互異門檻 Hamming>8。

## 3. 執行紀錄 (Run)
```
# 37:  python -m script.dedust run --input dedust_r15ga_input --store dedust_r15ga
# 218: python -m script.dedust run --input dedust_r15inf_input --store dedust_r15inf
```

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
- 待。

## 6. 後續決策 (Next)
- 數字進 thesis Ch.8.1（範式對比）;優勝解進名鑑（照鐵則公證）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_r15ga/`、`dedust_r15inf/`;SM v4 訓練記錄=sm_reanchor.py CLEAN_STORES v4 註記
