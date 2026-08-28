---
name: project-instrument-era
description: 2026-07-29=SM 儀器換代日（lr bug 修）——引用此前準度結論需注意斷點
metadata: 
  node_type: memory
  type: project
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
  modified: 2026-08-02T21:57:45.118Z
---

2026-07-29 修掉 ReduceLROnPlateau per-batch bug（lr 首 epoch 撞 1e-6 地板,訓練預算 39/40 空轉
了整個專案史）。v87 起 held-out 中位 1.20→0.63、凍結尺 1.10→0.54（-51%,固定集乾淨可比）。
獨立驗證（diffsim session 2026-08-03,analysis-10 §8,OOS 817 筆 v50-v100 十三版重量）:
lr 修在「選批」尺上值 ~3 倍——top10% 命中率 v50-85 中位 17.5% → v88-100 中位 52%
（ρ 0.567→0.823）;且 ρ↔命中率 Spearman +0.775=五軸 KPI① 是有效代理獲證。
工具=`script/sm_selection_audit.py`（機制性改動看 P(勝隨機),別只看 ρ;平手算輸,上限 ~86%）。

**Why:** 全史凍結尺 1.0-1.4 的「橫盤」就是 bug 本身——所有 2026-07-29 前的「SM 準度天花板/
資料報酬遞減/架構對比」結論都建立在壞儀器上,引用時必須標斷點。

**How to apply:** ①凍結尺跨斷點比較無意義（v86 前 vs v87 後）②影子對決/轉正判定 v87 起重洗
（two 轉正案 b3 被修好的 MLP 擋下=不轉正）③高原條件③（SM 凍結）判定作廢待新均衡重校
④gain L2 曲線等歷史學習曲線同受影響。詳見 round-47 §4/§5 與 discuss/audit-round2-2026-07-29.md。
相關:[[project_strategy_data_flywheel]]。
