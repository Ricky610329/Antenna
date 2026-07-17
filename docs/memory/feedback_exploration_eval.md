---
name: feedback-exploration-eval
description: "Ricky 2026-07-13 兩修正:①擴散型探索介入用效率 over 長 baseline 評估,不做單批因果判決 ②每輪硬上限 3 批"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

Ricky 2026-07-13 兩個方法修正（Opus 曾犯「急著判根稅」的錯,被糾正）：

**① 探索型介入的評估法**：根稅(--root-cap)/de novo/novelty/資訊臂是**擴散型介入**（改錨點抽樣分布）,
**不是瞄準某指標的定向槓桿**。期待它「第 N 批打穿帶外牆」＝類別錯誤;「有沒有這因果關係」問得對＝可能沒有。
- **正法**：用探索效率評——gain-check L2 學習曲線（邊際增益 over 累積 N）介入線 vs baseline 線比斜率,
  **≥5 round 後**才判;覆蓋度（根多樣性/新穎佔比）。**不對擴散介入下 per-batch 因果判決**;
  單批破紀錄=best-so-far 追蹤點,不是介入成敗證據。定向介入(O 帶外選鍵/H 劑量掃描)才可短視窗因果判讀。

**② 每輪硬上限 3 批**：一 round 內部最多 3 批,第 3 批判讀完即 /close-round → /new-round,不拖第 4 批
（R23 拖 4 批=過厚教訓）。擴散介入的長評估跨輪累積,不靠單輪撐多批。

**Why**: 防過早判決（延伸「防過早悲觀/樂觀」gain-check 哲學）;輪=一個假設的生命週期不是資料桶。

**How to apply**: 判讀擴散臂成效時忍住不下單批結論,存資料等 L2 跨輪對比;開輪就設 ≤3 批。
正式定案=decisions「探索型介入的評估法」「每輪硬上限 3 批」;工具=gain-check skill。
相關 [[project-strategy-data-flywheel]]（軸相關枯竭）[[feedback-value-axis-oob]]。
