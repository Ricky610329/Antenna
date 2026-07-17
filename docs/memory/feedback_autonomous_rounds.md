---
name: feedback-autonomous-rounds
description: "Ricky 授權自主續輪——round 收檔後主動開下一輪（R22,23,24,…）,宣告制不等核准"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

2026-07-12 Ricky：「那你會自動跑接下來的 round 嗎? 應該說我預期你會做這件事。除了 22 也許 23 24 25…」

**Why**: 資料工廠＋批次假設迴圈已成熟到判準可以寫死、決策點可以條文化——人只需要在方向級介入,
輪與輪之間的接棒等核准反而浪費機器時間（機器不空轉是 Ricky 的一貫要求）。

**How to apply**:
- round 收檔（/close-round）後,依證據與 ONGOING 🔜 候選**主動**開下一輪：開 round 檔（判準寫死）→
  select → check-dup → jobs-add,並在對話裡**宣告**新輪的假設與判準（Ricky 可隨時否決,沒否決就跑）。
- 護欄不變：紀錄級公證鐵則、check-dup 必跑、/gain-check 期望管理;
  判準寫死的「回報討論」節點與未預料異常仍停下等人。
- 新 session 接手：讀 ONGOING＋最新 round 檔＋[[project-research-log]],從 NAS 真相（jobs-ls/status）繼續,
  不重問方向。正式定案文字在 docs/discuss/decisions.md「自主續輪授權」。
- **擴權（同日）**：可自行發明新實驗臂——新臂協議＝①假設指向具體證據 ②先導配額 ≤15 筆
  ③存活判準隨臂寫死 ④宣告制。
- **漸進式成長條款（同日,Ricky「前期很爛沒關係」）**：分布外探索臂不用 6% 生死線——
  學費預算制（固定 N 批投資,KPI=進步趨勢,三標=畢業非存活,預算盡=回報裁決不自動處決）。
  首例=R23 D 臂（de novo 12/批×5,sm_harvest→sm_denovo）。
