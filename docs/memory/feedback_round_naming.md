---
name: feedback-round-naming
description: Ricky 2026-07-12:round 命名要規範——R23 起每輪接續編號、round 號貫穿所有產物命名
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

Ricky（2026-07-12,R22 期間）：「round 命名還是要規範一點,這次 R22 就算了,之後每一 R 都是接一個編號。」

**Why**: R21/R22 出現三套編號並存（round 22/批 b1/id 前綴 ?6_ 延續全域批數）＋填空池掛錯 round
（R22 時期發的池叫 r21g2）——查資料與對帳時混亂。

**How to apply**（R23 起）:
- **兩層規則（Ricky 補充定調）**：round **內部**編號自由（批 b1/b2、臂別、池號）;
  但**實驗紀錄檔**（docs/log/round-NN-*.md）嚴格守規範——一輪一檔、NN 全域嚴格遞增（單一編號宇宙,
  不另開平行線）、照 _TEMPLATE 七節、狀態流轉、README 索引同步;任何實驗產物可從其 round 檔追到。
- round 號貫穿產物命名:夾 `dedust_r23b1a`、id `m23b1_003_親`、填空池 `r23g*`、公證 `r23n*`。
- 廢除跨 round 全域批次計數（select_r22mix 的 `idn=5+batch` 是反例,R22 內沿用、R23 重寫）。
- 規範正文在 script/CLAUDE.md 批次線鐵則 §3;紀錄檔規範在 docs/log/CLAUDE.md。
