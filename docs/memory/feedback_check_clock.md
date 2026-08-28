---
name: feedback-check-clock
description: 寫任何時間戳記（round 檔/scratch/decisions/commit 訊息）前先跑 date 查當下時間
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

寫時間戳記前一律先 `date` 查實際時間，不要沿用對話中記得的日期。

**Why:** 2026-07-10 Ricky 指正——session 跨午夜時我把 07-10 的事戳成 07-09（round-19/scratch 多處），
研究日誌是 append-only 時間軸，日期錯了會誤導回溯。

**How to apply:** 每次要寫日期的動作（開 round 檔、scratch/decisions 條目、ONGOING 更新）前
先 `date "+%Y-%m-%d %H:%M"`；跨午夜的長 session 特別小心。相關 [[project_research_log]]。

**★ 2026-08-02 再犯一次（比跨午夜更隱蔽的版本）**：長 session 裡我**憑「感覺過了很久」推算時間**，
把 20:12–22:30 的工作全戳成「2026-08-03 01:50」「一夜完成」，寫進 analysis-08／log README／
ONGOING／scratch 共 12 處，事後靠 `git log --date=format:'%H:%M'` 才重建真實時間線。
→ 教訓補強：**不只跨午夜要查，任何超過一小時的 session 在寫時間戳前都要重查一次**；
文件裡不要寫「一夜/整晚/花了 X 小時」這種主觀時長，要寫就從 commit 時間戳取。
相關 [[feedback_shared_machine]]。
