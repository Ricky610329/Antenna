---
name: feedback-dev-machine-load
description: "開發機=Ricky 互動用機,重掃/重算工作有節流鐵則(2026-08-06 卡機事件)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
  modified: 2026-08-06T05:27:28.005Z
---

2026-08-06:瀏覽工具資料層 agent 在曲線已全量完成後,又在開發機自行加開一輪全庫增量重掃
(NAS IO+CPU ~28 分級),把 Ricky 的電腦拖到卡頓,Ricky 說「my computer is so lag」
「避免再次卡壞這台電腦」。

**Why:** 開發機是 Ricky 的日常互動機器,不是計算節點;重工作(全庫 NAS 掃描/大量 torch.load/
全測試/多 agent 並行大 IO)會直接傷他的使用體驗。

**How to apply:**
1. **重掃/重算單例制**:build_index 之類的全庫掃描一次只准一個行程;跑之前先查沒有同類行程在跑。
2. **必要才跑**:資料已全量就不重掃;增量刷新等「真的有新資料要收」再跑(23 秒級的輕增量 OK)。
3. **降優先權**:>1 分鐘的重工作啟動後立刻 `Stop-Process` 前先想想——正確做法=啟動後
   `(Get-Process -Id <pid>).PriorityClass='BelowNormal'`,或乾脆問 Ricky 再跑。
4. **subagent brief 必寫**:「不要在開發機自行啟動全庫掃描/重算;需要時回報主線批准」。
5. 長掃描盡量排在 Ricky 不在機器前的時段(晨報前/深夜)。
相關:[[feedback_shared_machine]](共機公約=正式機版本的同款規矩)。
