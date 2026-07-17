---
name: feedback_discuss_before_loss_change
description: 改動 loss 函式前一律先跟使用者討論、取得同意再動，不可自作主張
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f50fa6ba-3dc5-4431-aef3-886d58f8e71d
---

改動任何 loss 函式（`antenna/losses.py`：`beam_coverage_loss`、`boundary_loss`、`custom_loss_minmax`、DLF 等）前，**先跟使用者討論設計與動機、取得同意再動手**，不可直接改。

**Why:** loss 直接決定訓練目標與收斂行為，改錯影響大且不易事後察覺；使用者要先看過理由。這也呼應 CLAUDE.md「SM 單筆擬合過於激進…改前先想清楚」與 [[project_sm_training_redesign.md]] 的謹慎基調。

**How to apply:** 想調 loss 時，先提出「為什麼要改 / 改成什麼 / 對收斂與 golden 的影響」讓使用者決定；獲准後才動，並比照 [[feedback_tdd_quality_bar]] 補回歸測試、保 golden 零漂移。診斷類（畫圖、排序、加診斷腳本）不算改 loss，可逕行。
