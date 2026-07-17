---
name: feedback_tdd_quality_bar
description: 使用者重視 TDD 與「零警告」品質標準：每修一個 bug 補一條回歸測試，測試輸出不留警告噪音
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

使用者明確要「基於 TDD 把專案寫得更好」，對「這幾次各種 bug」與測試輸出的警告噪音感到品質不夠好。

**Why**：這是個研究 codebase（學長原碼較亂），使用者把「測試綠 + 輸出乾淨」當成可信度指標；一堆 UserWarning 會讓人懷疑系統正確性。

**How to apply**：
- 每修一個 bug，**先在 `tests/` 補一條會重現該 bug 的回歸測試**，再修到綠（例：`tests/test_surrogate_training.py` 鎖形狀廣播/copy-construct 警告）。
- 把 `python -m pytest tests/ -q` 的 **warnings summary 清到零** 當收尾標準；遇到警告先判斷是「真 bug」還是「良性誤報」：
  - 值保型（形狀對齊、`torch.tensor(t)`→`detach().clone()`）放心修。
  - 行為型（排程器自適應、收斂門檻）會動 golden，**先確認再改**，或只門控警告不動數值（如 ACP 建構期 init step 的誤報）。
- 一律先 audit 既有實作再動（見 [[feedback_audit_existing_first]]），最小變更、保 golden 零漂移（見 [[feedback_prefer_simplicity]]）。
- 2026-06-19 一輪示範：清掉 Ranger 棄用 API、`train_by_datas` 形狀廣播、`tensor()` copy-construct、ACP init 誤報四類；107→112 測、golden 零漂移。
