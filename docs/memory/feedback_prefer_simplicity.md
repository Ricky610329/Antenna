---
name: feedback-prefer-simplicity
description: 使用者反 over-design：偏好「單一註冊文件 + 名字指定」的 zoo 模式、管線層 default 元件，拒絕多檔分散的 registry 流程
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

使用者明確表示「這裡的代碼我覺得有很多都是 over design，都做簡化」，整體改動方向是解耦＋降複雜度。

**Why**：研究型 codebase，維護者是學生；間接層每多一層，理解成本都轉嫁給之後接手的人。曾否決我做的 registry-in-training.py（加模型要碰 4 個檔案）與「config 填數值」設計，要求改成 model zoo：一個專門文件註冊、config 用名字指定（`generator: sigmoid`）。

**How to apply**：
- 設計選項只有一個實作時不要先建抽象（YAGNI）；等第二個出現再加選擇器。
- 集中勝於分散：註冊/清單類的東西收在「一個檔案、一眼看完」（如 `antenna/zoo.py`）。
- 每個輸出都要過的步驟做成**管線層的 default 元件**，不要塞進各模型（例：STE 二值化 + ACP 的 tau 從 GEN forward 移到 run_training）。
- 提案前先問：「之後的人要碰幾個檔案？」>2 就重想。
- **型別註解＝輕量文件**（2026-06-11 確立）：簡單註解（`x: Tensor`、`-> dict`）歡迎；TypeVar/Generic/ParamSpec/@overload 一律不用（已全數清除，types.py 258→47 行）。不做靜態型別檢查，CI 只用 pyflakes 擋 undefined name。

注意：使用者說「AP」時指 **ACP (Adaptive Cyclical Policy)**，不是 AntennaPattern。

相關：[[feedback-audit-existing-first]]、[[feedback-dont-oversell]]
