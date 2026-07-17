---
name: project_lab_real_goal
description: 本 repo 的 online 系統是 per-task「找一個最佳 pattern」；「任意 spec」=貼片設計規範(非 target response)
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

**使用者更正(2026-06-15，以此為準)**：這個 repo 的整套 online 閉迴路系統，目的是**針對固定 target 找出「一個」最佳 pattern**（per-task 優化），**不是** amortized 地學一個 `G(spec)→pattern` 對任意目標泛化。

關鍵釐清：「**任意 spec**」指的是**貼片天線的設計規範**（板材/幾何/埠等先決條件），**不是** target response（目標響應曲線）。target response 在單次優化裡是固定的、整個 run 不變 —— 這也是 [[project_zoo_latent_generator]] 把 GEN 第一層輸入改成「可學習潛在向量 z」而非「餵 target」的動機（target 當輸入沒意義，只該進 loss）。

**Why**：先前我(及更早的筆記)把它理解成 amortized 生成模型 + production ms 級推論，方向錯了。實際是 per-task：HFSS 慢且貴，online learning 邊模擬邊訓 SM/GEN，收斂到單一可製造解。

**How to apply**：規劃任何「自動產 pattern」工作時，**預設這是 per-task 優化**（找一個最佳解），不要再往「amortize over 任意 target response」想。若日後真要做泛化版 amortized G，先跟使用者明確確認那是新方向。相關 [[feedback_audit_existing_first]] [[reference_paper_terminology]]。
