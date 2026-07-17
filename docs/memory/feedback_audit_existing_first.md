---
name: feedback_audit_existing_first
description: 接到任何「設計新系統」需求, 一律先 audit 既有 codebase, 不平行重建
type: feedback
originSessionId: da6ba2af-85c8-424d-8456-44fd05031698
---
接到任何「設計新 pipeline / 訓 model / 做 X」的需求前, **先 audit `antenna/` 既有 codebase 至少 30 分鐘**, 確認:

1. 既有有沒有相同/相似目標的實作
2. 如果有, 是 production ready / debug stage / abandoned?
3. 我的工作是替代它、補強它、還是完全 orthogonal?

**Why**: R94-R156 我花 156 rounds 建 per-task GD methodology, 結果發現 lab 早就有完整 amortized G(spec)→pattern + online learning pipeline 在 `antenna/training/trainer.py` (478 行)。我建的東西跟 lab 真實目標方向不同, 變成 sub-problem 工具而非主路徑。如果第一天就 audit 過 codebase, 整個方向定位會準確很多。

**How to apply**:
- 收到「想做 X」前: `Glob antenna/**/*.py` + `Grep` 關鍵詞 + 看 `antenna/conf/experiment/*.yaml` + 看 `result/` 既有 run dirs
- 找到相關 component 後**讀完原始碼再說 plan**, 不要看 init 就開始 design
- 在報告/計劃裡明確標出「我做的東西 vs 既有東西」的關係
- 對 user 講話時誠實: 「lab 已經有 X, 我這邊是補 Y」, 不要把自己工作說成全新系統
