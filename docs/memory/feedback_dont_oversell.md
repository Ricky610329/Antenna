---
name: feedback_dont_oversell
description: 不要把工具當 production system, 不要把 sub-problem methodology 講成 main pipeline
type: feedback
originSessionId: da6ba2af-85c8-424d-8456-44fd05031698
---
對 user 報告工作時嚴格區分:
- **「這套東西本身能做什麼 / 不能做什麼」** vs
- **「這套東西在 lab 整體目標下的位置」**

**Why**: R94-R156 我寫過多份「完整報告」「universal recipe」「ready-to-use deployment」這種 framing, 但實際是 per-task GD optimization tool, 對 lab 想要的 amortized inference 場景不能直接用。User 沒指正之前我一直自我吹捧成 "deployment-ready methodology"。

具體錯誤 framing examples:
- ❌ "R141 deployment API 6/6 PASS, ready to ship" — 實際只能 per-spec 跑 30s, lab production 需要 ms 級 G(spec) 一發
- ❌ "Patch transition methodology 完整 codified" — 實際我只 codify 了 sub-problem
- ❌ "Loss design 是 framework-agnostic, transfer 順利" — loss design 確實 transfer, 但 optimizer architecture 不能照搬

**How to apply**:
- 報告 / 文件開頭一律放 "Scope and limitations" / "What this is NOT" 一節
- 寫 "deployment", "production", "universal" 這類字眼前先停下來: 真的是 production-ready 嗎? 還是 only research-stage?
- 跟 user 報告時主動講 "這個東西在 lab pipeline 的什麼位置", 不要只講 standalone metric
- 如果 user 之前嘗試過 X (e.g. GAN), 不要默默繞過, 先問清楚 X 失敗的真實原因再決定是否要走完全不同方向
