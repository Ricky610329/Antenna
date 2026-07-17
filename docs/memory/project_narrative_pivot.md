---
name: project-narrative-pivot
description: "2026-07-08 敘事定調：線上學習=工具箱一員,agent+human-in-the-loop 共同優化;文獻背書在 docs/reference/"
metadata: 
  node_type: memory
  type: project
  originSessionId: 97b7781c-8c2e-4de9-81ed-48dbb416c186
---

2026-07-08 Ricky 定調：專案敘事從「單一優化的線上學習」轉為「線上學習是**工具**,由 agent 與 human-in-the-loop 一起優化」。是 2026-07-03「研究定位」（線上學習=方法論中的局部開採階段）的延伸——補上「誰調度階段」＝agent+人。

文獻背書：`docs/reference/` 三篇 Sengupta 組論文（索引在該夾 README.md）。錨點=2026 SSC Magazine（Human Inputs 終局圖、L1–L5 光譜、"AI is a collaborator"）；ISSCC 2026 13.2（一對多→人的取捨→可控性是採用前提）；2024 MWSCAS 當全自動對照組。

**Novelty 邊界**：文獻的 human-in-the-loop=事前約束+事後挑選、agent=RL agent；「agent+human 迭代迴圈調度工具」無現成先例=可 claim。表述紀律：說「線上學習正確角色是局部開採」,不說「沒用」。

行動候選（scratch 2026-07-08 塊）：雙保真 warm-start 配方（harvest 粗層+dedust 細層預訓 SM）、組件級 GA baseline、dedust 對外表述為 canvassing。不借:RL/pixel 級逆設計（需數十萬樣本離線攤提）。

相關：[[project_lab_real_goal]]、[[project_litreview_direction]]、[[project_generator_hyperfeature_pivot]]
