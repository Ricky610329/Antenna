---
name: project_senior_showcase_vs_f2
description: "學長招牌成果=論文圖4-4=t07_top(合規但含粉塵);F2=我們的對稱練習母本(非學長最好);報告框架=學長方法可合規,我們補可製造+系統化"
metadata: 
  node_type: memory
  type: project
  originSessionId: 13490732-42ae-445f-bea0-32ca5037c0cc
---

寫報告/敘事時對「學長成果 vs 我們」的正確定位（2026-07-14 校正，Ricky 指正＋比對論文 `docs/Paper.pdf`）：

- **學長的招牌好貨＝論文圖 4-4**（p65，「ACP 生成之完全合規貼片天線」）＝我們池 idx **16132** ＝ `dedust_r9_input/t07_top`。現行 HFSS 重測 **wm +0.35**（S11+0.46/Gain+0.35，帶內達標、撐過驗證），有 rad。**但含 13 個 <4px 粉塵碎片 → 不可製造**。→ 學長的方法**本身能產生完全合規的設計**，別把學長講差。
- **F2 ＝我們的對稱化練習母本**（池 idx 912＝`dedust_r7_input/p02_orig`），現行 HFSS −1.42、池值 −0.01。是 s05→w17→c21 構造式血統的起點，**不是學長的代表作**。早期報告誤把 F2 當「學長 vs 我們」的學長方，Ricky 指正。
- **oracle F0**＝池 idx 6471，池值 +0.38／現行重測 **+0.44**（撐過驗證）。學長池 18 個過標、R9 實測 8/18 撐過重測。
- **`docs/design_priors.md` 的「F2 錨點 −6.44」是誤植**：那個 **+6.2 dB 是 10-5-10 對稱化對「散亂投影家族」的救援量**（round-09 §4：F2 家族 +6.2），不是 F2 絕對值。（repo doc 尚未修，可提議修。）

**報告框架（公允版）**：學長方法可產生合規設計（圖4-4）；差別在**可製造性（我們零粉塵）＋可系統化衍生（可微擾演化）＋略高 margin**，不是「打敗爛 baseline」。對比圖＝`champ_compare --plain --new m23b4_030_r3_001 --old t07_top`。相關 [[project_w17_champion]] [[project_data_dual_track]]。
