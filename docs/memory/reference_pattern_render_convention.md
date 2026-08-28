---
name: reference-pattern-render-convention
description: "pattern 渲染方向鐵則:饋線邊=圖下緣(imshow origin upper);自畫 Rectangle 要轉座標,2026-08-06 Ricky 抓到轉 90° bug"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
  modified: 2026-08-05T16:52:42.676Z
---

**Pattern 圖方向鐵則**(2026-08-06,Ricky 抓到我一系列菱形圖轉了 90°):

- HFSS 幾何:`pixel_matrix[x][y]` → X=x·0.2mm、Y=y·0.2mm;**Lumped Port 在 x=27.5mm**
  (22.5mm 饋線,接貼片 X=5 邊=矩陣第一索引 i=24 那排;出處 diffsim.md §行49/93 SAB 解析)。
- **repo/論文慣例=`imshow(mat, origin="upper")`**:i 朝下、j 朝右 → **饋線邊在圖下緣**。
  全部既有 figs(r9 圖庫/champ_compare/report_*)都這樣。
- 自畫 Rectangle/Polygon 時必須轉換:像素 `(i,j)→左下角 (j*P, (24−i)*P)`;
  HFSS mm 座標點 `(cx,cy)→(cy, 5−cx)`(菱形站點/特寫視窗/比例尺都要)。
- 直畫 `(i*P, j*P)` = 相對慣例轉置,饋線跑到右側——**看圖自檢:下緣應是較連續的銅排(饋線接入邊)**。

相關:[[project_diffsim_status]]
- **橋位疊加鐵則(2026-08-13 Ricky 抓包)**:渲染菱形橋位置**必須直接呼叫
  `diag_bridge_sites`**(與 HFSS 同一份函式),嚴禁在畫圖腳本裡重新實作接點偵測——
  當日自寫版 ↙ 接點欄座標 off-by-one,23 顆菱形畫錯位(僅圖示錯;幾何與 wm_mfg
  數字無誤,fill/declean 用的是另一套自帶自證的正確邏輯)。同 align_curve 教訓:
  兩把尺會漂。imshow 座標換算:角點 (cx,cy)mm → scatter(cy/pmm−0.5, cx/pmm−0.5)。
