---
name: project-lowdiag-axis
description: R53+ 主軸=低對角左側(Ricky 2026-08-04 定向);兩閘裁路徑;Ricky 出現點限四種
metadata:
  type: project
---

2026-08-04 Ricky 定向(decisions「低對角左側軸」條):R53 起主軸=**可製造左側解**
(合格∧lo≤−3.46∧diagb≤4)。路徑由兩閘裁:閘一=r52dx 補角探針(有效→手術路/無效→生成路,
負片橋接當低對角票倉);閘二=t07 帶 lo 的 S1 驗票。背景=對角=0.01mm 真導體微橋
(HFSS 像素盒重疊,diffsim 發現)+蝕刻做不出=模擬實物落差。

**Why:** lo 票倉(左側家族 diagb 13-16/學長帶 14-30)與可製造性反相關=左側戰場的
可製造化問題;王系右側天生乾淨無此題。

**How to apply:** 開輪判準從 decisions 該條繼承;Ricky 出現點限四種(紀錄推播/裁決/重啟/
收輪報告),其餘自主;**每日早上 cron 播報資料量**(Ricky 2026-08-04 指示)。
相關:[[project_strategy_data_flywheel]][[feedback_value_axis_oob]]。
