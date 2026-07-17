---
name: project_benchmark_vs_random
description: 客觀 benchmark — 學習式搜尋目前輸給 random best-of-N(同 HFSS 預算)
metadata: 
  node_type: memory
  type: project
  originSessionId: 46183b31-ca6a-4a82-9ee0-98dfff4174bc
---

客觀評價演算法的方式(2026-06-26 定):**worst-margin (dB) = min(S11 餘裕, Gain 餘裕)**(雙目標都要滿足→看最差的;比 loss 直觀、有物理單位、可跨 config 比),再跟「random best-of-N(同 HFSS 預算)」+「可達天花板」比、歸一化 = (algo − random@N)/(ceiling − random@N)。

實測(harvest_single 真 HFSS 取樣 5000):天花板 worst-margin **+0.21 dB**(spec 可達但稀有 ~0.1% 樣本達標);random best-of-100 = **−1.23**、best-of-200 = **−0.74 dB**。三支 run @~200 epoch:boundary(sigmoid) **−3.96**、multiscale **−4.19**、zbatch **−6.79 dB** → **全部輸給亂猜 2.7–5.6 dB,歸一化分數為負**。pattern 演進:sigmoid 盲目漂移、multiscale 凍住(步幅 0.015)、zbatch 狂撒(步幅 0.170 卻最差)。

**Why:** 探索最多的 zbatch 結果最差 → 問題不是探索不夠,是 SM 梯度在誤導(SM 對我們 pattern 不準,optimize SM 心中的好→HFSS 現實差)。`sm_target` vs `sim_loss` 欄(已加)可驗。
**How to apply:** 任何新方法「至少要贏 random best-of-N」才有意義;用 worst-margin 客觀打分,別只看 loss。關聯 [[project_lab_real_goal]] [[project_sm_training_redesign]]。
