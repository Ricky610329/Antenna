---
name: feedback_profile_on_prod_real_hfss
description: 效能/瓶頸量測要在正式機用真實 HFSS 跑；開發機 mock 估出來的數字使用者不信
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

使用者要找訓練速度瓶頸時，明確要求「**在正式機跑、量到真實 HFSS 時間**」，不要在開發機用 mock 估。

**Why**：mock 用合成響應 → SM `train_one_data` 要擬合的目標跟真實 HFSS 不同 → 內層迭代數/耗時失真；
而且 HFSS 本身才是 wall-clock 大頭，排除它的 profile 不完整、使用者不信。

**How to apply**：
- profiling 腳本要用 `build_simulator(cfg, ...)`（真實 HFSS）而非 mock，並**把 HFSS 各階段也納入計時**
  （`AntennaPattern.simulate`＝求解+讀回、`simulator.start/end/clean`＝開專案/收尾/清理）。
- 這種腳本**只在正式機（conda `patch` + HFSS）跑**；開發機（conda `ant`）不要替使用者試跑（沒 HFSS、且數字不準）。
- 交付前用 `py_compile` + import 檢查語法/相依即可，不要實跑。
- 腳本：`script/profile_training.py`（2026-06-20 改成真實 HFSS 版）。
