---
name: project-config-driven-validated
description: config 驅動訓練路徑 (train.py + configs/*.yaml) 已在正式機 (真 HFSS) 驗證可跑
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

2026-06-10：使用者在正式機 (`C:\Users\user\Documents\Anntena\Antenna`, conda env `patch`) 測試 config 驅動的 `python train.py configs/*.yaml` 路徑，確認可正常運作（真 HFSS COM + NAS 掛載）。對應 commit `043e647` (GAN/main)。

**意義**：mock/golden 驗證之外，production 整鏈已通 → 可以放心在此基礎上疊新功能（TensorBoard 監控、新 generator type 等）。

**正式機 git 注意**：該機曾有 `train_single.py` 本機改動擋 pull，已用 `git checkout --` 捨棄後更新。

相關：[[project-lab-real-goal]]、[[reference-lab-pipeline-locations]]
