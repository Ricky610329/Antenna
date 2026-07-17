---
name: reference_lab_pipeline_locations
description: Lab 既有 amortized G 訓練 pipeline 各 component 的位置 (trainer, generators, losses, configs)
type: reference
originSessionId: da6ba2af-85c8-424d-8456-44fd05031698
---
Lab 既有的 G(spec)→pattern + online learning pipeline 已實作完成（不是我建的）。重要 component 位置:

**Trainer (主迴圈, 478 行)**:
- `antenna/training/trainer.py` — 完整 online learning loop:
  每 epoch: G 生 pattern → simulator 算 real_loss → online surrogate 自 train → G 走 surrogate 反傳 → 收斂不動 rollback + 全 retrain surrogate
- 入口: `python -m antenna train +experiment=<config>`

**Generators (`antenna/models/generators/`)**:
- `biased_gumbel_sigmoid_gen.py` — `BiasedGumbelSigmoidGEN` (反 collapse 校準, 主力 binary RIS)
- `gumbel_sigmoid_gen.py` — `GumbelSigmoidGEN` (原版)
- `wide_gumbel_sigmoid_gen.py`, `biased_gumbel_sigmoid_gen.py` — 變體
- `sigmoid_gen.py` — `SigmoidGEN` (無 Gumbel)
- `sp_gen.py` — `SPGEN` (Spike-and-slab)
- `cvae.py`, `mirror_cvae.py` — `CVAE`, `MirrorCVAE` (條件 VAE 變體)
- `gradient_estimator.py` — `BinarySTE` straight-through estimator (硬二值化 forward, 梯度直通 backward)

**Losses (`antenna/losses/` + `antenna/ris/__init__.py`)**:
- `antenna/ris/__init__.py:95` — `custom_loss_tolerance` (lab 主用 loss):
  - `side_excess = (prediction[mask_side] - sidelobe_threshold).clamp(min=0).pow(2).mean()`
  - `main_deficit = (main_target - prediction[mask_main]).clamp(min=0).pow(2).mean()`
  - 雙邊 dB² penalty, asymmetric (只懲罰違反 spec, 達標就 0)
- `antenna/losses/interval.py` — `custom_loss_interval` (區間 loss, 落在 [low, high] 內 = 0)
- `antenna/losses/regularization.py` — TV loss, island suppression, SC loss, GC loss, FeedReachability
- `antenna/losses/patch_losses.py` — patch 用的 r/g/minmax/boundary loss

**Surrogate (`antenna/models/surrogates/`)**:
- `hfss_net.py` — `HFSSNet` 6-layer MLP, 預設 `(625 → (3,17))` 但可調 dim
- `surrogate_model.py` — `SurrogateModel` 訓練封裝, 支援 `train_one_data` (每筆即時) 和 `train_by_datas` (batch full retrain)
- `unet.py` — `EnhancedHFSSUNet` UNet variant

**Pretrained checkpoint**:
- `result/_pretrained_surrogate/checkpoint/sm.pth` — n=15 RIS surrogate (5000 samples, 200 epochs, final_loss=56.37 ≈ 7.5 dB RMSE)
- 注意: 這個 surrogate 在 R152 我試過拿來當 frozen surrogate 跑 per-task GD, 結果太不準 (mean abs err 6.29 dB, max 29.47 dB) optimize stuck。但 lab pipeline 是把它當 init 然後**繼續 online finetune**, 不是凍住用。

**Config (Hydra YAML)**:
- `antenna/conf/experiment/train_ris_binary_pretrained.yaml` — 主 RIS 1-bit experiment config
- `antenna/conf/experiment/train_ris_binary.yaml` — RIS 1-bit cold start
- `antenna/conf/experiment/train_ris_phase1_continuous.yaml`, `train_ris_phase2_binary.yaml` — phase1→2 transition
- `antenna/conf/experiment/train_ris_v15combo.yaml`, `train_ris_binary_v2.yaml` — 各種 ablation

**Run history**:
- `result/RIS-binary-pretrained-v1/` — 上次跑的結果 (有 ~100 generator checkpoints, log 顯示前 8 epoch real_loss 200-300)
- `result/RIS-binary-v2/`, `RIS-multi-target-v5/v6/v7/`, `RIS-direct-v8/v9-plain/`, `RIS-phase1-v3/v4/`, `RIS-phase2-v3/v4/` — 多個 ablation run dirs
- `result/_archive_continuous_phase/RIS-v3-tau-annealed/` 到 `RIS-v13-directivity/` — 早期 continuous phase 階段歸檔
- 每個 run dir 有 `config.yaml`, `*.log`, `pic/{loss_curves,pattern_evolution,response_vs_target}.png`, `checkpoint/generator_*.pth`, `online.dataset`

**已知 bug / 待 debug**:
- `antenna.losses.regularization:__call__:195 - 饋入點座標越界` (FeedReachability 在 RIS 訓練時 trigger, 可能 trainer 共用 patch code 導致)
- `online_dataset` 加得慢 (只有 real_loss < 歷史平均才加, 6 epoch 才 3 筆)

**How to apply**: 之後任何「想自動產 pattern 給 spec」需求前, 先看這個 pipeline 怎麼跑、跑得怎麼樣, 再決定是否需要新東西。**不要平行重建**。
