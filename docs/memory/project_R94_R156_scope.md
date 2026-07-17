---
name: project_R94_R156_scope
description: R94-R156 是 per-task GD methodology 研究, 互補性 sub-problem; 不是 lab 主路徑的替代
type: project
originSessionId: da6ba2af-85c8-424d-8456-44fd05031698
---
R94-R156 (約 156 個 round, 期間 2026-04 到 2026-05) 我建的東西的真實 scope:

**做了什麼**:
- Per-task gradient descent 優化單一 binary RIS pattern 的工具鏈
- 核心: `script/ris_core.py` 提供 `optimize_ris_1bit(n, inc, freq, width)` API
- 內含: recipe selector (R134/R135), worst+ripple+mean loss (R94/R119), joint early-stop (R140), warm-start surrogate (R146/R147), multi-freq sum loss (R154)
- 完整報告: `outputs/REPORT_R94_to_R156.md`, 蒸餾: `outputs/EXPERIMENT_LESSONS.md`

**做了什麼不等於什麼**:
- ✗ 不是 amortized G(spec)→pattern 模型 (lab 在 `antenna/training/trainer.py` 已有)
- ✗ 不能 ms 級 inference (per-task 要 30s-5min)
- ✗ 沒解 inverse mapping ill-posedness, 沒做 generator architecture comparison, 沒做 binary discretization GAN tricks, 沒做 active learning acquisition
- ✗ 我用 RIS analytical sim 做 per-task GD, 但 patch 沒 closed-form simulator; 直接 transfer 不可行
- ⚠️ R150-R156 我自以為在做 "patch transition methodology", 實際是 sub-problem methodology research, 對 lab 真實 production 路徑幫助有限

**對 lab 真實貢獻 (補強, 不替代)**:
| Slot | 我的 component |
|------|--------------|
| Loss term in `custom_loss_tolerance` | 加 `mean(side)` area-penalty term (R119) |
| Pretraining data for G | per-task GD 跑 N 個 spec 出 (spec, gold_pattern) pairs |
| Validation oracle | G(spec) 的品質可以對比 `optimize_ris_1bit(spec)` |
| Surrogate noise robustness evidence | R148/R149 結論可指導 surrogate retrain frequency |
| Multi-band loss design | R154 sum-across-freq 可給 broadband G 用 |

**How to apply**: 之後 user 提 patch 相關時 — **不要重提 R94-R156 當主答案**。先看 lab pipeline (memory: `reference_lab_pipeline_locations.md`) 怎麼跑, 再從 R94-R156 工具箱挑能 plug-in 的 component。
