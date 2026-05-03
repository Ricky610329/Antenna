# Archive — superseded artifacts (post R94→R156 cleanup)

> 2026-05-03 cleanup pass. 156 rounds 累積的歷史檔案集中於此，方便查閱演進但
> 不再是 active references。

## 結構

```
_archive/
├── superseded_reports/   # 被 outputs/REPORT_R94_to_R156.md 取代的早期報告
│   ├── FINAL_REPORT.md            # R1-R97 paper-style 整理（pre-R94 起點）
│   ├── RESEARCH_REPORT_CN.md      # R57 free-phase 突破期完整報告
│   ├── REPORT_R121_universal_recipe.md  # R121 universal recipe 中段報告
│   ├── RIS_RESEARCH_REPORT.md     # R1-R57 早期 /loop 完整研究紀錄（移自 script/）
│   ├── RIS_DESIGN_GUIDE.md        # R1-R10 早期工作流程指引（移自 script/）
│   └── PATCH_METHODOLOGY.md       # R1-R75 distilled rules（移自 script/）
└── surrogate_checkpoints/  # Phase 2 surrogate 嘗試的中間 weights
    ├── r142_surrogate_n31.pt        # 標準 CNN, R²≈0（stuck on mean）
    ├── r143_physics_surrogate_n31.pt # Physics-aware random data, R²=-0.74
    ├── r144_trajectory_surrogate.pt  # Trajectory data, R²=-3.21
    ├── r145_warmstart_surrogate.pt   # Warm-start with bug, R²=-0.97
    └── r146_warmstart_fixed.pt       # Warm-start fixed, R²=1.000000 ★
```

## 為什麼留著

- **superseded_reports**：演進證據；未來如有 reviewer 想追溯 R57 max-max 失敗
  → R94 worst-case loss → R121 mean(side) 三段式損失的方法論演進，這些是原始紀錄。
- **surrogate_checkpoints**：R142-R146 是 Phase 2 的 4 連續 negative + R146 turning point。
  R146 是 untrained warm-start (R²=1.0)，可重現 R146/R147 的 surrogate-loop 對比結果。

## 不是 active reference 的東西

正式 deployment / 教學文件請看：

- **`outputs/REPORT_R94_to_R156.md`** — canonical 完整報告（R94→R156）
- **`outputs/EXPERIMENT_LESSONS.md`** — 156 rounds 的方法論心得（10 分鐘可讀完）
- **`outputs/PATCH_BRIDGE_PLAN.md`** — R151 patch transition 計畫（still valid）
- **`outputs/loop_summary_round*.md`** — 每 1-2 round 的 detailed logs

## 重新啟用

如要把 archived 報告重新作為 reference，`git mv` 回原位即可（git 保留了完整歷史）。
