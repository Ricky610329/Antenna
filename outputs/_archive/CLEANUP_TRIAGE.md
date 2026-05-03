# Cleanup Triage — Aggressive Trim (R156-era)

This pass consolidates the decided 1-bit RIS pipeline into 2 core files
(`script/ris_core.py` + `script/ris_demo.py`) and archives every per-round
experiment script plus the loose loop summaries.

## Pass-1 verdicts

### Core (KEEP at script/ top level)
- `kill.py`, `methodology_demo.py`, `demo_ris_pairs.py` — foundational utilities
- `ris_core.py` (NEW) — distillation of R134/R140/R141/R146/R150/R154
- `ris_demo.py` (NEW) — minimal runnable example calling `ris_core`
- `report_architecture.py` — generates report_arch_*.png; figures referenced
  by REPORT, but archive script (figures already saved). DECISION: ARCHIVE
  (per "report_*.py … archive after figures are saved" rule)

### script/r1XX_*.py — ARCHIVE all → outputs/_archive/round_experiments/
Distilled into `ris_core.py` + EXPERIMENT_LESSONS.md. Files:
- r95–r99 (stress / cross-freq / timing / scaling)
- r101–r156 (every per-round experiment script)
- r156_visualize_multifreq.py (one-off visualization, summary png saved)
- r156_n71_breaks_bw_boundary.py (untracked, never finished; ARCHIVE not delete)

### script/report_fig*.py + inference_visualization*.py — ARCHIVE
- report_fig6_selector_tree.py / report_fig7_early_stop.py /
  report_fig8_surrogate_robustness.py / report_fig9_1bit_validation.py
- report_visualization.py
- inference_visualization.py / inference_visualization_1bit.py
All outputs already saved as PNG and referenced from REPORT.

### script/ — older one-off scripts → ARCHIVE
Kept in git history; not part of the decided pipeline. Move to
`outputs/_archive/round_experiments/script_legacy/`:
- active_learning_*, analyze_dataset, benchmark_*, binary_sa_finetune,
  build_dataset*, check_gpu, compare_*, continuous_vs_binary_eval,
  design_*, direct_pattern_search, evaluate_r92_partial,
  filter_dataset_by_rw, flat_top_aperture_scaling, generate_*,
  get_local_ip, het_ensemble_gradient_quality, img2video, inspect_ris_run,
  measure_gradient_quality, optimize_worst_case, pareto_frontier_n51,
  post_quantize_eval, pretrain_surrogate, process_files,
  render_*, run_*.sh, surrogate_*, sweep_*, test_determinism,
  train_*.py, verify_*, worst_case_eval

### outputs/loop_summary_round*.md (~73 files) — ARCHIVE
Move to `outputs/_archive/loop_summaries/`. Per-2-round logs no longer
needed at top level; REPORT + EXPERIMENT_LESSONS.md cover content.

### outputs/REPORT_R94_to_R156.html / .pdf — DELETE
- HTML regenerable from .md; user explicitly said skip PDF (broken CJK).

### outputs/ top-level non-loop files
KEEP (referenced by REPORT_R94_to_R156.md):
- REPORT_R94_to_R156.md, EXPERIMENT_LESSONS.md, PATCH_BRIDGE_PLAN.md
- report_arch_pipeline.png, report_arch_timeline.png
- report_fig1..fig9 (9 figures)
- r120_baseline_vs_winner.png, r122_three_recipes.png
- r156_multifreq_summary.png

ARCHIVE (older legacy figures still tracked at outputs/ top, not in REPORT):
- aperture_scaling.png, best_record_38ghz_n41.png, dataset_v1_gallery.png,
  method_comparison.png, pareto_compare_38GHz_n31.png,
  r104_n41_vs_n51_pareto.png, r81_surrogate_ranking.png,
  r85_active_learning.png, r86_ucb_vs_greedy.png, r88_mc_dropout.png,
  r93_max_max_vs_worst_case.png, r94_pareto_n51.png,
  record_progression.png
→ Move to outputs/_archive/legacy_figures/

### Untracked .pt files (r142..r146 surrogate checkpoints)
ARCHIVE → outputs/_archive/surrogate_checkpoints/ (already the convention
for r142–r146 checkpoints; user already moved them once). The 5 untracked
.pt files at outputs/ top should be moved there.

## Pass-2 execution plan
1. Write ris_core.py + ris_demo.py.
2. Smoke-test ris_demo.py with conda env `ant`.
3. git mv per above.
4. Move untracked files via plain `mv` (since git mv won't track them).
5. git rm REPORT_R94_to_R156.html and .pdf.
6. Verify nothing referenced by REPORT was archived.
7. Final commit + push.
