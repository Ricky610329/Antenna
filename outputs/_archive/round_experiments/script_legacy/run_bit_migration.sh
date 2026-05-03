#!/usr/bin/env bash
# 位元遷移端到端 — phase 1 連續 → phase 2 binary → inspect 評估報告
#
# 使用方式：
#     bash script/run_bit_migration.sh                          # 用預設名稱
#     bash script/run_bit_migration.sh my-run-tag               # 自訂前綴
#
# 預設輸出：
#     result/RIS-phase1-{tag}/      # 連續相位階段
#     result/RIS-phase2-{tag}/      # binary 階段（載入 phase1 權重）
#     result/RIS-phase2-{tag}/pic/samples/summary.md  ← 最終 binary 評估報告

set -e
TAG="${1:-v3}"
PHASE1_NAME="RIS-phase1-${TAG}"
PHASE2_NAME="RIS-phase2-${TAG}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ant

echo "========================================"
echo " Bit migration end-to-end run: tag=${TAG}"
echo "========================================"

# Phase 1 — 連續相位
echo ">>> Phase 1: continuous phase training (60 epochs)"
python -m antenna train \
  +experiment=train_ris_phase1_continuous \
  environment.device=cuda:0 \
  environment.rootdir=. \
  experiment_name="${PHASE1_NAME}" \
  2>&1 | tail -200

# Phase 2 — binary，載入 phase 1 權重
echo ">>> Phase 2: binary training (50 epochs, migrated from phase 1)"
python -m antenna train \
  +experiment=train_ris_phase2_binary \
  environment.device=cuda:0 \
  environment.rootdir=. \
  experiment_name="${PHASE2_NAME}" \
  generator.pretrained_path="result/${PHASE1_NAME}" \
  2>&1 | tail -200

# Inspect — 出 binary 評估報告
echo ">>> Inspect: binary evaluation report"
python script/inspect_ris_run.py "result/${PHASE2_NAME}" 2>&1 | tail -20

echo ""
echo "========================================"
echo "Done. Final binary evaluation:"
echo "    result/${PHASE2_NAME}/pic/samples/summary.md"
echo "========================================"
cat "result/${PHASE2_NAME}/pic/samples/summary.md"
