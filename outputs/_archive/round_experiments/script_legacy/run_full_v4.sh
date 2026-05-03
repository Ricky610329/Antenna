#!/usr/bin/env bash
# v4 — 完整端到端：結構化 surrogate + 位元遷移
#
# 序列：
#   1. pretrain surrogate（5000 random + 1000 structured）
#   2. phase 1 連續相位
#   3. phase 2 binary（位元遷移）
#   4. inspect 出 binary 評估報告
#
# 用法：
#     bash script/run_full_v4.sh           # 預設 tag=v4
#     bash script/run_full_v4.sh v5        # 自訂 tag

set -e
TAG="${1:-v4}"
PHASE1_NAME="RIS-phase1-${TAG}"
PHASE2_NAME="RIS-phase2-${TAG}"
SURR_DIR="result/_pretrained_surrogate_${TAG}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ant

echo "========================================"
echo " v4 full pipeline: tag=${TAG}"
echo "========================================"

# 1. 重訓 surrogate（混入結構化 pattern）
echo ">>> Step 1/4: pretrain surrogate (5000 random + 1000 structured)"
python script/pretrain_surrogate.py \
  --element_num 15 \
  --n_samples 5000 \
  --n_structured 1000 \
  --epochs 200 \
  --batch_size 64 \
  --device cuda:0 \
  --out_dir "${SURR_DIR}" \
  2>&1 | tail -50

# 2. Phase 1 — 連續相位
echo ">>> Step 2/4: phase 1 continuous (60 epochs)"
python -m antenna train \
  +experiment=train_ris_phase1_continuous \
  environment.device=cuda:0 \
  environment.rootdir=. \
  experiment_name="${PHASE1_NAME}" \
  surrogate.pretrained_path="${SURR_DIR}" \
  2>&1 | tail -100

# 3. Phase 2 — binary（載入 phase 1 權重 + 結構化 surrogate）
echo ">>> Step 3/4: phase 2 binary (50 epochs, migrated)"
python -m antenna train \
  +experiment=train_ris_phase2_binary \
  environment.device=cuda:0 \
  environment.rootdir=. \
  experiment_name="${PHASE2_NAME}" \
  generator.pretrained_path="result/${PHASE1_NAME}" \
  surrogate.pretrained_path="${SURR_DIR}" \
  2>&1 | tail -100

# 4. Inspect — binary 評估報告
echo ">>> Step 4/4: inspect (binary evaluation)"
python script/inspect_ris_run.py "result/${PHASE2_NAME}" 2>&1 | tail -10

echo ""
echo "========================================"
echo "Done. Final summary:"
echo "    result/${PHASE2_NAME}/pic/samples/summary.md"
echo "========================================"
cat "result/${PHASE2_NAME}/pic/samples/summary.md"
