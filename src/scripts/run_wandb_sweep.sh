#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/200_wandb_sweep/200_wandb_sweep.py"
BASE_CFG="$PROJECT_ROOT/src/scripts/conf/100_train_model/100_train_model_default.yaml"

# sweep回数（1 agent が何 trial 回すか）
COUNT=60

# ===== WandB 設定 =====
export WANDB_API_KEY="local-73f67a791cf323a6e8cd6e10844f6f50dace4076"
export WANDB_BASE_URL="https://toyota.wandb.io"
export WANDB_PROJECT="Csiro-Image2BiomassPrediction"
export WANDB_ENTITY="hidebu"

# （任意）使うGPUを固定したい場合
# export CUDA_VISIBLE_DEVICES=0

echo "[INFO] create sweep..."
SWEEP_ID=$(python "$SCRIPT_PATH" \
  --action create \
  --base_cfg "$BASE_CFG" \
  --project "$WANDB_PROJECT" \
  --entity "$WANDB_ENTITY" | tail -n 1)

echo "[INFO] sweep_id = $SWEEP_ID"

echo "[INFO] run agent... count=$COUNT"
python "$SCRIPT_PATH" \
  --action agent \
  --sweep_id "$SWEEP_ID" \
  --base_cfg "$BASE_CFG" \
  --count "$COUNT"