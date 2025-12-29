#!/usr/bin/env bash
set -e

# ==========================================
# run_wandb_sweep.sh
#
# ✅ 推奨: Sweepは「1GPU=1agent」で並列化する
#   - GPUが2枚なら agent を2つ立てるのが最も簡単＆安定
# ==========================================

PROJECT_ROOT="/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/200_wandb_sweep/200_wandb_sweep.py"
BASE_CFG="$PROJECT_ROOT/src/scripts/conf/100_train_model/100_train_model_default.yaml"

# ===== WandB settings =====
export WANDB_API_KEY="local-73f67a791cf323a6e8cd6e10844f6f50dace4076"
export WANDB_BASE_URL="https://toyota.wandb.io"
export WANDB_PROJECT="Csiro-Image2BiomassPrediction"
export WANDB_ENTITY="hidebu"

# ===== Sweep settings =====
COUNT=30
FOLD=0
EPOCHS=100

echo "=== Create sweep & run agent ==="

# ==========================================
# ✅ 1GPUで回す（最小）
# ==========================================
python "$SCRIPT_PATH" --config "$BASE_CFG" --count "$COUNT" --fold "$FOLD" --epochs "$EPOCHS"

# ----------------------------------------------------------
# ✅ 2GPU を “並列 sweep” で使いたい場合（おすすめ）
# 1) sweep_id を作る
# SWEEP_ID=$(python "$SCRIPT_PATH" --config "$BASE_CFG" --create_only)
# echo "SWEEP_ID=$SWEEP_ID"
#
# 2) GPUごとに agent を起動（同じ sweep_id を共有）
# CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_PATH" --agent --sweep_id "$SWEEP_ID" --count 30 --fold "$FOLD" --epochs "$EPOCHS" &
# CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_PATH" --agent --sweep_id "$SWEEP_ID" --count 30 --fold "$FOLD" --epochs "$EPOCHS" &
# wait
# ----------------------------------------------------------