#!/usr/bin/env bash
set -e

# ===== ユーザー設定 =====
PROJECT_ROOT="/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/100_train_model/100_train_model.py"
NUM_GPUS=2

MASTER_PORT=29501

# ★ WANDB_API_KEY をここに書く場合（※ git管理しないこと！）
export WANDB_API_KEY="local-73f67a791cf323a6e8cd6e10844f6f50dace4076"
export WANDB_BASE_URL="https://toyota.wandb.io"    
# 必要なら WandB のプロジェクトやエンティティもここで上書き可能
export WANDB_PROJECT="offroad_rugd"
export WANDB_ENTITY="hidebu"

# ===== 実行 =====
echo "Running DDP training on $NUM_GPUS GPUs..."
echo "  Script : $SCRIPT_PATH"
echo "  Project: $PROJECT_ROOT"

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  "$SCRIPT_PATH"
