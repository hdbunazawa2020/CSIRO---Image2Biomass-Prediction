#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction"
SCRIPT_PATH="$PROJECT_ROOT/src/scripts/203_wandb_splitcropauxregressor/203_wandb_splitcropauxregressor.py"
BASE_CFG="$PROJECT_ROOT/src/scripts/conf/203_wandb_splitcropauxregressor/203_wandb_splitcropauxregressor_default.yaml"

COUNT=200

export WANDB_API_KEY="local-73f67a791cf323a6e8cd6e10844f6f50dace4076"
export WANDB_BASE_URL="https://toyota.wandb.io"
export WANDB_PROJECT="Csiro-Image2BiomassPrediction-Sweep"
export WANDB_ENTITY="hidebu"
export WANDB_AGENT_DISABLE_FLAPPING=true

[[ -f "$SCRIPT_PATH" ]] || { echo "[ERROR] SCRIPT_PATH not found: $SCRIPT_PATH"; exit 1; }
[[ -f "$BASE_CFG" ]] || { echo "[ERROR] BASE_CFG not found: $BASE_CFG"; exit 1; }

echo "[INFO] create sweep..."
SWEEP_ID="$(python "$SCRIPT_PATH" \
  --action create \
  --base_cfg "$BASE_CFG" \
  --project "$WANDB_PROJECT" \
  --entity "$WANDB_ENTITY" | tail -n 1 | tr -d '\r')"

echo "[INFO] sweep_id = $SWEEP_ID"
echo "[INFO] run 2 agents (gpu0 & gpu1)... count=$COUNT each"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_PATH" --action agent --sweep_id "$SWEEP_ID" --base_cfg "$BASE_CFG" --count "$COUNT" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_PATH" --action agent --sweep_id "$SWEEP_ID" --base_cfg "$BASE_CFG" --count "$COUNT" &
PID1=$!

wait $PID0
wait $PID1
echo "[INFO] agents finished"

# #!/usr/bin/env bash
# set -euo pipefail

# PROJECT_ROOT="/mnt/nfs/home/hidebu/study/CSIRO---Image2Biomass-Prediction"
# # SCRIPT_PATH="$PROJECT_ROOT/src/scripts/200_wandb_sweep/200_wandb_sweep.py"
# # BASE_CFG="$PROJECT_ROOT/src/scripts/conf/200_wandb_sweep/200_wandb_sweep_default.yaml"
# # SCRIPT_PATH="$PROJECT_ROOT/src/scripts/201_wandb_bmh_sweep/201_wandb_bmh_sweep.py"
# # BASE_CFG="$PROJECT_ROOT/src/scripts/conf/201_wandb_bmh_sweep/201_wandb_bmh_sweep_default.yaml"
# SCRIPT_PATH="$PROJECT_ROOT/src/scripts/203_wandb_splitcropauxregressor/203_wandb_splitcropauxregressor.py"
# BASE_CFG="$PROJECT_ROOT/src/scripts/conf/203_wandb_splitcropauxregressorp/203_wandb_splitcropauxregressor_default.yaml"

# # sweep回数（1 agent が何 trial 回すか）
# COUNT=60

# # ===== WandB 設定 =====
# export WANDB_API_KEY="local-73f67a791cf323a6e8cd6e10844f6f50dace4076"
# export WANDB_BASE_URL="https://toyota.wandb.io"
# export WANDB_PROJECT="Csiro-Image2BiomassPrediction-Sweep"
# export WANDB_ENTITY="hidebu"
# export WANDB_AGENT_DISABLE_FLAPPING=true

# # （任意）使うGPUを固定したい場合
# # export CUDA_VISIBLE_DEVICES=0

# echo "[INFO] create sweep..."
# SWEEP_ID=$(python "$SCRIPT_PATH" \
#   --action create \
#   --base_cfg "$BASE_CFG" \
#   --project "$WANDB_PROJECT" \
#   --entity "$WANDB_ENTITY" | tail -n 1)

# echo "[INFO] sweep_id = $SWEEP_ID"

# echo "[INFO] run agent... count=$COUNT"
# python "$SCRIPT_PATH" \
#   --action agent \
#   --sweep_id "$SWEEP_ID" \
#   --base_cfg "$BASE_CFG" \
#   --count "$COUNT"