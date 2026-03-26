#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-inspired}"
REC_DATA_ROOT="${REC_DATA_ROOT:-${DATASET_ROOT:-rec_data}}"
CONV_DATA_ROOT="${CONV_DATA_ROOT:-conv_data}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
DEVICE="${DEVICE:-cuda}"
PHASES="${PHASES:-alignment pretrain recommendation conversation}"
LM_MODEL="${LM_MODEL:-models/DialoGPT-small}"
TEXT_MODEL="${TEXT_MODEL:-models/roberta_base}"
OUTPUT_DIR="${OUTPUT_DIR:-output/dc_mome}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-MCRS}"
WANDB_ENTITY="${WANDB_ENTITY:-longvh-research-vnu}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"

echo "DC-MoME training"
echo "  dataset      : $DATASET"
echo "  rec_data_root: $REC_DATA_ROOT"
echo "  conv_data_root: $CONV_DATA_ROOT"
echo "  batch_size   : $BATCH_SIZE"
echo "  eval_batch   : $EVAL_BATCH_SIZE"
echo "  num_epochs   : $NUM_EPOCHS"
echo "  device       : $DEVICE"
echo "  phases       : $PHASES"
echo "  lm_model     : $LM_MODEL"
echo "  text_model   : $TEXT_MODEL"
echo "  output_dir   : $OUTPUT_DIR"
echo "  use_wandb    : $USE_WANDB"
echo "  wandb_project: $WANDB_PROJECT"
echo "  wandb_entity : ${WANDB_ENTITY:-<unset>}"
echo "  wandb_name   : ${WANDB_NAME:-<unset>}"
echo "  wandb_mode   : $WANDB_MODE"

TRAIN_ARGS=(
  -m dc_mome.train
  --dataset "$DATASET"
  --rec-data-root "$REC_DATA_ROOT"
  --conv-data-root "$CONV_DATA_ROOT"
  --phases "$PHASES"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-epochs "$NUM_EPOCHS"
  --lm-model-name-or-path "$LM_MODEL"
  --text-model-name-or-path "$TEXT_MODEL"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ "$USE_WANDB" == "1" || "$USE_WANDB" == "true" || "$USE_WANDB" == "TRUE" ]]; then
  TRAIN_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE")
  if [[ -n "$WANDB_ENTITY" ]]; then
    TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_NAME" ]]; then
    TRAIN_ARGS+=(--wandb-name "$WANDB_NAME")
  fi
fi

"$PYTHON_BIN" "${TRAIN_ARGS[@]}"
