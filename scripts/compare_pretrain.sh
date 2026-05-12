#!/bin/bash
# Compare MSCRS vs DCMoME pretrain on INSPIRED across 3 seeds.
# Same hyperparams for fair comparison.
# Usage: bash scripts/compare_pretrain.sh [SEEDS...]
# Example: bash scripts/compare_pretrain.sh 1 2 3

set -e
cd "$(dirname "$0")/.."

DCMOME="dc_mome/rec/src"
MSCRS="MSCRS-main_old/MSCRS-main/rec/src"
DATA_DIR="rec_data"
MODEL_DIR="models"
OUTPUT_BASE="output/compare"
LR=7e-4
EPOCHS=10
EFF_BATCH=8
GRAD_ACCUM=8

SEEDS=("${@:-1 2 3}")
# If no args, default to 3 seeds
if [ $# -eq 0 ]; then
    SEEDS=(1 2 3)
fi

mkdir -p "$OUTPUT_BASE"

for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "MSCRS pretrain  seed=$SEED  lr=$LR  epochs=$EPOCHS"
    echo "============================================"
    accelerate launch $MSCRS/train_pre_inspired.py \
      --dataset_dir $DATA_DIR \
      --dataset inspired \
      --tokenizer $MODEL_DIR/DialoGPT-small \
      --model $MODEL_DIR/DialoGPT-small \
      --text_tokenizer $MODEL_DIR/roberta_base \
      --text_encoder $MODEL_DIR/roberta_base \
      --num_train_epochs $EPOCHS \
      --per_device_train_batch_size $EFF_BATCH \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --num_warmup_steps 168 \
      --max_length 200 \
      --prompt_max_length 200 \
      --entity_max_length 32 \
      --seed $SEED \
      --output_dir $OUTPUT_BASE/mscrs-pretrain-LR${LR}-seed${SEED}

    echo ""
    echo "============================================"
    echo "DCMoME pretrain  seed=$SEED  lr=$LR  epochs=$EPOCHS"
    echo "============================================"
    accelerate launch $DCMOME/train_pre_inspired.py \
      --dataset_dir $DATA_DIR \
      --dataset inspired \
      --tokenizer $MODEL_DIR/DialoGPT-small \
      --model $MODEL_DIR/DialoGPT-small \
      --text_tokenizer $MODEL_DIR/roberta_base \
      --text_encoder $MODEL_DIR/roberta_base \
      --num_train_epochs $EPOCHS \
      --per_device_train_batch_size $EFF_BATCH \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --num_warmup_steps 168 \
      --max_length 200 \
      --prompt_max_length 200 \
      --entity_max_length 32 \
      --seed $SEED \
      --output_dir $OUTPUT_BASE/dcmome-pretrain-LR${LR}-seed${SEED}

    echo ""
done

echo "============================================"
echo "All runs done. Generating report..."
echo "============================================"
python scripts/report_pretrain.py --output_base "$OUTPUT_BASE"
