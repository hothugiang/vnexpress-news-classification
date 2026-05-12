#!/bin/bash
# Compare MSCRS vs DCMoME finetune rec on INSPIRED across 3 seeds.
# Requires pretrained models in output/compare/ (run compare_pretrain.sh first).
# Usage: bash scripts/compare_rec.sh [SEEDS...]
# Example: bash scripts/compare_rec.sh 1 2 3

set -e
cd "$(dirname "$0")/.."

DCMOME="dc_mome/rec/src"
MSCRS="MSCRS-main_old/MSCRS-main/rec/src"
DATA_DIR="rec_data"
MODEL_DIR="models"
OUTPUT_BASE="output/compare"
LR=1e-5
EPOCHS=10
EFF_BATCH=8
GRAD_ACCUM=8

SEEDS=("${@:-1 2 3}")
if [ $# -eq 0 ]; then
    SEEDS=(1 2 3)
fi

mkdir -p "$OUTPUT_BASE"

for SEED in "${SEEDS[@]}"; do
    # echo "============================================"
    # echo "MSCRS finetune rec  seed=$SEED  lr=$LR  epochs=$EPOCHS"
    # echo "============================================"
    # accelerate launch $MSCRS/train_rec_inspired.py \
    #   --dataset_dir $DATA_DIR \
    #   --dataset inspired \
    #   --tokenizer $MODEL_DIR/DialoGPT-small \
    #   --model $MODEL_DIR/DialoGPT-small \
    #   --text_tokenizer $MODEL_DIR/roberta_base \
    #   --text_encoder $MODEL_DIR/roberta_base \
    #   --prompt_encoder $OUTPUT_BASE/mscrs-pretrain-LR7e-4-seed${SEED}/best \
    #   --n_prefix_rec 10 \
    #   --num_train_epochs $EPOCHS \
    #   --per_device_train_batch_size $EFF_BATCH \
    #   --per_device_eval_batch_size 64 \
    #   --gradient_accumulation_steps $GRAD_ACCUM \
    #   --learning_rate $LR \
    #   --weight_decay 0 \
    #   --num_warmup_steps 33 \
    #   --seed $SEED \
    #   --output_dir $OUTPUT_BASE/mscrs-rec-LR${LR}-seed${SEED}

    echo ""
    echo "============================================"
    echo "DCMoME finetune rec  seed=$SEED  lr=$LR  epochs=$EPOCHS"
    echo "============================================"
    accelerate launch $DCMOME/train_rec_inspired.py \
      --dataset_dir $DATA_DIR \
      --dataset inspired \
      --tokenizer $MODEL_DIR/DialoGPT-small \
      --model $MODEL_DIR/DialoGPT-small \
      --text_tokenizer $MODEL_DIR/roberta_base \
      --text_encoder $MODEL_DIR/roberta_base \
      --prompt_encoder $OUTPUT_BASE/dcmome-pretrain-LR7e-4-seed${SEED}/best \
      --n_prefix_rec 10 \
      --num_train_epochs $EPOCHS \
      --per_device_train_batch_size $EFF_BATCH \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --weight_decay 0 \
      --num_warmup_steps 33 \
      --seed $SEED \
      --output_dir $OUTPUT_BASE/dcmome-rec-LR${LR}-seed${SEED}

    echo ""
done

echo "============================================"
echo "All runs done. Generating report..."
echo "============================================"
python scripts/report_pretrain.py --output_base "$OUTPUT_BASE"
