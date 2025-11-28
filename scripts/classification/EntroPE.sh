#!/bin/bash

### ================================
###  CONFIG
### ================================

export CUDA_VISIBLE_DEVICES=3
model_name="EntroPE"

entropy_model_checkpoint_dir="./entropy_model_checkpoints/"
random_seed=2025

# List of datasets used in PatchTST runs
datasets=(
  "EthanolConcentration"
  # "FaceDetection"
  # "Handwriting"
  # "Heartbeat"
  # "JapaneseVowels"
  # "PEMS-SF"
  # "SelfRegulationSCP1"
  # "SelfRegulationSCP2"
  # "SpokenArabicDigits"
  # "UWaveGestureLibrary"
)

# Hyperparameters 
dim=32
heads=4
layers=1
batch_size=32
learning_rate=0.001
train_epochs=100
patience=40
dropout=0.3
patch_length=64
cross_attn_k=1
enc_in=1

mkdir -p logs/Classification

echo "Running EntroPE on GPU $CUDA_VISIBLE_DEVICES"
echo "Total datasets: ${#datasets[@]}"

### ================================
###  LOOP
### ================================

for dataset in "${datasets[@]}"; do
    echo "===================================================="
    echo " Running EntroPE on dataset: $dataset"
    echo "===================================================="

    log_file="logs/Classification/EntroPE_${dataset}.log"

    python -u run.py \
        --task_name classification \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path ./dataset/${dataset}/ \
        --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
        --model_id $dataset \
        --model_id_name $dataset \
        --model $model_name \
        --data UEA \
        --features M \
        --quant_range 3 \
        --n_layers_local_encoder $layers \
        --n_layers_local_decoder $layers \
        --n_layers_global $layers \
        --dim_global $dim \
        --dim_local_encoder $dim \
        --dim_local_decoder $dim \
        --cross_attn_k $cross_attn_k \
        --n_heads_local_encoder $heads \
        --n_heads_local_decoder $heads \
        --n_heads_global $heads \
        --cross_attn_nheads $heads \
        --dropout $dropout \
        --max_patch_length $patch_length \
        --patching_threshold 1 \
        --patching_threshold_add 1 \
        --monotonicity 1 \
        --des "Exp" \
        --train_epochs $train_epochs \
        --patience $patience \
        --lradj cosine \
        --pct_start 0.4 \
        --itr 1 \
        --batch_size $batch_size \
        --patching_batch_size $((batch_size * enc_in)) \
        --learning_rate $learning_rate \
        > "$log_file" 2>&1

    echo "Completed: $dataset"
done

echo "===================================================="
echo " All EntroPE UEA dataset runs completed!"
echo "===================================================="
