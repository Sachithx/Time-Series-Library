#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p ./logs/AnomalyDetection

########################
# Common hyperparameters
########################
model_name=EntroPE
seq_len=100
pred_len=0          # anomaly detection (reconstruction)
batch_size=256
train_epochs=5
anomaly_ratio=1
random_seed=1025

# EntroPE-specific hyperparameters (same for all datasets; edit here if needed)
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
vocab_size=256
quant_range=3

dim=8
multiple_of=256
heads=1
layers=1
dropout=0.1
max_patch_length=48
patching_threshold=0.25
patching_threshold_add=0.15
pct_start=0.5
patience=5
cross_attn_k=1
attn_window=96
learning_rate=0.001

########################
# Dataset-specific enc_in
########################
# From your table:
# SMD  -> 38
# MSL  -> 55
# SMAP -> 25
# SWaT -> 51
# PSM  -> 25

datasets=("SMD" "MSL" "SMAP" "SWaT" "PSM")

declare -A enc_in_map
enc_in_map=( \
  ["SMD"]=38 \
  ["MSL"]=55 \
  ["SMAP"]=25 \
  ["SWaT"]=51 \
  ["PSM"]=25 \
)

########################
# Run experiments
########################
for dataset in "${datasets[@]}"; do
  enc_in=${enc_in_map[$dataset]}

  echo "===================================================="
  echo "Running EntroPE anomaly detection on $dataset (enc_in=$enc_in)"
  echo "===================================================="

  log_file="logs/AnomalyDetection/${model_name}_${dataset}_${seq_len}_${pred_len}_seed${random_seed}.log"

  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --random_seed $random_seed \
    --root_path ./dataset/${dataset} \
    --entropy_model_checkpoint_dir "$entropy_model_checkpoint_dir" \
    --data_path $dataset \
    --model_id ${dataset}_${seq_len}_${pred_len} \
    --model_id_name $dataset \
    --model $model_name \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --c_out $enc_in \
    --anomaly_ratio $anomaly_ratio \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --vocab_size $vocab_size \
    --quant_range $quant_range \
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
    --cross_attn_window_encoder $attn_window \
    --cross_attn_window_decoder $attn_window \
    --local_attention_window_len $attn_window \
    --dropout $dropout \
    --multiple_of $multiple_of \
    --max_patch_length $max_patch_length \
    --patching_threshold $patching_threshold \
    --patching_threshold_add $patching_threshold_add \
    --monotonicity 1 \
    --des 'Exp' \
    --patience $patience \
    --lradj 'cosine' \
    --pct_start $pct_start \
    --itr 1 \
    --patching_batch_size $((batch_size * enc_in)) \
    --learning_rate $learning_rate \
    > "$log_file" 2>&1

  echo "Finished $dataset (logs: $log_file)"
done

echo "All EntroPE anomaly detection experiments completed!"
