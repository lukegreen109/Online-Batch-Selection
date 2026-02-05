#!/bin/bash

# Usage: ./run_ablation.sh <seed> <wandb_not_upload>
# Example: ./run_ablation.sh 16 True

SEED=$1
WANDB_NOT_UPLOAD=$2

CONFIG_DIR="configs"

METHOD_CONFIGS=("$CONFIG_DIR/method/"*)
MODEL_CONFIGS=("$CONFIG_DIR/model/"*)
OPTIM_CONFIGS=("$CONFIG_DIR/optim/"*)
DATA_CONFIGS=("$CONFIG_DIR/data/"*)

for data in "${DATA_CONFIGS[@]}"; do
  for model in "${MODEL_CONFIGS[@]}"; do
    for optim in "${OPTIM_CONFIGS[@]}"; do
      for method in "${METHOD_CONFIGS[@]}"; do
        echo "Running: --method $method --data $data --model $model --optim $optim --seed $SEED --wandb_not_upload $WANDB_NOT_UPLOAD"
        CUDA_VISIBLE_DEVICES=0 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --seed "$SEED" $( [ "$WANDB_NOT_UPLOAD" = "True" ] && echo "--wandb_not_upload" )
      done
    done
  done
done