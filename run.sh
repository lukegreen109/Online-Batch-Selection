#!/bin/bash

# Usage: ./run_ablation.sh <seed> <wandb_not_upload>
# Example: ./run_ablation.sh 16 True

SEED=$1
WANDB_NOT_UPLOAD=$2

CONFIG_DIR="configs"

# Explicit config lists
METHODS=("rholoss-0.1.yaml")
MODELS=("resnet18.yaml")
OPTIM=("adam-320-0.001-0.0.yaml")
DATA=("cifar10.yaml")

for data in "${DATA[@]}"; do
  for model in "${MODELS[@]}"; do
    for optim in "${OPTIM[@]}"; do
      for method in "${METHODS[@]}"; do
        DATA_PATH="$CONFIG_DIR/data/$data"
        MODEL_PATH="$CONFIG_DIR/model/$model"
        OPTIM_PATH="$CONFIG_DIR/optim/$optim"
        METHOD_PATH="$CONFIG_DIR/method/$method"
        echo "Running: --method $METHOD_PATH --data $DATA_PATH --model $MODEL_PATH --optim $OPTIM_PATH --seed $SEED --wandb_not_upload $WANDB_NOT_UPLOAD"
        CUDA_VISIBLE_DEVICES=0 uv run main.py --method "$METHOD_PATH" --data "$DATA_PATH" --model "$MODEL_PATH" --optim "$OPTIM_PATH" --seed "$SEED" $( [ "$WANDB_NOT_UPLOAD" = "True" ] && echo "--wandb_not_upload" )
      done
    done
  done
done