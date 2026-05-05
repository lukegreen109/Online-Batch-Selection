#!/bin/bash

# Usage: ./run_ablation.sh <seed> <wandb_not_upload>
# Example: ./run_ablation.sh 16 True

SEED=$1
WANDB_NOT_UPLOAD=$2

CONFIG_DIR="configs"

# Explicit config lists
# METHODS=("uniform-0.1.yaml" "divbs-0.1.yaml" "bayesian-0.1.yaml" "rholoss-0.1.yaml")
rm -r exp/
METHODS=("rholoss-0.1.yaml")
MODELS=("smallcnn.yaml")
OPTIM=("sgd-320-0.1-0.0-minimal.yaml")
DATA=("cifar10-minimal.yaml")

for data in "${DATA[@]}"; do
  for model in "${MODELS[@]}"; do
    for optim in "${OPTIM[@]}"; do
      for method in "${METHODS[@]}"; do
        DATA_PATH="$CONFIG_DIR/data/$data"
        MODEL_PATH="$CONFIG_DIR/model/$model"
        OPTIM_PATH="$CONFIG_DIR/optim/$optim"
        METHOD_PATH="$CONFIG_DIR/method/$method"
        echo "Running: --method $METHOD_PATH --data $DATA_PATH --model $MODEL_PATH --optim $OPTIM_PATH --seed $SEED --wandb_not_upload $WANDB_NOT_UPLOAD"
        CUDA_VISIBLE_DEVICES=1 uv run main.py --method "$METHOD_PATH" --data "$DATA_PATH" --model "$MODEL_PATH" --optim "$OPTIM_PATH" --seed "$SEED"
      done
    done
  done
done
