#!/bin/bash

# Usage: ./run.sh <seed> <wandb_not_upload>
# Example: ./run.sh 16 True

SEED=$1
WANDB_NOT_UPLOAD=$2

CONFIG_DIR="configs/cifar10"

METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/resnet18.yaml")
# OPTIMS=("$CONFIG_DIR/optim/"*)
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/cifar10.yaml")

for data in "${DATAS[@]}"; do
  for model in "${MODELS[@]}"; do
    for optim in "${OPTIMS[@]}"; do
      for method in "${METHODS[@]}"; do
        echo "Running: --method $method --data $data --model $model --optim $optim --seed $SEED --wandb_not_upload $WANDB_NOT_UPLOAD"
        CUDA_VISIBLE_DEVICES=1 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --seed "$SEED"  $( [ "$WANDB_NOT_UPLOAD" = "True" ] && echo "--wandb_not_upload" )
      done
    done
  done
done