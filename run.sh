#!/bin/bash

# Usage: ./run.sh <seed> <wandb_not_upload>
# Example: ./run.sh 16 True
# $( [ "$WANDB_NOT_UPLOAD" = "True" ] && echo "--wandb_not_upload" ) --wandb_not_upload $WANDB_NOT_UPLOAD"

SEED=$1
WANDB_NOT_UPLOAD=$2

CONFIG_DIR="configs/teacher_generated"

METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/twolayer.yaml")
OPTIMS=("$CONFIG_DIR/optim/adam-320-0.001-1e-05.yaml")
DATAS=("$CONFIG_DIR/data/teacher_generated.yaml")

for data in "${DATAS[@]}"; do
  for model in "${MODELS[@]}"; do
    for optim in "${OPTIMS[@]}"; do
      for method in "${METHODS[@]}"; do
        echo "Running: --method $method --data $data --model $model --optim $optim --seed $SEED"
        CUDA_VISIBLE_DEVICES=1 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --seed "$SEED"  
      done
    done
  done
done