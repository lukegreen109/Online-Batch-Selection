#!/bin/bash

# Usage: ./run.sh

SEEDS=(2 3 4 5)
DIAGNOSTICS="configs/diagnostics/snapshots_log_interval.yaml"

CONFIG_DIR="configs/cifar10"

METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/resnet18.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/cifar10_noise.yaml")

for SEED in "${SEEDS[@]}"; do
  for data in "${DATAS[@]}"; do
    for model in "${MODELS[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        for method in "${METHODS[@]}"; do
          echo "Running: --method $method --data $data --model $model --optim $optim --diagnostics $DIAGNOSTICS --seed $SEED"
          CUDA_VISIBLE_DEVICES=0 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --diagnostics "$DIAGNOSTICS" --seed "$SEED" --wandb_not_upload
        done
      done
    done
  done
done