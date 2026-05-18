#!/bin/bash

# Usage: ./run.sh

SEEDS=(1)
DIAGNOSTICS="configs/diagnostics/snapshots_log_interval.yaml"

CONFIG_DIR="configs/mnist"

METHODS=("$CONFIG_DIR/method/rholoss-0.1.yaml")
MODELS=("$CONFIG_DIR/model/lenet.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/seed_1/mnist_noise_0.1p_v1.yaml" "$CONFIG_DIR/data/seed_1/mnist_noise_0.2p_v1.yaml" "$CONFIG_DIR/data/seed_1/mnist_noise_0.0p_v1.yaml")

for SEED in "${SEEDS[@]}"; do
  for data in "${DATAS[@]}"; do
    for model in "${MODELS[@]}"; do
      for optim in "${OPTIMS[@]}"; do
        for method in "${METHODS[@]}"; do
          echo "Running: --method $method --data $data --model $model --optim $optim --diagnostics $DIAGNOSTICS --seed $SEED"
          CUDA_VISIBLE_DEVICES=0 uv run main.py --method "$method" --data "$data" --model "$model" --optim "$optim" --diagnostics "$DIAGNOSTICS" --seed "$SEED"
        done
      done
    done
  done
done