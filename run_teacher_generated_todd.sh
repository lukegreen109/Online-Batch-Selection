#!/bin/bash

# Usage: ./run.sh

SEEDS=(1 2)
DIAGNOSTICS="configs/diagnostics/all_log_interval.yaml"

CONFIG_DIR="configs/teacher_generated"

METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/twolayer.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/teacher_generated.yaml")

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

SEEDS=(1 2)
DIAGNOSTICS="configs/diagnostics/all_log_interval.yaml"

CONFIG_DIR="configs/teacher_generated"

METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/twolayer.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/teacher_generated_noise.yaml")

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