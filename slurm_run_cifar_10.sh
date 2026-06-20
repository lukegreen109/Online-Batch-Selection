#!/bin/bash

SEEDS=(3 4)
DIAGNOSTICS="configs/diagnostics/snapshots_log_interval.yaml"
CONFIG_DIR="configs/cifar10"
METHODS=("$CONFIG_DIR/method/"*)
MODELS=("$CONFIG_DIR/model/lenet.yaml")
OPTIMS=("$CONFIG_DIR/optim/adamw-320-0.001-0.01.yaml")
DATAS=("$CONFIG_DIR/data/cifar10.yaml")
mkdir -p logs
SAVE_DIRS_FILE="logs/save_dirs_$(date +%Y%m%d_%H%M%S).txt"

for SEED in "${SEEDS[@]}"; do
for data in "${DATAS[@]}"; do
for model in "${MODELS[@]}"; do
for optim in "${OPTIMS[@]}"; do
for method in "${METHODS[@]}"; do

python perform_downloads.py --method "$method"

SAVE_DIR=$(python get_save_dir.py --method "$method" --data "$data" --model "$model" --optim "$optim" --seed "$SEED")
echo "$SAVE_DIR" >> "$SAVE_DIRS_FILE"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=cifar10_s${SEED}
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time 2:00:00
python main.py \
    --method "$method" \
    --data "$data" \
    --model "$model" \
    --optim "$optim" \
    --diagnostics "$DIAGNOSTICS" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --wandb_not_upload
EOF

done
done
done
done
done

echo "All jobs submitted. Running Weights & Biases sync daemon. Ctrl+C to stop syncing"
python wandb-sync-daemon.py --save_dirs "$SAVE_DIRS_FILE"