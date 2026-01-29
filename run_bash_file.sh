#!/bin/bash
CFG_DIR="cfg/Noise_test"
CFG_PATTERN="*.yaml"         # <— match ALL YAML files you generated
PYTHON_SCRIPT="main.py"

for cfg in $CFG_DIR/$CFG_PATTERN; do
    echo "Running test with config: $cfg"
    base_name=$(basename "$cfg" .yaml)
    CUDA_VISIBLE_DEVICES=0 uv run $PYTHON_SCRIPT --cfg $cfg --base_dir exp/Noise_test/$base_name 

done
