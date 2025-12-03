#!/bin/bash

# H-RDT RobotWin2 Data Processing Setup
# Set your paths here

# Required paths - modify these according to your environment
export REAL_ROBOT_DATA_ROOT="/home/yinzhenhan/Human_Pre/dataset/zhenji"

export T5_MODEL_PATH="/home/yinzhenhan/Models/t5-v1_1-xxl"

# Project structure (auto-detected)
export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_finetune_real_robot.yaml"

# export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/real_robot"

# Processing parameters
# export NUM_PROCESSES=64
# export NUM_GPUS=4
# export PROCESSES_PER_GPU=6

# Create output directory
mkdir -p "$HRDT_OUTPUT_DIR"

# Add project to Python path
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

echo "Real_Robot environment setup completed"

echo "Data Root: $REAL_ROBOT_DATA_ROOT"

echo "T5 Model: $T5_MODEL_PATH"
# echo "Output Dir: $HRDT_OUTPUT_DIR" 