#!/bin/bash

# H-RDT Pretrain Data Processing Setup
# Set your paths here

# Required paths - modify these according to your environment
export EGODEX_DATA_ROOT="/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain"
export T5_MODEL_PATH="/home/yinzhenhan/Models/t5-v1_1-xxl"

# Project structure (auto-detected)
export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_pretrain.yaml"
export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/co_pretrain"

# Fix path calculation - we're already in H_RDT/datasets/co_pretrain
export HRDT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

# Processing parameters
export NUM_PROCESSES=8
export FORCE_OVERWRITE=true

# Create output directory
mkdir -p "$HRDT_OUTPUT_DIR"

# Add project to Python path
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"

echo "Pretrain environment setup completed"
echo "Data Root: $EGODEX_DATA_ROOT"
echo "T5 Model: $T5_MODEL_PATH"
echo "Output Dir: $HRDT_OUTPUT_DIR" 