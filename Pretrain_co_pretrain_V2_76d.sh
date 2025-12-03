#!/bin/bash


export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1np0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=4,5

export NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1 # 尝试禁用P2P，有时可以解决NVLink配置问题




# source /home/yinzhenhan/anaconda3/etc/profile.d/conda.sh
conda activate hrdt
source /home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain_v2/setup_pretrain.sh




export VISION_ENCODER_NAME="dino-siglip"


cd /home/yinzhenhan/Human_Pre/H_RDT

echo "Current working directory: $(pwd)"

# 输出配置
export OUTPUT_DIR="./checkpoints/exp_cotrain_V2"



echo "=========================================="
echo "🎯 启动 EgoDex-Robotwin 协同预训练"
echo "=========================================="
echo "=========================================="

# 创建输出目录
mkdir -p $OUTPUT_DIR



echo "🔧 修复路径问题：确保在H_RDT目录下启动"
echo "✅ 协同预训练已启动！"

NCCL_TIMEOUT=3600 accelerate launch --main_process_port 29800 main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero1.json" \
    --config_path="configs/hrdt_pretrain_76d.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=32 \
    --max_train_steps=50000 \
    --checkpointing_period=500 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant_with_warmup" \
    --checkpoint_latest_period=1000 \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --upsample_rate=3 \
    --image_aug \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="pretrain" \
    --action_stats_file="/home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain_v2/co_stat_76d.json" \
    --egodex_weight=0.8 \
    --dataset_name="egodex_robotwin_copretrain_V2" \
    --pretrained_model_name_or_path="./checkpoints/pretrain-0618/checkpoint-500000-apapted-76d" \