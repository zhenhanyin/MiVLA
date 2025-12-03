#!/bin/bash

# 62D微调脚本：保留预训练人类先验，只训练机器人动作
# 策略：48D噪声 + 14D真实数据，只计算后14D损失

export NCCL_TIMEOUT=3600

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1np0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=4


export NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1 # 尝试禁用P2P，有时可以解决NVLink配置问题




# export CFLAGS="-I/usr/include"
# export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
# export CUTLASS_PATH="/data/lingxuan/cutlass"

export WANDB_PROJECT="Union_finetune_62d_robotwin2_V3"
export OUTPUT_DIR="./checkpoints/finetune_union_single_task_V3/place_a2b_left/finetune_76d_selective_14d_joint"

export VISION_ENCODER_NAME="dino-siglip"

conda activate hrdt

# Add current directory to PYTHONPATH
export PYTHONPATH=/home/yinzhenhan/Human_Pre/H_RDT:$PYTHONPATH

source datasets/robotwin2/setup_robotwin2.sh

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...



NCCL_TIMEOUT=3600 accelerate launch --main_process_port 29540 main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero1.json" \
    --config_path="configs/robotwin_finetune_76d.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=32 \
    --max_train_steps=10000 \
    --checkpointing_period=2500 \
    --sample_period=250 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="cosine" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=10 \
    --upsample_rate=3 \
    --image_aug \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode="finetune_76d_with_noise_loss_robot" \
    --dataset_name="robotwin_agilex_76D_with_noise" \
    --pretrained_backbone_path="./checkpoints/pretrain_union_76d_V4/checkpoint-70000/pytorch_model.bin"


    #--report_to=wandb \
    # For finetune mode with specific robot embodiment, use these parameters instead:
    # --mode="finetune" \
    # --pretrained_backbone_path="./checkpoints/pretrain-0618/pytorch_model.bin" \
    # --config_path="configs/hrdt_finetune.yaml" \  # Config with different action_dim for target robot
    # --dataset_type="finetune" \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    #    --gradient_checkpointing \