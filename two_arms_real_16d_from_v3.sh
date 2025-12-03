export NCCL_TIMEOUT=3600

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1np0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=2,3


export NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1 # 尝试禁用P2P，有时可以解决NVLink配置问题

source /home/yinzhenhan/Human_Pre/H_RDT/datasets/real_robot/setup_real_robot.sh

# export CFLAGS="-I/usr/include"
# export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
# export CUTLASS_PATH="/data/lingxuan/cutlass"


export OUTPUT_DIR="./checkpoints/real_robot_finetune/two_arms_put_the_ball/HRDT_76d_sele_16d_eef_con"


export VISION_ENCODER_NAME="dino-siglip"
conda activate hrdt
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

accelerate launch --main_process_port 29510 main.py \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --deepspeed="./configs/zero1.json" \
    --config_path="configs/eef_16d.yaml" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=48 \
    --max_train_steps=15000 \
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=10 \
    --report_to=tensorboard  \
    --upsample_rate=3 \
    --image_aug \
    --precomp_lang_embed \
    --training_mode="lang" \
    --mode='finetune' \
    --task_name="two_arms_put_the_tissue" \
    --dataset_name="real_robot_eef_16d" \
    --pretrained_backbone_path="./checkpoints/pretrain_union_62d_V3/checkpoint-100000/pytorch_model.bin"  \
    # --action_stats_file="/home/yinzhenhan/Human_Pre/H_RDT/datasets/real_robot/place_the_cup_xianwei_14d.json" \
    # --normalize_actions \



    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training

    # --unfreeze_vision_encoder \
    # --vision_encoder_lr_ratio=0.1 \