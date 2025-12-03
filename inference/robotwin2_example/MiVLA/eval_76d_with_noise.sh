#!/bin/bash

policy_name="H-RDT"
task_name="lift_pot"  # 替换为您的任务名称
task_config="demo_clean"

# 76D模型检查点路径 (替换为您训练的76D模型路径)
ckpt_setting="checkpoints/Robotwin_muti_task_from_V4_noise/checkpoint-30000"


seed="42"
gpu_id="5"

# 76D模型配置
use_76d_model=true
use_12_2=false
noise_strategy="gaussian"
noise_scale="1.0"

# 新增：固定测试场景参数
fixed_test_scenes=true  # 启用固定测试场景

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUROBO_LOG_LEVEL=ERROR
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mUsing 76D model: ${use_76d_model}\033[0m"
echo -e "\033[33mFixed test scenes: ${fixed_test_scenes}\033[0m"
echo -e "\033[33mNoise strategy: ${noise_strategy}, scale: ${noise_scale}\033[0m"

cd ../.. # move to root for correct module imports

PYTHONWARNINGS=ignore::UserWarning \

echo -e "\033[34mUsing eval_policy.py with 76D model evaluation mode\033[0m"
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting "${ckpt_setting}" \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --use_76d_model ${use_76d_model} \
    --use_12_2 ${use_12_2} \
    --noise_strategy ${noise_strategy} \
    --noise_scale ${noise_scale} \
    --fixed_test_scenes ${fixed_test_scenes}

echo -e "\033[32m76D model evaluation completed!\033[0m" || echo "76D model evaluation completed!"