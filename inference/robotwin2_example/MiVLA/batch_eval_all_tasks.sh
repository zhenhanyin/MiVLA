#!/bin/bash

# 批量测试脚本 - 自动测试多个任务在多个GPU上的性能
# 支持demo_clean和demo_randomized两种配置

policy_name="MiVLA"
task_configs=("demo_clean" "demo_randomized")
ckpt_setting="checkpoints/Muti_from_V4_robotwin_agilex_76D_with_noise_arx_x5/checkpoint-15000"
seed="42"

# 可用GPU列表
gpu_ids=(2 3 4 )
max_concurrent_per_gpu=2  # 每个GPU最多同时执行2个任务

# 其他配置
use_76d_model=true
use_12_2=false
noise_strategy="gaussian"
noise_scale="1.0"
fixed_test_scenes=true

# 日志目录 - 使用绝对路径
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$script_dir/batch_ARX_MiVLA_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# 任务列表文件 - 使用绝对路径
task_list_file="$script_dir/task_need_test.txt"

# 临时目录用于GPU任务计数
temp_dir="/tmp/batch_test_$$"
mkdir -p "$temp_dir"

# 初始化GPU任务计数器
for gpu_id in "${gpu_ids[@]}"; do
    echo "0" > "$temp_dir/gpu_${gpu_id}_count"
done

# 函数：获取当前任务最少的GPU
get_least_busy_gpu() {
    local best_gpu=""
    local min_count=999

    for gpu_id in "${gpu_ids[@]}"; do
        local count=$(cat "$temp_dir/gpu_${gpu_id}_count")
        if [ "$count" -lt "$min_count" ]; then
            min_count="$count"
            best_gpu="$gpu_id"
        fi
    done

    echo "$best_gpu"
}

# 函数：增加GPU任务计数
increment_gpu_count() {
    local gpu_id="$1"
    local current_count=$(cat "$temp_dir/gpu_${gpu_id}_count")
    echo $((current_count + 1)) > "$temp_dir/gpu_${gpu_id}_count"
}

# 函数：减少GPU任务计数
decrement_gpu_count() {
    local gpu_id="$1"
    local current_count=$(cat "$temp_dir/gpu_${gpu_id}_count")
    echo $((current_count - 1)) > "$temp_dir/gpu_${gpu_id}_count"
}

# 函数：等待GPU可用槽位
wait_for_gpu_slot() {
    local gpu_id="$1"
    while true; do
        local count=$(cat "$temp_dir/gpu_${gpu_id}_count")
        if [ "$count" -lt "$max_concurrent_per_gpu" ]; then
            break
        fi
        sleep 5
    done
}

# 函数：运行单个测试（同步执行）
run_test_sync() {
    local task_name="$1"
    local task_config="$2"
    local gpu_id="$3"

    # 增加GPU计数
    increment_gpu_count "$gpu_id"

    local log_file="$log_dir/${task_name}_${task_config}_gpu${gpu_id}.log"

    echo -e "\033[34m[$(date '+%Y-%m-%d %H:%M:%S')] Starting task: $task_name, config: $task_config, GPU: $gpu_id\033[0m"

    # 保存当前目录
    local current_dir=$(pwd)

    # 同步运行测试任务
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export CUROBO_LOG_LEVEL=ERROR

        cd ../..

        echo "开始执行任务: $task_name ($task_config) on GPU $gpu_id" | tee -a "$log_file"
        echo "执行时间: $(date)" | tee -a "$log_file"
        echo "命令: python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml --overrides --task_name $task_name --task_config $task_config --ckpt_setting $ckpt_setting --seed $seed --policy_name $policy_name --use_76d_model $use_76d_model --use_12_2 $use_12_2 --noise_strategy $noise_strategy --noise_scale $noise_scale --fixed_test_scenes $fixed_test_scenes" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"

        PYTHONWARNINGS=ignore::UserWarning python script/eval_policy.py \
            --config policy/$policy_name/deploy_policy.yml \
            --overrides \
            --task_name "$task_name" \
            --task_config "$task_config" \
            --ckpt_setting "$ckpt_setting" \
            --seed "$seed" \
            --policy_name "$policy_name" \
            --use_76d_model "$use_76d_model" \
            --use_12_2 "$use_12_2" \
            --noise_strategy "$noise_strategy" \
            --noise_scale "$noise_scale" \
            --fixed_test_scenes "$fixed_test_scenes" \
            2>&1 | tee -a "$log_file"

        local exit_code=${PIPESTATUS[0]}

        echo "----------------------------------------" | tee -a "$log_file"
        echo "任务完成时间: $(date)" | tee -a "$log_file"
        echo "退出代码: $exit_code" | tee -a "$log_file"

        if [ $exit_code -eq 0 ]; then
            echo -e "\033[32m[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETED: $task_name ($task_config) on GPU $gpu_id\033[0m"
            echo "任务状态: 成功" >> "$log_file"
        else
            echo -e "\033[31m[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $task_name ($task_config) on GPU $gpu_id (exit code: $exit_code)\033[0m"
            echo "任务状态: 失败 (退出代码: $exit_code)" >> "$log_file"
        fi

    )

    local exit_code=$?

    # 减少GPU计数
    decrement_gpu_count "$gpu_id"

    return $exit_code
}

# 函数：运行单个测试（异步执行）
run_test() {
    local task_name="$1"
    local task_config="$2"
    local gpu_id="$3"

    # 增加GPU计数
    increment_gpu_count "$gpu_id"

    local log_file="$log_dir/${task_name}_${task_config}_gpu${gpu_id}.log"
    local pid_file="$temp_dir/${task_name}_${task_config}_gpu${gpu_id}.pid"

    echo -e "\033[34m[$(date '+%Y-%m-%d %H:%M:%S')] Starting task: $task_name, config: $task_config, GPU: $gpu_id\033[0m"

    # 后台运行测试任务
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export CUROBO_LOG_LEVEL=ERROR

        cd ../..

        echo "开始执行任务: $task_name ($task_config) on GPU $gpu_id" | tee -a "$log_file"
        echo "执行时间: $(date)" | tee -a "$log_file"
        echo "========================================" | tee -a "$log_file"

        PYTHONWARNINGS=ignore::UserWarning python script/eval_policy.py \
            --config policy/$policy_name/deploy_policy.yml \
            --overrides \
            --task_name "$task_name" \
            --task_config "$task_config" \
            --ckpt_setting "$ckpt_setting" \
            --seed "$seed" \
            --policy_name "$policy_name" \
            --use_76d_model "$use_76d_model" \
            --use_12_2 "$use_12_2" \
            --noise_strategy "$noise_strategy" \
            --noise_scale "$noise_scale" \
            --fixed_test_scenes "$fixed_test_scenes" \
            2>&1 | tee -a "$log_file"

        local exit_code=${PIPESTATUS[0]}

        echo "========================================" | tee -a "$log_file"
        echo "任务完成时间: $(date)" | tee -a "$log_file"
        echo "退出代码: $exit_code" | tee -a "$log_file"

        if [ $exit_code -eq 0 ]; then
            echo -e "\033[32m[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETED: $task_name ($task_config) on GPU $gpu_id\033[0m"
            echo "任务状态: 成功" >> "$log_file"
        else
            echo -e "\033[31m[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $task_name ($task_config) on GPU $gpu_id (exit code: $exit_code)\033[0m"
            echo "任务状态: 失败 (退出代码: $exit_code)" >> "$log_file"
        fi

    ) &

    # 记录进程ID和GPU信息
    local pid=$!
    echo "$pid:$gpu_id" > "$pid_file"
}

# 主函数
main() {
    echo -e "\033[33m=== 批量测试开始 ===\033[0m"
    echo -e "\033[33m使用GPU: ${gpu_ids[*]}\033[0m"
    echo -e "\033[33m每个GPU最大并发: $max_concurrent_per_gpu\033[0m"
    echo -e "\033[33m日志目录: $log_dir\033[0m"
    echo -e "\033[33m任务配置: ${task_configs[*]}\033[0m"
    echo ""

    # 检查任务列表文件是否存在
    if [ ! -f "$task_list_file" ]; then
        echo -e "\033[31m错误: 任务列表文件 $task_list_file 不存在!\033[0m"
        exit 1
    fi

    # 读取任务列表
    local tasks=()
    while IFS= read -r task_name; do
        # 跳过空行和注释行
        if [[ -n "$task_name" && ! "$task_name" =~ ^[[:space:]]*# ]]; then
            tasks+=("$task_name")
        fi
    done < "$task_list_file"

    echo -e "\033[33m找到 ${#tasks[@]} 个任务需要测试\033[0m"
    echo ""

    # 计算总任务数
    local total_tasks=$((${#tasks[@]} * ${#task_configs[@]}))
    local completed_tasks=0

    echo -e "\033[33m开始并发执行任务队列，${#gpu_ids[@]}个GPU同时工作\033[0m"

    # 创建任务队列
    local task_queue=()
    for task_name in "${tasks[@]}"; do
        for task_config in "${task_configs[@]}"; do
            task_queue+=("$task_name:$task_config")
        done
    done

    echo -e "\033[33m任务队列中共有 ${#task_queue[@]} 个任务\033[0m"

    # 持续监控和分配任务
    local task_index=0
    local total_queued_tasks=${#task_queue[@]}
    local launched_tasks=0

    while [ "$completed_tasks" -lt "$total_tasks" ]; do
        # 尝试启动新任务
        while [ "$launched_tasks" -lt "$total_queued_tasks" ]; do
            local gpu_id=""
            local found_gpu=false

            # 寻找可用GPU
            for check_gpu in "${gpu_ids[@]}"; do
                local count_file="$temp_dir/gpu_${check_gpu}_count"
                local count=0
                if [ -f "$count_file" ]; then
                    count=$(cat "$count_file" 2>/dev/null || echo "0")
                fi

                if [ "$count" -lt "$max_concurrent_per_gpu" ]; then
                    gpu_id="$check_gpu"
                    found_gpu=true
                    break
                fi
            done

            if [ "$found_gpu" = true ]; then
                # 启动新任务
                local task_info="${task_queue[$task_index]}"
                local task_name="${task_info%:*}"
                local task_config="${task_info#*:}"

                echo -e "\033[34m分配新任务: $task_name ($task_config) -> GPU:$gpu_id (进度: $launched_tasks/$total_queued_tasks)\033[0m"
                run_test "$task_name" "$task_config" "$gpu_id"

                launched_tasks=$((launched_tasks + 1))
                task_index=$((task_index + 1))

                # 短暂延迟避免资源竞争
                sleep 1
            else
                # 没有可用GPU，跳出循环等待任务完成
                break
            fi
        done

        # 检查已完成的任务
        local all_pids=($(ls "$temp_dir"/*.pid 2>/dev/null))
        local active_tasks=0

        for pid_file in "${all_pids[@]}"; do
            if [ -f "$pid_file" ]; then
                local pid_info=$(cat "$pid_file")
                local pid="${pid_info%:*}"
                local gpu_id="${pid_info#*:}"

                if kill -0 "$pid" 2>/dev/null; then
                    active_tasks=$((active_tasks + 1))
                else
                    # 进程已完成
                    wait "$pid" 2>/dev/null

                    # 减少GPU计数
                    decrement_gpu_count "$gpu_id"

                    # 清理PID文件
                    rm -f "$pid_file"
                    completed_tasks=$((completed_tasks + 1))

                    echo -e "\033[32m任务完成! 进度: $completed_tasks/$total_tasks (活跃: $active_tasks)\033[0m"
                fi
            fi
        done

        # 如果所有任务都启动了且没有活跃任务，说明完成了
        if [ "$launched_tasks" -eq "$total_queued_tasks" ] && [ "$active_tasks" -eq 0 ]; then
            break
        fi

        # 等待一段时间再检查
        sleep 3
    done

    echo ""
    echo -e "\033[32m=== 批量测试完成 ===\033[0m"
    echo -e "\033[33m所有日志保存在: $log_dir\033[0m"

    # 生成汇总报告
    echo -e "\033[33m生成汇总报告...\033[0m"
    local summary_file="$log_dir/summary.txt"
    echo "批量测试汇总报告 - $(date)" > "$summary_file"
    echo "=================================" >> "$summary_file"
    echo "总任务数: $total_tasks" >> "$summary_file"
    echo "使用GPU: ${gpu_ids[*]}" >> "$summary_file"
    echo "" >> "$summary_file"

    # 统计成功和失败的任务
    local success_count=0
    local failed_count=0

    for task_name in "${tasks[@]}"; do
        for task_config in "${task_configs[@]}"; do
            echo "任务: $task_name ($task_config)" >> "$summary_file"
            local found=false

            for gpu_id in "${gpu_ids[@]}"; do
                local log_file="$log_dir/${task_name}_${task_config}_gpu${gpu_id}.log"
                if [ -f "$log_file" ]; then
                    if tail -10 "$log_file" | grep -q "evaluation completed"; then
                        echo "  状态: 成功 (GPU $gpu_id)" >> "$summary_file"
                        success_count=$((success_count + 1))
                        found=true
                        break
                    fi
                fi
            done

            if [ "$found" = false ]; then
                echo "  状态: 失败或未完成" >> "$summary_file"
                failed_count=$((failed_count + 1))
            fi
        done
        echo "" >> "$summary_file"
    done

    echo "成功: $success_count, 失败: $failed_count" >> "$summary_file"
    echo -e "\033[33m汇总报告: $summary_file\033[0m"

    # 清理临时文件
    rm -rf "$temp_dir"

    echo -e "\033[32m批量测试完成!\033[0m"
}

# 信号处理函数
cleanup() {
    echo -e "\033[33m接收到中断信号，正在清理...\033[0m"

    # 终止所有后台进程
    local all_pids=($(ls "$temp_dir"/*.pid 2>/dev/null))
    for pid_file in "${all_pids[@]}"; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null
            rm -f "$pid_file"
        fi
    done

    # 清理临时文件
    rm -rf "$temp_dir"

    echo -e "\033[33m清理完成\033[0m"
    exit 1
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 运行主函数
main "$@"