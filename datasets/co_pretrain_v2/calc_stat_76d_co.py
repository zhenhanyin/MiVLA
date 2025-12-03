#!/usr/bin/env python3
"""
协同训练数据集76维动作统计信息计算脚本

支持分别计算EgoDex和Robotwin数据集的76维动作统计信息
使用多进程并行处理，提高计算效率
"""

import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_egodex_hdf5_files(egodex_root):
    """递归查找所有EgoDex HDF5文件（包含actions_76d）"""
    logger.info(f"🔍 正在从 '{egodex_root}' 搜索EgoDex HDF5文件...")
    all_hdf5_files = list(Path(egodex_root).rglob('*.hdf5'))

    valid_files = []
    for hdf5_file in tqdm(all_hdf5_files, desc="验证EgoDex文件"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if 'actions_76d' in f:
                    actions = f['actions_76d'][:]
                    if actions.shape[1] == 76:  # 确保是76维
                        valid_files.append(hdf5_file)
        except Exception as e:
            logger.warning(f"无法读取文件 {hdf5_file}: {e}")
            continue

    logger.info(f"📊 找到 {len(valid_files)} 个有效的EgoDex HDF5文件")
    return valid_files


def find_robotwin_hdf5_files(robotwin_root):
    """递归查找所有Robotwin HDF5文件（包含virtual_human_real_robot_76d）"""
    logger.info(f"🔍 正在从 '{robotwin_root}' 搜索Robotwin HDF5文件...")
    all_hdf5_files = list(Path(robotwin_root).rglob('*.hdf5'))

    valid_files = []
    for hdf5_file in tqdm(all_hdf5_files, desc="验证Robotwin文件"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if 'virtual_human_real_robot_76d' in f:
                    actions = f['virtual_human_real_robot_76d'][:]
                    if actions.shape[1] == 76:  # 确保是76维
                        valid_files.append(hdf5_file)
        except Exception as e:
            logger.warning(f"无法读取文件 {hdf5_file}: {e}")
            continue

    logger.info(f"📊 找到 {len(valid_files)} 个有效的Robotwin HDF5文件")
    return valid_files


def worker_process_egodex_stats(file_chunk, worker_id):
    """多进程工作函数：计算EgoDex文件的局部统计信息"""
    logger.info(f"👷 EgoDex工作进程 {worker_id} 已启动，处理 {len(file_chunk)} 个文件...")

    local_min = np.full(76, np.inf, dtype=np.float32)
    local_max = np.full(76, -np.inf, dtype=np.float32)
    local_large_values_files = []
    local_error_files = []
    threshold = 10.0
    processed_count = 0

    for hdf5_file in file_chunk:
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if 'actions_76d' not in f:
                    continue

                actions = f['actions_76d'][:]

                # 检查数据有效性
                if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                    local_error_files.append(f"EgoDex NaN/Inf: {hdf5_file}")
                    continue

                # 检查异常值
                if np.any(np.abs(actions) > threshold):
                    max_val = np.abs(actions).max()
                    local_large_values_files.append(f"EgoDex: {hdf5_file} (max abs: {max_val:.3f})")

                local_min = np.minimum(local_min, np.min(actions, axis=0))
                local_max = np.maximum(local_max, np.max(actions, axis=0))
                processed_count += 1

        except Exception as e:
            local_error_files.append(f"EgoDex Error: {hdf5_file} - {str(e)}")
            continue

    return local_min, local_max, local_large_values_files, local_error_files, processed_count


def worker_process_robotwin_stats(file_chunk, worker_id):
    """多进程工作函数：计算Robotwin文件的局部统计信息"""
    logger.info(f"👷 Robotwin工作进程 {worker_id} 已启动，处理 {len(file_chunk)} 个文件...")

    local_min = np.full(76, np.inf, dtype=np.float32)
    local_max = np.full(76, -np.inf, dtype=np.float32)
    local_large_values_files = []
    local_error_files = []
    threshold = 10.0
    processed_count = 0

    for hdf5_file in file_chunk:
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if 'virtual_human_real_robot_76d' not in f:
                    continue

                actions = f['virtual_human_real_robot_76d'][:]

                # 检查数据有效性
                if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                    local_error_files.append(f"Robotwin NaN/Inf: {hdf5_file}")
                    continue

                # 检查异常值
                if np.any(np.abs(actions) > threshold):
                    max_val = np.abs(actions).max()
                    local_large_values_files.append(f"Robotwin: {hdf5_file} (max abs: {max_val:.3f})")

                local_min = np.minimum(local_min, np.min(actions, axis=0))
                local_max = np.maximum(local_max, np.max(actions, axis=0))
                processed_count += 1

        except Exception as e:
            local_error_files.append(f"Robotwin Error: {hdf5_file} - {str(e)}")
            continue

    return local_min, local_max, local_large_values_files, local_error_files, processed_count


def process_dataset_parallel(file_list, dataset_name, num_processes):
    """并行处理数据集统计计算"""
    if not file_list:
        logger.warning(f"❌ {dataset_name} 数据集没有找到有效文件")
        return None, [], [], 0

    # 分割文件列表
    num_files = len(file_list)
    chunk_size = max(1, (num_files + num_processes - 1) // num_processes)
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, num_files, chunk_size)]

    # 初始化全局统计变量
    global_min = np.full(76, np.inf, dtype=np.float32)
    global_max = np.full(76, -np.inf, dtype=np.float32)
    all_large_values_files = []
    all_error_files = []
    total_files_processed = 0

    # 选择工作函数
    if dataset_name == "egodex":
        worker_func = worker_process_egodex_stats
    else:  # robotwin
        worker_func = worker_process_robotwin_stats

    # 并行处理
    logger.info(f"🚀 使用 {len(file_chunks)} 个工作进程开始并行计算 {dataset_name} 统计...")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(worker_func, chunk, i) for i, chunk in enumerate(file_chunks)]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"处理{dataset_name}进程结果"):
            try:
                local_min, local_max, local_large_values, local_errors, count = future.result()

                global_min = np.minimum(global_min, local_min)
                global_max = np.maximum(global_max, local_max)
                all_large_values_files.extend(local_large_values)
                all_error_files.extend(local_errors)
                total_files_processed += count

            except Exception as e:
                logger.error(f"{dataset_name} 工作进程出错: {e}")

    return (global_min, global_max), all_large_values_files, all_error_files, total_files_processed


def main():
    parser = argparse.ArgumentParser(description='协同训练数据集76维动作统计信息计算')

    # EgoDex数据集参数
    parser.add_argument('--egodex_root', type=str,
                       default='/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain',
                       help='EgoDex数据集根目录')

    # Robotwin数据集参数
    parser.add_argument('--robotwin_root', type=str,
                       default='/home/yinzhenhan/Human_Pre/dataset/Robotwin-gen-human-full-62d',
                       help='Robotwin数据集根目录')

    # 输出参数
    parser.add_argument('--output_path', type=str,
                       default='/home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain_v2/co_stat_76d.json',
                       help='输出的JSON统计文件路径')
    parser.add_argument('--log_dir', type=str,
                       default='/home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain_v2',
                       help='日志文件输出目录')

    # 处理参数
    parser.add_argument('--num_processes', type=int, default=24,
                       help='使用的进程数量')
    parser.add_argument('--threshold', type=float, default=4,
                       help='异常值检测阈值')

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("🎯 协同训练数据集76维动作统计计算")
    print("=" * 60)
    print("--- 配置信息 ---")
    print(f"- EgoDex数据源: {args.egodex_root}")
    print(f"- Robotwin数据源: {args.robotwin_root}")
    print(f"- 输出文件: {args.output_path}")
    print(f"- 日志目录: {args.log_dir}")
    print(f"- 并行进程数: {args.num_processes}")
    print(f"- 异常值阈值: {args.threshold}")
    print("--------------------")

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1. 查找所有文件
    print("\n📋 查找数据文件...")
    egodex_files = find_egodex_hdf5_files(args.egodex_root)
    robotwin_files = find_robotwin_hdf5_files(args.robotwin_root)

    if not egodex_files and not robotwin_files:
        logger.error("❌ 未找到任何有效文件，程序退出")
        return

    # 2. 并行计算统计信息
    print("\n🔢 并行计算统计信息...")

    egodex_stats = None
    robotwin_stats = None
    egodex_errors = []
    robotwin_errors = []
    egodex_processed = 0
    robotwin_processed = 0

    # 处理EgoDex数据集
    if egodex_files:
        egodex_stats, egodex_large_vals, egodex_errors, egodex_processed = process_dataset_parallel(
            egodex_files, "egodex", args.num_processes)

    # 处理Robotwin数据集
    if robotwin_files:
        robotwin_stats, robotwin_large_vals, robotwin_errors, robotwin_processed = process_dataset_parallel(
            robotwin_files, "robotwin", args.num_processes)

    # 3. 生成统计结果
    stat_dict = {}

    if egodex_stats:
        egodex_min, egodex_max = egodex_stats
        stat_dict["egodex_76d"] = {
            "min": egodex_min.tolist(),
            "max": egodex_max.tolist(),
        }

        # 保存EgoDex异常值日志
        egodex_log_path = os.path.join(args.log_dir, "egodex_large_values_76d.txt")
        with open(egodex_log_path, 'w') as f:
            f.write(f"EgoDex数据集中绝对值 > {args.threshold} 的文件:\n\n")
            for file_info in egodex_large_vals:
                f.write(f"{file_info}\n")

    if robotwin_stats:
        robotwin_min, robotwin_max = robotwin_stats
        stat_dict["robotwin_76d"] = {
            "min": robotwin_min.tolist(),
            "max": robotwin_max.tolist(),
        }

        # 保存Robotwin异常值日志
        robotwin_log_path = os.path.join(args.log_dir, "robotwin_large_values_76d.txt")
        with open(robotwin_log_path, 'w') as f:
            f.write(f"Robotwin数据集中绝对值 > {args.threshold} 的文件:\n\n")
            for file_info in robotwin_large_vals:
                f.write(f"{file_info}\n")

    # 保存错误日志
    all_errors = egodex_errors + robotwin_errors
    if all_errors:
        error_log_path = os.path.join(args.log_dir, "processing_errors_76d.txt")
        with open(error_log_path, 'w') as f:
            f.write("处理过程中遇到的错误文件:\n\n")
            for error_info in all_errors:
                f.write(f"{error_info}\n")

    # 保存统计文件
    with open(args.output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)

    end_time = time.time()

    # 4. 打印最终报告
    print("\n" + "=" * 60)
    print("🎉 协同训练数据集76维动作统计完成!")
    print("=" * 60)
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 EgoDex: 成功处理 {egodex_processed} 个文件，异常 {len(egodex_errors)} 个")
    print(f"📊 Robotwin: 成功处理 {robotwin_processed} 个文件，异常 {len(robotwin_errors)} 个")

    if egodex_stats:
        egodex_min, egodex_max = egodex_stats
        print(f"📈 EgoDex 76维范围: [{egodex_min.min():.6f}, {egodex_max.max():.6f}]")

    if robotwin_stats:
        robotwin_min, robotwin_max = robotwin_stats
        print(f"📈 Robotwin 76维范围: [{robotwin_min.min():.6f}, {robotwin_max.max():.6f}]")

    print(f"📄 统计结果已保存至: {args.output_path}")

    if all_errors:
        print(f"⚠️  发现 {len(all_errors)} 个处理错误，详见日志文件")

    print("=" * 60)


if __name__ == "__main__":
    main()