#!/usr/bin/env python3
"""
Egodex坐标系转换脚本 (V3 - 原地覆盖模式)

功能：
- 递归遍历指定的 egodex(含14维标签) 数据目录。
- 读取每个 HDF5 文件中的 'actions_62d' 数据集。
- 将其中前48维（人手动作）从 Egodex 坐标系转换到机器人（PyBullet）坐标系。
- 直接在原文件中覆盖转换后的48维数据，保持后14维不变。
- 使用多进程加速。

注意：此脚本会直接修改原始HDF5文件，请先备份重要数据！
parser.add_argument('--input_dir', type=str, required=True,
                    help='包含Egodex HDF5文件的数据根目录')
parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                    help='使用的进程数')
"""

import h5py
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import multiprocessing
from datetime import datetime

# --- 核心变换矩阵 ---
# 该矩阵将一个在 Egodex 坐标系下表示的向量，转换为在机器人坐标系下的表示
# robot_vec = R_egodex_to_robot @ egodex_vec
R_egodex_to_robot = np.array([
    [ 0,  0, -1],
    [-1,  0,  0],
    [ 0,  1,  0]
])

def convert_48d_coords(actions_48d_egodex: np.ndarray) -> np.ndarray:
    """
    将一个48维动作向量（或N x 48的数组）从 Egodex 坐标系转换为机器人坐标系。
    """
    if actions_48d_egodex.ndim == 1:
        actions_48d_egodex = actions_48d_egodex.reshape(1, -1)

    num_frames = actions_48d_egodex.shape[0]
    actions_48d_robot = np.zeros_like(actions_48d_egodex)

    for i in range(num_frames):
        # 分解左右手
        left_24d_e = actions_48d_egodex[i, 0:24]
        right_24d_e = actions_48d_egodex[i, 24:48]

        # 分别转换
        left_24d_r = _convert_24d(left_24d_e)
        right_24d_r = _convert_24d(right_24d_e)

        actions_48d_robot[i] = np.concatenate([left_24d_r, right_24d_r])

    return actions_48d_robot.squeeze()

def _convert_24d(action_24d_egodex: np.ndarray) -> np.ndarray:
    """转换单只手的24维动作"""
    # 分解 24D: wrist_pos(3), wrist_rot_6d(6), fingertips_pos(15)
    pos_e = action_24d_egodex[0:3]
    rot6d_e = action_24d_egodex[3:9]
    tips_e = action_24d_egodex[9:].reshape(5, 3)

    # --- 1. 转换位置向量 ---
    pos_r = R_egodex_to_robot @ pos_e
    tips_r = (R_egodex_to_robot @ tips_e.T).T

    # --- 2. 转换姿态 ---
    r1_e, r2_e = rot6d_e[:3], rot6d_e[3:]
    r3_e = np.cross(r1_e, r2_e)
    r3_e /= np.linalg.norm(r3_e) if np.linalg.norm(r3_e) > 1e-6 else 1
    rot_matrix_e = np.column_stack([r1_e, r2_e, r3_e])
    rot_matrix_r = R_egodex_to_robot @ rot_matrix_e
    rot6d_r = np.concatenate([rot_matrix_r[:, 0], rot_matrix_r[:, 1]])

    return np.concatenate([pos_r, rot6d_r, tips_r.flatten()])

def process_file_inplace_worker(file_path):
    """
    工作进程函数，用于多进程原地处理

    Args:
        file_path: HDF5文件路径

    Returns:
        str: 处理结果信息
    """
    return process_file_inplace(file_path)

def process_file_inplace(file_path: str) -> str:
    """
    原地处理单个HDF5文件，直接在原文件中覆盖转换后的48维数据

    Args:
        file_path: HDF5文件路径

    Returns:
        str: 处理结果信息
    """
    try:
        # 检查文件是否已经被转换
        with h5py.File(file_path, 'r') as f:
            if 'actions_62d' not in f:
                return f"跳过 (无 'actions_62d' 数据集): {file_path}"

            # 检查是否已经被转换过
            dset = f['actions_62d']
            if dset.attrs.get('coordinate_system') == 'pybullet_unified':
                return f"跳过 (已转换): {file_path}"

        # 读取原始数据
        with h5py.File(file_path, 'r+') as f:
            if 'actions_62d' not in f:
                return f"跳过 (无 'actions_62d' 数据集): {file_path}"

            # 读取数据
            actions_62d_original = f['actions_62d'][:]

            # 分离
            actions_48d_egodex = actions_62d_original[:, :48]
            actions_14d_robot = actions_62d_original[:, 48:]

            # 转换48维数据
            actions_48d_robot = convert_48d_coords(actions_48d_egodex)

            # 组合：转换后的48维 + 原始14维
            actions_62d_converted = np.concatenate([actions_48d_robot, actions_14d_robot], axis=1)

            # 原地覆盖数据
            del f['actions_62d']
            dset = f.create_dataset('actions_62d', data=actions_62d_converted)

            # 添加转换标记
            dset.attrs['coordinate_system'] = 'pybullet_unified'
            dset.attrs['description'] = '48D (hand) and 14D (robot arm) actions, both in PyBullet coordinate system.'
            dset.attrs['conversion_timestamp'] = datetime.now().isoformat()
            dset.attrs['original_coordinate_system'] = 'egodex'

        return f"成功转换: {file_path}"

    except Exception as e:
        return f"转换失败: {file_path} - 错误: {e}"

def process_directory_inplace(input_dir: str, num_workers: int = None):
    """
    使用多进程原地处理整个目录

    Args:
        input_dir: 输入目录
        num_workers: 工作进程数
    """
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在 '{input_dir}'")
        return

    # 设置默认进程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"🚀 开始原地转换目录: {input_dir}")
    print(f"🛠️ 使用 {num_workers} 个工作进程")
    print("⚠️  警告: 此操作将直接修改原始文件!")

    # 递归查找所有 HDF5 文件
    all_input_files = glob.glob(os.path.join(input_dir, '**', '*.hdf5'), recursive=True)
    if not all_input_files:
        print("❌ 在指定输入目录中未找到任何 HDF5 文件。")
        return

    print(f"📊 找到 {len(all_input_files)} 个 HDF5 文件。")

    # 使用多进程池进行处理
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file_inplace_worker, all_input_files),
            total=len(all_input_files),
            desc="原地转换进度"
        ))

    # 统计结果
    success_count = sum(1 for r in results if r.startswith('成功转换'))
    skipped_converted = sum(1 for r in results if '已转换' in r)
    skipped_no_data = sum(1 for r in results if '无 \'actions_62d\'' in r)
    failed_count = sum(1 for r in results if '转换失败' in r or '失败已恢复' in r or '失败且恢复失败' in r)

    print("\n✅ 原地转换完成！")
    print(f"   - 成功转换: {success_count}")
    print(f"   - 已跳过 (已转换): {skipped_converted}")
    print(f"   - 已跳过 (无数据): {skipped_no_data}")
    print(f"   - 转换失败: {failed_count}")

    if failed_count > 0:
        print("\n🔍 失败文件列表:")
        for r in results:
            if '转换失败' in r:
                print(f"   - {r}")

def main():
    parser = argparse.ArgumentParser(description="Egodex HDF5文件坐标系原地转换工具")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含Egodex HDF5文件的数据根目录')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='使用的进程数')

    args = parser.parse_args()

    print("="*70)
    print("🎯 Egodex坐标系原地转换工具 (V3)")
    print("="*70)
    print("🔄 功能: 直接在原文件中转换前48维数据坐标系")
    print("📊 数据结构: 48维(手部) + 14维(机械臂) = 62维")
    print("⚙️  坐标系: Egodex → PyBullet")
    print("="*70)

    # 处理目录
    process_directory_inplace(args.input_dir, args.num_workers)

if __name__ == '__main__':
    main()