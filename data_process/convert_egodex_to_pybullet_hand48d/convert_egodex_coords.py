#!/usr/bin/env python3
"""
离线坐标系转换脚本 (V2 - 保存到新目录)

功能：
- 递归遍历指定的 Egodex 数据目录。
- 读取每个 HDF5 文件中的 'actions_62d' 数据集。
- 将其中前48维（人手动作）从 Egodex 坐标系转换到机器人（PyBullet）坐标系。
- 将包含转换后数据的 *完整新文件* 保存到指定的输出目录，并保持原始的目录结构。
- 使用多进程加速。
parser.add_argument('--input_dir', type=str, required=True,
                    help='包含原始Egodex HDF5文件的数据根目录')
parser.add_argument('--output_dir', type=str, required=True,
                    help='保存转换后HDF5文件的根目录')
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

def process_file(paths: tuple):
    """
    处理单个HDF5文件，读取源文件，转换数据，并写入到新的目标文件。
    Args:
        paths (tuple): 一个包含 (input_path, output_path) 的元组。
    """
    input_path, output_path = paths
    
    try:
        # 如果目标文件已存在，可以选择跳过以支持断点续传
        if os.path.exists(output_path):
            return f"Skipped (output exists): {output_path}"

        with h5py.File(input_path, 'r') as f_in:
            if 'actions_62d' not in f_in:
                return f"Skipped (no 'actions_62d'): {input_path}"
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with h5py.File(output_path, 'w') as f_out:
                # 复制所有原始数据到新文件
                for key, item in f_in.items():
                    f_in.copy(key, f_out)
                for key, item in f_in.attrs.items():
                    f_out.attrs[key] = item

                # 读取需要转换的数据
                actions_62d_original = f_in['actions_62d'][:]
                
                # 分离
                actions_48d_egodex = actions_62d_original[:, :48]
                actions_14d_robot = actions_62d_original[:, 48:]
                
                # 转换
                actions_48d_robot = convert_48d_coords(actions_48d_egodex)
                
                # 组合
                actions_62d_unified = np.concatenate([actions_48d_robot, actions_14d_robot], axis=1)
                
                # 在新文件中 *更新* actions_62d 数据集
                del f_out['actions_62d']
                dset = f_out.create_dataset('actions_62d', data=actions_62d_unified)
                
                # 添加属性标记
                dset.attrs['coordinate_system'] = 'pybullet_unified'
                dset.attrs['description'] = '48D (hand) and 14D (robot arm) actions, both in PyBullet coordinate system.'
                dset.attrs['source_egodex_file'] = input_path
            
            return f"Success: {input_path} -> {output_path}"
            
    except Exception as e:
        return f"Failed: {input_path} with error: {e}"

def main():
    parser = argparse.ArgumentParser(description="离线转换Egodex HDF5文件坐标系并保存到新目录")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含原始Egodex HDF5文件的数据根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='保存转换后HDF5文件的根目录')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(),
                        help='使用的进程数')
    args = parser.parse_args()

    print(f"🚀 开始转换目录: {args.input_dir}")
    print(f"📁 输出到新目录: {args.output_dir}")
    print(f"🛠️ 使用 {args.num_workers} 个工作进程")
    
    # 递归查找所有 HDF5 文件
    all_input_files = glob.glob(os.path.join(args.input_dir, '**', '*.hdf5'), recursive=True)
    if not all_input_files:
        print("❌ 在指定输入目录中未找到任何 HDF5 文件。")
        return
        
    print(f"📊 找到 {len(all_input_files)} 个 HDF5 文件。")

    # --- 构造输入输出路径对 ---
    path_pairs = []
    for input_path in all_input_files:
        # 计算相对路径
        relative_path = os.path.relpath(input_path, args.input_dir)
        # 构造输出路径
        output_path = os.path.join(args.output_dir, relative_path)
        path_pairs.append((input_path, output_path))
    
    # 使用多进程池进行处理
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, path_pairs), 
                            total=len(path_pairs), desc="转换进度"))

    # 统计结果
    success_count = sum(1 for r in results if r.startswith('Success'))
    skipped_exists = sum(1 for r in results if 'output exists' in r)
    skipped_no_data = sum(1 for r in results if 'no \'actions_62d\'' in r)
    failed_count = sum(1 for r in results if r.startswith('Failed'))
    
    print("\n✅ 转换完成！")
    print(f"   - 成功转换: {success_count}")
    print(f"   - 已跳过 (输出已存在): {skipped_exists}")
    print(f"   - 已跳过 (无数据): {skipped_no_data}")
    print(f"   - 转换失败: {failed_count}")

    if failed_count > 0:
        print("\n🔍 失败文件列表:")
        for r in results:
            if r.startswith('Failed'):
                print(f"   - {r}")

if __name__ == '__main__':
    main()
    