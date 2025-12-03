#!/usr/bin/env python3
"""
根据人类手部动作计算虚拟机器人末端执行器位姿脚本
递归遍历Egodex-full目录结构，生成对应的eef_14d文件
"""

import numpy as np
import h5py
import os
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple
import glob
import multiprocessing
import argparse
from tqdm import tqdm


class OptimizedHumanToEEFMapper:
    """人类手部到机器人末端执行器位姿映射器"""

    def __init__(self):
        self.coordinate_transform_matrix = None
        self.human_start_pos = {}
        self.robot_start_pos = {}
        self.initial_poses = {}

    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """计算坐标变换校准"""
        # 记录人类在Egodex坐标系下的起始位置
        self.human_start_pos = {}
        self.human_start_pos['left'] = human_init_transforms['left'][:3, 3]
        self.human_start_pos['right'] = human_init_transforms['right'][:3, 3]

        # 设置机器人在PyBullet坐标系下的起始位置（与原脚本保持一致）
        # 左臂初始位置
        self.robot_start_pos['left'] = np.array([0.3, 0.2, 0.5])  # 左臂自然下垂位置
        # 右臂初始位置
        self.robot_start_pos['right'] = np.array([0.3, -0.2, 0.5])  # 右臂自然下垂位置

        # 设置坐标变换矩阵（与原脚本保持一致）
        self.coordinate_transform_matrix = np.array([
            [0, 0, -1],  # 人类-Z -> 机器人X
            [-1, 0, 0],  # 人类-X -> 机器人Y
            [0, 1, 0]    # 人类Y -> 机器人Z
        ])

        # 设置初始姿态（四元数表示，与原脚本初始姿态对应）
        self.initial_poses = {
            'left': {'orientation': np.array([0.0, 0.0, 0.0, 1.0])},  # 左臂初始朝向
            'right': {'orientation': np.array([0.0, 0.0, 0.0, 1.0])}  # 右臂初始朝向
        }

        self.human_init_transforms = human_init_transforms

    def transform_human_to_robot_eef(self, human_transform: np.ndarray, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """变换人类手部位姿到机器人末端执行器位姿"""
        # 位置变换
        human_current_pos = human_transform[:3, 3]
        human_delta = human_current_pos - self.human_start_pos[arm]
        robot_delta = self.coordinate_transform_matrix @ human_delta
        robot_target_pos = self.robot_start_pos[arm] + robot_delta

        # 姿态变换
        human_rot_matrix = human_transform[:3, :3]
        robot_rot_matrix = self.coordinate_transform_matrix @ human_rot_matrix
        # 转换为四元数 (scipy默认使用 [x, y, z, w] 格式)
        robot_quat = Rotation.from_matrix(robot_rot_matrix).as_quat()

        return robot_target_pos, robot_quat


def process_hdf5_file_final(input_path: str, output_path: str) -> bool:
    """处理HDF5文件，生成末端执行器位姿数据"""
    try:
        mapper = OptimizedHumanToEEFMapper()

        with h5py.File(input_path, 'r') as f:
            # 读取手部变换数据
            left_hand_transforms = f['/transforms/leftThumbIntermediateBase'][:]
            right_hand_transforms = f['/transforms/rightThumbIntermediateBase'][:]

            num_frames = len(left_hand_transforms)

            # 用第0帧计算坐标变换
            human_init_transforms = {
                'left': left_hand_transforms[0],
                'right': right_hand_transforms[0]
            }
            mapper.calculate_coordinate_transform(human_init_transforms)

            # 预分配14维数组 (左臂7维 + 右臂7维)
            eef_14d = np.zeros((num_frames, 14), dtype=np.float32)

            # 逐帧处理
            for frame_idx in range(num_frames):
                try:
                    # 读取手部变换
                    left_transform = left_hand_transforms[frame_idx]
                    right_transform = right_hand_transforms[frame_idx]

                    # 坐标变换得到末端执行器位姿
                    left_pos, left_quat = mapper.transform_human_to_robot_eef(left_transform, 'left')
                    right_pos, right_quat = mapper.transform_human_to_robot_eef(right_transform, 'right')

                    # 组装14维末端执行器位姿
                    # 左臂：位置xyz + 四元数xyzw
                    # 右臂：位置xyz + 四元数xyzw
                    current_eef = np.concatenate([
                        left_pos,           # 0-2: 左臂位置 [x, y, z]
                        left_quat,          # 3-6: 左臂四元数 [qx, qy, qz, qw]
                        right_pos,          # 7-9: 右臂位置 [x, y, z]
                        right_quat          # 10-13: 右臂四元数 [qx, qy, qz, qw]
                    ])

                    eef_14d[frame_idx] = current_eef

                except Exception as e:
                    # 异常情况使用上一帧数据
                    if frame_idx > 0:
                        eef_14d[frame_idx] = eef_14d[frame_idx - 1]
                    else:
                        # 第一帧异常时使用默认值
                        default_eef = np.array([
                            0.3, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0,  # 左臂默认位姿
                            0.3, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0  # 右臂默认位姿
                        ])
                        eef_14d[frame_idx] = default_eef
                    continue

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存结果
            with h5py.File(output_path, 'w') as out_f:
                # 保存14维末端执行器位姿数据
                out_f.create_dataset('eef_14d', data=eef_14d)

                # 保存必要的元数据
                out_f.attrs['source_file'] = input_path
                out_f.attrs['total_frames'] = num_frames
                out_f.attrs['coordinate_transform_matrix'] = mapper.coordinate_transform_matrix.flatten()
                out_f.attrs['left_robot_start_pos'] = mapper.robot_start_pos['left']
                out_f.attrs['right_robot_start_pos'] = mapper.robot_start_pos['right']

            return True

    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")
        return False


def find_hdf5_files_recursive(root_dir, exclude_dirs=None):
    """
    递归查找所有HDF5文件，并可以排除指定的顶层子目录。

    Args:
        root_dir (str): 搜索的根目录。
        exclude_dirs (list, optional): 要排除的顶层子目录名称列表。默认为None。
    """
    if exclude_dirs is None:
        exclude_dirs = []

    hdf5_files = []
    print(f"🔍 搜索HDF5文件于 '{root_dir}', 排除目录: {exclude_dirs}")

    for root, dirs, files in os.walk(root_dir):
        # 排除指定目录
        if root == root_dir:
            dirs_to_remove = [d for d in dirs if d in exclude_dirs]
            for d in dirs_to_remove:
                print(f"  -> 忽略目录: {d}")
                dirs.remove(d)

        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))

    return hdf5_files


def map_input_to_output_path(input_path, input_root, output_root):
    """将输入路径映射到输出路径"""
    # 计算相对路径
    relative_path = os.path.relpath(input_path, input_root)
    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(relative_path))[0]
    # 构造输出文件名
    output_filename = f"{filename}.hdf5"
    # 构造输出路径
    output_path = os.path.join(output_root, os.path.dirname(relative_path), output_filename)
    return output_path


def worker_process(file_list, worker_id, input_root, output_root, progress_queue):
    """工作进程"""
    try:
        success_count = 0
        for input_file in file_list:
            output_file = map_input_to_output_path(input_file, input_root, output_root)

            success = process_hdf5_file_final(input_file, output_file)
            if success:
                success_count += 1

            # 向队列发送进度信号
            progress_queue.put(1)

        return success_count

    except Exception:
        return 0


def main_final_process():
    """主函数"""
    parser = argparse.ArgumentParser(description='末端执行器14D位姿预计算')
    parser.add_argument('--input_root', type=str, default="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/yinzhenhan/data/dataset/Egodex-full",
                       help='输入根目录')
    parser.add_argument('--output_root', type=str, required=True, help='输出根目录')
    parser.add_argument('--num_processes', type=int, default=10, help='进程数量')

    args = parser.parse_args()

    # 定义要排除的目录（与原脚本保持一致）
    EXCLUDE_DIRECTORIES = ['extra', 'part3', 'part4', 'part1', 'test']

    if not os.path.exists(args.input_root):
        print(f"❌ 输入根目录不存在: {args.input_root}")
        return

    print(f"🚀 开始递归处理目录: {args.input_root}")
    print(f"📁 输出根目录: {args.output_root}")

    # 递归查找所有HDF5文件
    print("🔍 搜索HDF5文件...")
    hdf5_files = find_hdf5_files_recursive(args.input_root, exclude_dirs=EXCLUDE_DIRECTORIES)

    if not hdf5_files:
        print(f"❌ 在目录 {args.input_root} 中没有找到HDF5文件")
        return

    total_files = len(hdf5_files)
    print(f"📊 找到 {total_files} 个HDF5文件")

    # 优化：使用实际需要的进程数
    actual_processes = min(args.num_processes, total_files)
    files_per_process = total_files // actual_processes
    file_lists = []

    for i in range(actual_processes):
        start_idx = i * files_per_process
        if i == actual_processes - 1:
            end_idx = total_files
        else:
            end_idx = start_idx + files_per_process

        file_lists.append(hdf5_files[start_idx:end_idx])

    print(f"📊 分配文件到 {actual_processes} 个工作进程")

    # 使用 Manager.Queue 和 tqdm
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    with multiprocessing.Pool(processes=actual_processes) as pool:
        results = []
        for i, file_list in enumerate(file_lists):
            if file_list:
                print(f"👷 启动工作进程 {i}，处理 {len(file_list)} 个文件")
                result = pool.apply_async(worker_process, (file_list, i, args.input_root, args.output_root, progress_queue))
                results.append(result)

        # 创建并管理进度条
        with tqdm(total=total_files, desc="处理 HDF5 文件") as pbar:
            processed_count = 0
            while processed_count < total_files:
                # 从队列中获取一个完成信号（阻塞直到有信号）
                progress_queue.get()
                processed_count += 1
                pbar.update(1)  # 更新进度条

        print("⏳ 所有文件处理完毕，正在收集最终结果...")
        total_success = sum(result.get() for result in results)

    print(f"✅ 递归处理完成!")
    print(f"   总文件数: {total_files}")
    print(f"   成功处理: {total_success}")
    print(f"   失败文件: {total_files - total_success}")
    print(f"   成功率: {total_success/total_files*100:.1f}%")
    print(f"📁 输出目录: {args.output_root}")

    return total_success


if __name__ == "__main__":
    main_final_process()