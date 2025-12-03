#!/usr/bin/env python3
"""
根据人类手部动作计算虚拟机器人末端执行器位姿脚本（相对位移法）
递归遍历Egodex-full目录结构，在目标文件中添加eef_14d和actions_76d数据集
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
        self.robot_start_pos['left'] = np.array([ 0.350, 0.217, 0.945])  # 左臂自然下垂位置
        # 右臂初始位置
        self.robot_start_pos['right'] = np.array([0.350,-0.224,0.945])  # 右臂自然下垂位置

        # 设置坐标变换矩阵（与原脚本保持一致）
        self.coordinate_transform_matrix = np.array([
            [0, 0, -1],  # 人类-Z -> 机器人X
            [-1, 0, 0],  # 人类-X -> 机器人Y
            [0, 1, 0]    # 人类Y -> 机器人Z
        ])

        # 设置初始姿态（四元数表示，与原脚本初始姿态对应）
        self.initial_poses = {
            'left': {'orientation': np.array([1.0, 0.0, 0.0, 0.0])},  # 左臂初始朝向 [w,x,y,z]
            'right': {'orientation': np.array([1.0, 0.0, 0.0, 0.0])}  # 右臂初始朝向 [w,x,y,z]
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
        # 转换为四元数，并转换为 [w, x, y, z] 格式
        quat_xyzw = Rotation.from_matrix(robot_rot_matrix).as_quat()  # [x, y, z, w]
        robot_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]

        return robot_target_pos, robot_quat


def process_file(paths):
    """处理单个文件"""
    input_path, output_path = paths

    try:
        mapper = OptimizedHumanToEEFMapper()

        with h5py.File(input_path, 'r') as f_in:
            left_key = 'transforms/leftThumbIntermediateBase'
            right_key = 'transforms/rightThumbIntermediateBase'

            if left_key not in f_in or right_key not in f_in:
                return f"No EEF data: {input_path}"

            # 读取手部变换数据
            left_hand_transforms = f_in[left_key][:]
            right_hand_transforms = f_in[right_key][:]
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
                    # 左臂：位置xyz + 四元数wxyz
                    # 右臂：位置xyz + 四元数wxyz
                    current_eef = np.concatenate([
                        left_pos,           # 0-2: 左臂位置 [x, y, z]
                        left_quat,          # 3-6: 左臂四元数 [qw, qx, qy, qz]
                        right_pos,          # 7-9: 右臂位置 [x, y, z]
                        right_quat          # 10-13: 右臂四元数 [qw, qx, qy, qz]
                    ])

                    eef_14d[frame_idx] = current_eef

                except Exception as e:
                    # 异常情况使用上一帧数据
                    if frame_idx > 0:
                        eef_14d[frame_idx] = eef_14d[frame_idx - 1]
                    else:
                        # 第一帧异常时使用默认值
                        default_eef = np.array([
                           0.350, 0.217, 0.945, 1.0, 0.0, 0.0, 0.0,  # 左臂默认位姿 [x,y,z,qw,qx,qy,qz]
                            0.350,-0.224,0.944, 1.0, 0.0, 0.0, 0.0  # 右臂默认位姿 [x,y,z,qw,qx,qy,qz]
                        ])
                        eef_14d[frame_idx] = default_eef
                    continue

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存到已有文件
            with h5py.File(output_path, 'a') as f_out:  # 'a'模式追加写入
                # 添加eef_14d数据集
                if 'eef_14d' in f_out:
                    del f_out['eef_14d']  # 删除已存在的数据集
                dset_eef = f_out.create_dataset('eef_14d', data=eef_14d)
                dset_eef.attrs['coordinate_system'] = 'right_hand'
                dset_eef.attrs['method'] = 'relative_displacement'

                # 添加actions_76d数据集
                if 'actions_62d' in f_out:
                    actions_62d = f_out['actions_62d'][:]
                    # 检查帧数是否匹配
                    if actions_62d.shape[0] != eef_14d.shape[0]:
                        raise ValueError(f"Frame count mismatch: actions_62d has {actions_62d.shape[0]} frames, eef_14d has {eef_14d.shape[0]} frames in {output_path}")

                    actions_76d = np.concatenate([actions_62d, eef_14d], axis=1)

                    if 'actions_76d' in f_out:
                        del f_out['actions_76d']  # 删除已存在的数据集
                    dset_action = f_out.create_dataset('actions_76d', data=actions_76d)
                    dset_action.attrs['description'] = '62D actions + 14D EEF poses (relative displacement method)'
                else:
                    raise ValueError(f"No actions_62d found in {output_path}")

            return f"Added EEF and actions_76d: {output_path}"

    except Exception as e:
        return f"Failed: {input_path} - {e}"


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






def main():
    parser = argparse.ArgumentParser(description='末端执行器14D位姿预计算（相对位移法）')
    parser.add_argument('--input_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-full", help='Egodex数据目录')
    parser.add_argument('--output_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain", help='输出目录')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    parser.add_argument('--exclude_dirs', nargs='+', default=[], help='排除的目录')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return

    print(f"�� 开始处理目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📁 排除目录: {args.exclude_dirs}")

    # 递归查找HDF5文件
    files = find_hdf5_files_recursive(args.input_dir, exclude_dirs=args.exclude_dirs)
    if not files:
        print("❌ 未找到任何HDF5文件")
        return

    print(f"📊 找到 {len(files)} 个HDF5文件")

    # 构建路径对
    path_pairs = []
    for f in files:
        rel_path = os.path.relpath(f, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        path_pairs.append((f, out_path))

    # 多进程处理
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, path_pairs),
                           total=len(path_pairs), desc="转换进度（相对位移法）"))

    # 统计
    success = sum(1 for r in results if r.startswith('Added EEF'))
    failed = [r for r in results if r.startswith('Failed') or r.startswith('No EEF data')]

    print(f"完成: 成功 {success}, 失败 {len(failed)}, 总计 {len(files)}")

    if args.verbose and success > 0:
        print("\n成功处理的文件:")
        for r in results:
            if r.startswith('Added EEF'):
                print(f"  {r}")

    if failed:
        print("\n失败详情:")
        for f in failed:
            print(f"  {f}")


if __name__ == '__main__':
    main()