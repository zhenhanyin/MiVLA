#!/usr/bin/env python3
"""
Egodex EEF坐标系转换脚本

功能：
- 读取transforms/leftThumbIntermediateBase和transforms/rightThumbIntermediateBase (帧数,4,4)
- 转换为位姿：前三维xyz，后四维四元数qw qx qy qz
- 坐标系变换到右手坐标系
- 保存为eef_14d数据集（左右拇指各7维）
"""

import h5py
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import multiprocessing

# 坐标系变换矩阵：Egodex -> 右手坐标系
R_egodex_to_right = np.array([
    [ 0,  0, -1],
    [-1,  0,  0],
    [ 0,  1,  0]
])

def transform_matrix_to_pose(transform_matrix):
    """4x4变换矩阵转位姿 [x,y,z,qw,qx,qy,qz]"""
    if transform_matrix.ndim == 2:
        transform_matrix = transform_matrix.reshape(1, 4, 4)

    N = transform_matrix.shape[0]
    poses = np.zeros((N, 7))

    for i in range(N):
        rot_matrix = transform_matrix[i, :3, :3]
        translation = transform_matrix[i, :3, 3]
        quaternion = rotation_matrix_to_quaternion(rot_matrix)
        poses[i] = np.concatenate([translation, quaternion])

    return poses.squeeze()

def rotation_matrix_to_quaternion(R):
    """旋转矩阵转四元数 [w,x,y,z]"""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    w, x, y, z = q / np.linalg.norm(q)

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def convert_pose_coords(pose_7d):
    """位姿坐标系转换"""
    if pose_7d.ndim == 1:
        pose_7d = pose_7d.reshape(1, -1)

    N = pose_7d.shape[0]
    converted = np.zeros_like(pose_7d)

    for i in range(N):
        pos = pose_7d[i, :3]
        quat = pose_7d[i, 3:]

        # 位置转换
        pos_converted = R_egodex_to_right @ pos

        # 四元数转换（通过旋转矩阵）
        R = quaternion_to_rotation_matrix(quat)
        R_converted = R_egodex_to_right @ R
        quat_converted = rotation_matrix_to_quaternion(R_converted)

        converted[i] = np.concatenate([pos_converted, quat_converted])

    return converted.squeeze()

def process_file(paths):
    """处理单个文件"""
    input_path, output_path = paths

    try:
        with h5py.File(input_path, 'r') as f_in:
            left_key = 'transforms/leftThumbIntermediateBase'
            right_key = 'transforms/rightThumbIntermediateBase'

            if left_key not in f_in or right_key not in f_in:
                return f"No EEF data: {input_path}"

            # 读取变换矩阵
            left_transforms = f_in[left_key][:]
            right_transforms = f_in[right_key][:]

            # 转位姿
            left_poses = transform_matrix_to_pose(left_transforms)
            right_poses = transform_matrix_to_pose(right_transforms)

            # 坐标系转换
            left_poses = convert_pose_coords(left_poses)
            right_poses = convert_pose_coords(right_poses)

            # 组合14维
            eef_14d = np.concatenate([left_poses, right_poses], axis=1)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存到已有文件
            with h5py.File(output_path, 'a') as f_out:  # 'a'模式追加写入
                # 添加eef_14d数据集
                if 'eef_14d' in f_out:
                    del f_out['eef_14d']  # 删除已存在的数据集
                dset_eef = f_out.create_dataset('eef_14d', data=eef_14d)
                dset_eef.attrs['coordinate_system'] = 'right_hand'

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
                    dset_action.attrs['description'] = '62D actions + 14D EEF poses'
                else:
                    raise ValueError(f"No actions_62d found in {output_path}")

            return f"Added EEF and actions_76d: {output_path}"

    except Exception as e:
        return f"Failed: {input_path} - {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-full", help='Egodex数据目录')
    parser.add_argument('--output_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain", help='输出目录')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    args = parser.parse_args()

    # 查找HDF5文件
    files = glob.glob(os.path.join(args.input_dir, '**', '*.hdf5'), recursive=True)
    print(f"找到 {len(files)} 个文件")

    # 构建路径对
    path_pairs = []
    for f in files:
        rel_path = os.path.relpath(f, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        path_pairs.append((f, out_path))

    # 多进程处理
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, path_pairs),
                           total=len(path_pairs), desc="转换进度"))

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