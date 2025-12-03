#!/usr/bin/env python3
"""
RoboTwin到人手48维数据生成脚本

功能：
1. 从RoboTwin HDF5文件读取机械臂数据
2. 根据算法计算虚拟人手的48维数据
3. 将48维数据保存到新的HDF5文件中

48维数据结构：
- 左手 (24维) + 右手 (24维)
- 每只手24维 = 手腕位置(3D) + 手腕姿态(6D) + 五个指尖位置(15D)
"""

import numpy as np
import h5py
import os
import argparse
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 手部模型定义
# =============================================================================

class HandModel:
    """简化的人手模型"""

    def __init__(self):
        # 手部解剖学参数 (单位: 米)
        self.palm_width = 0.085  # 手掌宽度
        self.palm_length = 0.095  # 手掌长度

        # 手腕相对于拇指根关节的位置 (在拇指根关节局部坐标系中)
        self.wrist_offset_from_thumb_base = np.array([-0.050, 0.030, 0.000])  # 向后50mm，向右30mm

        # 手腕到各手指根关节的位置 (在手腕局部坐标系中，XOY平面)
        self.finger_base_offsets = {
            'thumb': np.array([0.020, -0.040, 0.000]),      # 拇指：前方20mm，左侧40mm
            'index': np.array([0.080, -0.015, 0.000]),      # 食指：前方80mm，左侧15mm
            'middle': np.array([0.090, 0.000, 0.000]),      # 中指：前方90mm，中心位置
            'ring': np.array([0.085, 0.015, 0.000]),        # 无名指：前方85mm，右侧15mm
            'pinky': np.array([0.075, 0.030, 0.000])        # 小指：前方75mm，右侧30mm
        }

        # 各手指长度 (从根关节到指尖，沿正X轴方向)
        self.finger_lengths = {
            'thumb': 0.060,     # 拇指长度：60mm
            'index': 0.075,     # 食指长度：75mm
            'middle': 0.085,    # 中指长度：85mm (最长)
            'ring': 0.080,      # 无名指长度：80mm
            'pinky': 0.065      # 小指长度：65mm
        }

    def thumb_base_to_wrist(self, thumb_base_pos, thumb_base_quat):
        """
        根据拇指根关节位姿计算手腕位姿

        Args:
            thumb_base_pos: 拇指根关节位置 [x, y, z]
            thumb_base_quat: 拇指根关节四元数 [qw, qx, qy, qz]

        Returns:
            wrist_pos, wrist_quat: 手腕位置和四元数
        """
        # 将四元数转换为旋转矩阵
        rotation = R.from_quat([thumb_base_quat[1], thumb_base_quat[2],
                               thumb_base_quat[3], thumb_base_quat[0]])
        rot_matrix = rotation.as_matrix()

        # 手腕在拇指根关节局部坐标系中的偏移
        world_offset = rot_matrix @ self.wrist_offset_from_thumb_base

        # 计算手腕位置
        wrist_pos = thumb_base_pos + world_offset

        # 手腕姿态：与拇指根关节相同（整个手部朝向一致）
        wrist_quat = thumb_base_quat.copy()

        return wrist_pos, wrist_quat

    def wrist_to_fingertips_dynamic(self, wrist_pos, wrist_quat, gripper_value, hand_side='left'):
        """
        根据手腕位姿和夹爪开合值动态计算5个手指指尖位置

        Args:
            wrist_pos: 手腕位置 [x, y, z]
            wrist_quat: 手腕四元数 [qw, qx, qy, qz]
            gripper_value: 夹爪开合值 (0=完全闭合, 1=完全张开)
            hand_side: 手侧 ('left' 或 'right')

        Returns:
            Dict[str, np.ndarray]: 各手指指尖位置
        """
        # 将四元数转换为旋转矩阵
        rotation = R.from_quat([wrist_quat[1], wrist_quat[2],
                               wrist_quat[3], wrist_quat[0]])
        rot_matrix = rotation.as_matrix()

        fingertips = {}

        # 计算各手指根关节位置
        finger_base_positions = {}
        # 右手需要镜像处理手指偏移参数
        finger_base_offsets = self.finger_base_offsets.copy()
        if hand_side == 'right':
            # Y轴取反，实现左右对称
            for finger_name, base_offset in finger_base_offsets.items():
                finger_base_offsets[finger_name] = base_offset.copy()
                finger_base_offsets[finger_name][1] *= -1

        for finger_name, base_offset in finger_base_offsets.items():
            finger_base_pos = wrist_pos + rot_matrix @ base_offset
            finger_base_positions[finger_name] = finger_base_pos

        # === 动态手指位置计算 ===

        # 1. 拇指：根据夹爪开合值从根关节向外或向内移动
        thumb_direction_local = np.array([1.0, 0.0, 0.0])  # 正X轴方向
        world_thumb_direction = rot_matrix @ thumb_direction_local

        # 拇指动态长度：基础长度 + 夹爪影响的伸展
        base_thumb_length = self.finger_lengths['thumb']
        thumb_extension_factor = 0.6 + gripper_value * 0.8
        dynamic_thumb_length = base_thumb_length * thumb_extension_factor

        thumb_tip_pos = finger_base_positions['thumb'] + world_thumb_direction * dynamic_thumb_length
        fingertips['thumb'] = thumb_tip_pos

        # 2. 食指：主要跟随对象，根据拇指-食指距离动态调整
        index_direction_local = np.array([1.0, 0.0, 0.0])  # 正X轴方向
        world_index_direction = rot_matrix @ index_direction_local

        # 食指动态长度：基础长度 + 夹爪影响的伸展
        base_index_length = self.finger_lengths['index']
        index_extension_factor = 0.5 + gripper_value * 0.8
        dynamic_index_length = base_index_length * index_extension_factor

        index_tip_pos = finger_base_positions['index'] + world_index_direction * dynamic_index_length
        fingertips['index'] = index_tip_pos

        # 3. 中指、无名指、小指：跟随食指运动，保持自然手势
        follower_fingers = ['middle', 'ring', 'pinky']
        base_lengths = [self.finger_lengths['middle'], self.finger_lengths['ring'], self.finger_lengths['pinky']]

        for i, finger_name in enumerate(follower_fingers):
            finger_direction_local = np.array([1.0, 0.0, 0.0])  # 正X轴方向
            world_finger_direction = rot_matrix @ finger_direction_local

            # 跟随食指的伸展程度，但有独立的缩放因子
            base_length = base_lengths[i]

            # 动态长度：基础长度 + 跟随食指的伸展
            follower_extension_factor = 0.4 + gripper_value * 0.7  # 略小于食指的伸展幅度
            dynamic_finger_length = base_length * follower_extension_factor

            finger_tip_pos = finger_base_positions[finger_name] + world_finger_direction * dynamic_finger_length
            fingertips[finger_name] = finger_tip_pos

        return fingertips

# =============================================================================
# 数据处理器
# =============================================================================

def rotation_matrix_to_6d(rot_matrix):
    """
    将3x3旋转矩阵转换为6D表示（前两列拼接）

    Args:
        rot_matrix: 3x3旋转矩阵

    Returns:
        6D表示向量 [r11, r21, r31, r12, r22, r32]
    """
    # 取前两列并展平
    col1 = rot_matrix[:, 0]  # 第一列
    col2 = rot_matrix[:, 1]  # 第二列
    return np.concatenate([col1, col2])

def compute_hand_48d_data(left_endpose, left_gripper, right_endpose, right_gripper, hand_model):
    """
    计算单帧的48维人手数据

    Args:
        left_endpose: 左臂末端位姿 [x, y, z, qw, qx, qy, qz]
        left_gripper: 左夹爪开合值
        right_endpose: 右臂末端位姿 [x, y, z, qw, qx, qy, qz]
        right_gripper: 右夹爪开合值
        hand_model: 手部模型对象

    Returns:
        48维数据向量
    """
    # 左手处理
    left_thumb_base_pos = left_endpose[:3]
    left_thumb_base_quat = left_endpose[3:7]
    left_wrist_pos, left_wrist_quat = hand_model.thumb_base_to_wrist(left_thumb_base_pos, left_thumb_base_quat)

    # 左手动态指尖位置
    left_fingertips = hand_model.wrist_to_fingertips_dynamic(left_wrist_pos, left_wrist_quat, left_gripper, 'left')

    # 左手腕姿态 (6D)
    left_rotation = R.from_quat([left_wrist_quat[1], left_wrist_quat[2], left_wrist_quat[3], left_wrist_quat[0]])
    left_rot_matrix = left_rotation.as_matrix()
    left_wrist_pose_6d = rotation_matrix_to_6d(left_rot_matrix)

    # 左手数据 (24维)
    left_hand_data = np.concatenate([
        left_wrist_pos,  # 3D: 手腕位置
        left_wrist_pose_6d,  # 6D: 手腕姿态
        left_fingertips['thumb'],  # 3D: 拇指指尖
        left_fingertips['index'],  # 3D: 食指指尖
        left_fingertips['middle'], # 3D: 中指指尖
        left_fingertips['ring'],   # 3D: 无名指指尖
        left_fingertips['pinky']   # 3D: 小指指尖
    ])

    # 右手处理
    right_thumb_base_pos = right_endpose[:3]
    right_thumb_base_quat = right_endpose[3:7]

    # 右手需要镜像处理：修改手腕偏移参数
    right_wrist_offset = hand_model.wrist_offset_from_thumb_base.copy()
    right_wrist_offset[1] *= -1  # Y轴取反，实现左右镜像

    # 计算右手腕位置
    right_rotation = R.from_quat([right_thumb_base_quat[1], right_thumb_base_quat[2],
                                   right_thumb_base_quat[3], right_thumb_base_quat[0]])
    right_rot_matrix = right_rotation.as_matrix()
    right_world_offset = right_rot_matrix @ right_wrist_offset
    right_wrist_pos = right_thumb_base_pos + right_world_offset
    right_wrist_quat = right_thumb_base_quat.copy()

    # 右手动态指尖位置
    right_fingertips = hand_model.wrist_to_fingertips_dynamic(right_wrist_pos, right_wrist_quat, right_gripper, 'right')

    # 右手腕姿态 (6D)
    right_rotation = R.from_quat([right_wrist_quat[1], right_wrist_quat[2], right_wrist_quat[3], right_wrist_quat[0]])
    right_rot_matrix = right_rotation.as_matrix()
    right_wrist_pose_6d = rotation_matrix_to_6d(right_rot_matrix)

    # 右手数据 (24维)
    right_hand_data = np.concatenate([
        right_wrist_pos,  # 3D: 手腕位置
        right_wrist_pose_6d,  # 6D: 手腕姿态
        right_fingertips['thumb'],  # 3D: 拇指指尖
        right_fingertips['index'],  # 3D: 食指指尖
        right_fingertips['middle'], # 3D: 中指指尖
        right_fingertips['ring'],   # 3D: 无名指指尖
        right_fingertips['pinky']   # 3D: 小指指尖
    ])

    # 合并为48维数据
    hand_48d_data = np.concatenate([left_hand_data, right_hand_data])

    return hand_48d_data

def convert_output_path(input_path):
    """
    转换输出路径：从 Robotwin_gen_48d_process 到 Robotwin-gen-human-full

    Args:
        input_path: 输入文件路径

    Returns:
        输出文件路径
    """
    # 替换根目录名
    output_path = input_path.replace('Robotwin_gen_48d_process', 'Robotwin-gen-human-full')
    return output_path

def ensure_directory_exists(file_path):
    """
    确保目录存在，如果不存在则创建

    Args:
        file_path: 文件路径
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def process_single_file(input_file, output_file=None):
    """
    处理单个HDF5文件

    Args:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径（可选）
    """
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在 '{input_file}'")
        return False

    # 如果没有指定输出文件，自动生成路径
    if output_file is None:
        output_file = convert_output_path(input_file)

    # 确保输出目录存在
    ensure_directory_exists(output_file)

    print(f"🚀 开始处理文件:")
    print(f"   输入: {input_file}")
    print(f"   输出: {output_file}")

    try:
        # 读取输入文件
        with h5py.File(input_file, 'r') as f_in:
            # 读取数据
            left_endpose = f_in['endpose/left_endpose'][:]
            left_gripper = f_in['endpose/left_gripper'][:]
            right_endpose = f_in['endpose/right_endpose'][:]
            right_gripper = f_in['endpose/right_gripper'][:]

            num_frames = left_endpose.shape[0]
            print(f"📊 数据信息: {num_frames} 帧")

            # 初始化手部模型
            hand_model = HandModel()

            # 计算48维数据
            print("🔄 计算虚拟人手48维数据...")
            virtual_48d_data = np.zeros((num_frames, 48))

            for i in range(num_frames):
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"   处理进度: {i+1}/{num_frames}")

                hand_48d = compute_hand_48d_data(
                    left_endpose[i], left_gripper[i],
                    right_endpose[i], right_gripper[i],
                    hand_model
                )
                virtual_48d_data[i] = hand_48d

            # 保存到输出文件
            print("💾 保存48维数据到HDF5文件...")
            with h5py.File(output_file, 'w') as f_out:
                f_out.create_dataset('virtual_48d', data=virtual_48d_data)

                # 复制一些元数据（如果存在）
                for key in f_in.attrs:
                    f_out.attrs[key] = f_in.attrs[key]

        print(f"✅ 处理完成! 48维数据已保存到: {output_file}")
        print(f"   数据形状: {virtual_48d_data.shape}")
        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def process_directory(input_dir, output_dir=None):
    """
    处理整个目录中的HDF5文件

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（可选）
    """
    if not os.path.exists(input_dir):
        print(f"❌ 错误: 输入目录不存在 '{input_dir}'")
        return

    # 如果没有指定输出目录，自动生成
    if output_dir is None:
        output_dir = convert_output_path(input_dir)

    # 确保输出目录存在
    ensure_directory_exists(output_dir)

    print(f"🚀 开始处理目录:")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")

    # 查找所有HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.hdf5'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(
                    output_dir,
                    os.path.relpath(root, input_dir),
                    file
                )
                hdf5_files.append((input_file, output_file))

    if not hdf5_files:
        print("⚠️  未找到HDF5文件")
        return

    print(f"📁 找到 {len(hdf5_files)} 个HDF5文件")

    # 处理每个文件
    success_count = 0
    for i, (input_file, output_file) in enumerate(hdf5_files):
        print(f"\n[{i+1}/{len(hdf5_files)}] 处理文件: {os.path.basename(input_file)}")
        if process_single_file(input_file, output_file):
            success_count += 1

    print(f"\n✅ 目录处理完成!")
    print(f"   成功: {success_count}/{len(hdf5_files)} 个文件")

# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoboTwin到人手48维数据生成工具")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入文件或目录路径')
    parser.add_argument('--output', '-o', type=str,
                        help='输出文件或目录路径（可选，默认自动生成）')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='递归处理目录中的所有HDF5文件')

    args = parser.parse_args()

    if os.path.isfile(args.input):
        # 处理单个文件
        process_single_file(args.input, args.output)
    elif os.path.isdir(args.input):
        # 处理目录
        if args.recursive:
            process_directory(args.input, args.output)
        else:
            # 非递归：只处理顶层文件
            output_dir = args.output if args.output else convert_output_path(args.input)
            ensure_directory_exists(output_dir)

            hdf5_files = [f for f in os.listdir(args.input) if f.endswith('.hdf5')]
            if not hdf5_files:
                print("⚠️  未找到HDF5文件")
                return

            print(f"📁 找到 {len(hdf5_files)} 个HDF5文件")
            success_count = 0

            for file in hdf5_files:
                input_file = os.path.join(args.input, file)
                output_file = os.path.join(output_dir, file)
                print(f"\n处理文件: {file}")
                if process_single_file(input_file, output_file):
                    success_count += 1

            print(f"\n✅ 处理完成! 成功: {success_count}/{len(hdf5_files)} 个文件")
    else:
        print(f"❌ 错误: 输入路径不存在 '{args.input}'")

if __name__ == "__main__":
    main()