#!/usr/bin/env python3
"""
批量处理脚本：从EgoDex HDF5文件生成62维机械臂伪标签
- 读取原始48维动作数据
- 计算对应的14维机械臂关节数据
- 合并为62维数据保存到新的HDF5文件

输出格式：
- actions_62d: 62维动作数据
  - 前48维：原始人类动作数据
  - 后14维：机械臂关节数据（左臂6关节+左夹爪+右臂6关节+右夹爪）

功能特点：
- 多进程处理，默认使用所有CPU核心
- 断点续传支持
- 进度条显示
- 保留单文件处理功能
- 完整的错误处理和日志记录
"""

import numpy as np
import h5py
import os
import argparse
import pybullet as p
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
import json
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import hashlib
import signal
import sys
import fnmatch

# =============================================================================#
# 核心配置类（从原脚本复制）                                            #
# =============================================================================#

class RobotConfig:
    """机器人配置类"""
    URDF_PATH = "/home/yinzhenhan/Human_Pre/dataset/robot/aloha-agilex/urdf/arx5_description_isaac.urdf"
    LEFT_ARM_JOINTS = [18, 19, 20, 21, 22, 23]
    RIGHT_ARM_JOINTS = [28, 29, 30, 31, 32, 33]
    LEFT_END_EFFECTOR_LINK = "fl_link6"
    RIGHT_END_EFFECTOR_LINK = "fr_link6"
    IK_MAX_ITERATIONS = 200
    IK_THRESHOLD = 1e-3
    WORKSPACE_SCALE = 1.2

class PyBulletManager:
    """PyBullet环境管理器"""
    
    def __init__(self):
        self.robot_id = None
        self.link_indices = {}
        self.initial_poses = {}
        self.physics_client = None
        self.movable_joint_indices = []
        self.joint_ll = []
        self.joint_ul = []
        self.joint_jr = []
        
        self._setup_pybullet()
    
    def _setup_pybullet(self):
        """设置PyBullet环境"""
        self.physics_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(RobotConfig.URDF_PATH, useFixedBase=True)
        self.link_indices = self._get_link_indices()
        self._get_joint_constraints()
        self._set_initial_poses()
    
    def _get_joint_constraints(self):
        """获取关节约束信息"""
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.movable_joint_indices.append(i)
                self.joint_ll.append(info[8])
                self.joint_ul.append(info[9])
                self.joint_jr.append(info[9] - info[8])
    
    def _get_link_indices(self):
        """获取末端执行器链接索引"""
        link_indices = {}
        link_indices['left_end_effector'] = 23
        link_indices['right_end_effector'] = 33
        return link_indices
    
    def _set_initial_poses(self):
        """设置机器人初始姿态"""
        # 使用姿态(把手搭在桌子上)
        left_initial_angles = [-0.6, 0, 0, 0, 0, 0]
        right_initial_angles = [0.6, 0, 0, 0, 0, 0]
        
        # 应用关节角度
        for i, angle in enumerate(left_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.LEFT_ARM_JOINTS[i], angle)
        
        for i, angle in enumerate(right_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.RIGHT_ARM_JOINTS[i], angle)
        
        # 获取初始末端位姿
        left_pos, left_orn = p.getLinkState(self.robot_id, self.link_indices['left_end_effector'])[:2]
        right_pos, right_orn = p.getLinkState(self.robot_id, self.link_indices['right_end_effector'])[:2]
        
        self.initial_poses = {
            'left': {'position': np.array(left_pos), 'orientation': np.array(left_orn)},
            'right': {'position': np.array(right_pos), 'orientation': np.array(right_orn)}
        }
    
    def get_robot_info(self):
        """获取机器人信息"""
        return self.robot_id, self.link_indices, self.initial_poses
    
    def solve_ik(self, target_position, target_orientation, is_left_arm=True):
        """求解逆运动学 - 使用差异化的jointDamping权重"""
        try:
            arm_joints_indices = RobotConfig.LEFT_ARM_JOINTS if is_left_arm else RobotConfig.RIGHT_ARM_JOINTS
            end_effector_link = (self.link_indices['left_end_effector'] if is_left_arm
                            else self.link_indices['right_end_effector'])
            
            joint_states = p.getJointStates(self.robot_id, self.movable_joint_indices)

            current_positions = [state[0] for state in joint_states]
            
            # 创建差异化的jointDamping权重配置

            joint_damping = self._create_differential_damping(is_left_arm)
            
            joint_angles_movable = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=self.joint_ll,
                upperLimits=self.joint_ul,
                jointRanges=self.joint_jr,
                #restPoses=current_positions, # 使用当前姿态作为参考，实现平滑过渡
                #jointDamping=joint_damping, # 添加差异化的权重因子
                maxNumIterations=RobotConfig.IK_MAX_ITERATIONS,
                residualThreshold=RobotConfig.IK_THRESHOLD
            )

            if joint_angles_movable is None:
                return None

            # 提取对应臂的关节角度
            movable_joints_map = dict(zip(self.movable_joint_indices, joint_angles_movable))
            arm_angles = [movable_joints_map.get(i, 0.0) for i in arm_joints_indices]

            return np.array(arm_angles, dtype=np.float32)

        except Exception as e:
            return None
    
    def _create_differential_damping(self, is_left_arm=True):
        """
        创建差异化的jointDamping权重配置
        
        根据分析结果：
        - 左右臂关节1和关节2突变最大（0.76-2.5弧度），使用高阻尼系数0.3
        - 其他关节突变较小，使用低阻尼系数0.05以保持精度
        
        参数:
            is_left_arm: 是否为左臂
            
        返回:
            joint_damping: 针对所有可动关节的阻尼系数列表
        """
        # 创建基础阻尼配置（低阻尼）
        joint_damping = [0.0001] * len(self.movable_joint_indices)
        
        # 获取目标臂的关节索引
        if is_left_arm:
            # 左臂关节索引：18, 19, 20, 21, 22, 23

            arm_joint_indices = RobotConfig.LEFT_ARM_JOINTS
        else:
            # 右臂关节索引：28, 29, 30, 31, 32, 33
            # 关节1: 29, 关节2: 30 (在数组中的位置)
            arm_joint_indices = RobotConfig.RIGHT_ARM_JOINTS
        
        # 为关节1和关节2设置更高的阻尼系数
        # 关节1 (第一个关节): 在arm_joint_indices[0]
        if len(arm_joint_indices) > 0:
            joint1_idx = arm_joint_indices[0]  # 关节1
            # 在movable_joint_indices中找到对应的位置
            if joint1_idx in self.movable_joint_indices:
                pos = self.movable_joint_indices.index(joint1_idx)
                joint_damping[pos] = 0.1
        
        # 关节2 (第二个关节): 在arm_joint_indices[1]
        if len(arm_joint_indices) > 1:
            joint2_idx = arm_joint_indices[1]  # 关节2
            # 在movable_joint_indices中找到对应的位置
            if joint2_idx in self.movable_joint_indices:
                pos = self.movable_joint_indices.index(joint2_idx)
                joint_damping[pos] = 0.1
        
        return joint_damping
    
    def debug_damping_config(self, is_left_arm=True):
        """调试方法：打印当前的阻尼配置"""
        print(f"\n🔧 {'左臂' if is_left_arm else '右臂'} 阻尼配置:")
        
        # 获取阻尼配置
        joint_damping = self._create_differential_damping(is_left_arm)
        
        # 获取目标臂的关节索引
        if is_left_arm:
            arm_joint_indices = RobotConfig.LEFT_ARM_JOINTS
        else:
            arm_joint_indices = RobotConfig.RIGHT_ARM_JOINTS
        
        print(f"  可动关节数量: {len(self.movable_joint_indices)}")
        print(f"  臂关节索引: {arm_joint_indices}")
        
        # 打印每个臂关节的阻尼系数
        for i, joint_idx in enumerate(arm_joint_indices):
            if joint_idx in self.movable_joint_indices:
                pos = self.movable_joint_indices.index(joint_idx)
                damping_value = joint_damping[pos]
                joint_name = f"关节{i+1}"
                print(f"    {joint_name} (索引{joint_idx}): 阻尼系数 = {damping_value}")
        
        # 统计不同阻尼值的关节数量
        high_damping_joints = sum(1 for d in joint_damping if d == 0.3)
        low_damping_joints = sum(1 for d in joint_damping if d == 0.05)
        
        print(f"  高阻尼关节数(0.3): {high_damping_joints}")
        print(f"  低阻尼关节数(0.05): {low_damping_joints}")
    
    def close(self):
        """关闭PyBullet连接"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)

class OptimizedHumanToRobotMapper:
    """人类到机器人坐标变换映射器"""
    
    def __init__(self, pybullet_manager: PyBulletManager):
        _, _, self.initial_poses = pybullet_manager.get_robot_info()
        
        # 直接坐标变换公式：EgoDex (x,y,z) -> PyBullet (-z,-x,y)
        self.coordinate_transform_matrix = np.array([
            [0, 0, -1],  # 人类-Z -> 机器人X
            [-1, 0, 0],  # 人类-X -> 机器人Y
            [0, 1, 0]    # 人类Y -> 机器人Z
        ])
        
        # 存储起始位置
        self.human_start_pos = {}
        self.robot_start_pos = {}

    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """计算坐标变换校准"""
        # 记录人类在Egodex坐标系下的起始位置
        self.human_start_pos['left'] = human_init_transforms['left'][:3, 3]
        self.human_start_pos['right'] = human_init_transforms['right'][:3, 3]

        # 记录机器人在PyBullet坐标系下的起始位置
        self.robot_start_pos['left'] = self.initial_poses['left']['position']
        self.robot_start_pos['right'] = self.initial_poses['right']['position']

    def transform_egodex_vector_to_pybullet(self, egodex_vec):
        """变换EgoDex向量到PyBullet坐标系"""
        return self.coordinate_transform_matrix @ egodex_vec  

    def transform_human_to_robot_target(self, human_transform: np.ndarray, arm: str) -> Tuple[np.ndarray, Rotation]:
        """变换人类手部位姿到机器人目标位姿"""
        # 位置变换
        human_current_pos = human_transform[:3, 3]
        human_delta = human_current_pos - self.human_start_pos[arm]
        robot_delta = self.transform_egodex_vector_to_pybullet(human_delta)
        robot_target_pos = self.robot_start_pos[arm] + robot_delta

        # 姿态变换
        human_rot_matrix = human_transform[:3, :3]
        robot_rot_matrix = self.coordinate_transform_matrix @ human_rot_matrix
        robot_rot = Rotation.from_matrix(robot_rot_matrix)
        
        return robot_target_pos, robot_rot

    def calculate_gripper_state_from_tips(self, thumb_pos: np.ndarray, index_pos: np.ndarray, 
                                         min_distance: float, max_distance: float) -> float:
        """根据拇指和食指尖端位置计算夹爪状态 - 使用动态归一化参数"""
        try:
            distance = np.linalg.norm(thumb_pos - index_pos)
            # 防止分母为零
            denominator = max_distance - min_distance
            if denominator < 1e-6:
                denominator = 0.1  # 默认范围
            gripper_state = (distance - min_distance) / denominator
            gripper_state = np.clip(gripper_state, 0.0, 1.0)
            return gripper_state
        except Exception:
            return 0.5

    def calculate_dynamic_gripper_ranges(self, hdf5_file, num_frames):
        """计算动态夹爪范围参数"""
        left_distances = []
        right_distances = []
        
        # 检查指尖数据可用性
        has_left_thumb = 'leftThumbTip' in hdf5_file['transforms']
        has_left_index = 'leftIndexFingerTip' in hdf5_file['transforms']
        has_right_thumb = 'rightThumbTip' in hdf5_file['transforms']
        has_right_index = 'rightIndexFingerTip' in hdf5_file['transforms']
        
        # 计算所有帧的指尖距离
        for frame_idx in range(num_frames):
            try:
                # 左手距离
                if has_left_thumb and has_left_index:
                    left_thumb_pos = hdf5_file['/transforms/leftThumbTip'][frame_idx][:3, 3]
                    left_index_pos = hdf5_file['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                    left_distance = np.linalg.norm(left_thumb_pos - left_index_pos)
                    left_distances.append(left_distance)
                
                # 右手距离
                if has_right_thumb and has_right_index:
                    right_thumb_pos = hdf5_file['/transforms/rightThumbTip'][frame_idx][:3, 3]
                    right_index_pos = hdf5_file['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                    right_distance = np.linalg.norm(right_thumb_pos - right_index_pos)
                    right_distances.append(right_distance)
                    
            except (KeyError, IndexError):
                continue
        
        # 计算动态范围
        gripper_ranges = {}
        
        # 左手夹爪范围
        if left_distances:
            left_min_dist = np.min(left_distances)
            left_max_dist = np.max(left_distances)
            left_range = left_max_dist - left_min_dist
            if left_range < 1e-6:  # 防止范围过小
                left_min_dist = left_min_dist - 0.005
                left_max_dist = left_min_dist + 0.005
            gripper_ranges['left'] = {'min': left_min_dist, 'max': left_max_dist}
        else:
            gripper_ranges['left'] = {'min': 0.01, 'max': 0.2}
        
        # 右手夹爪范围
        if right_distances:
            right_min_dist = np.min(right_distances)
            right_max_dist = np.max(right_distances)
            right_range = right_max_dist - right_min_dist
            if right_range < 1e-6:  # 防止范围过小
                right_min_dist = right_min_dist - 0.005
                right_max_dist = right_min_dist + 0.005
            gripper_ranges['right'] = {'min': right_min_dist, 'max': right_max_dist}
        else:
            gripper_ranges['right'] = {'min': 0.01, 'max': 0.2}
        
        return gripper_ranges

# =============================================================================#
# 文件处理工具函数                                                        #
# =============================================================================#

def get_file_hash(file_path: str) -> str:
    """获取文件的MD5哈希值，用于断点续传"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_output_path(input_path: str, input_root: str, output_root: str) -> str:
    """根据输入路径生成对应的输出路径"""
    # 获取相对于输入根目录的相对路径
    rel_path = os.path.relpath(input_path, input_root)
    # 构建输出路径
    output_path = os.path.join(output_root, rel_path)
    return output_path

def create_output_directory(output_path: str):
    """创建输出目录"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def is_file_processed(input_file: str, progress_file: str) -> bool:
    """检查文件是否已经处理过"""
    if not os.path.exists(progress_file):
        return False
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        file_hash = get_file_hash(input_file)
        processed_files = progress_data.get('processed_files', {})
        
        return processed_files.get(input_file) == file_hash
    except:
        return False

def update_progress_file(input_file: str, progress_file: str, success: bool = True):
    """更新进度文件"""
    file_hash = get_file_hash(input_file)
    
    progress_data = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except:
            pass
    
    if 'processed_files' not in progress_data:
        progress_data['processed_files'] = {}
    
    if success:
        progress_data['processed_files'][input_file] = file_hash
    else:
        # 如果处理失败，移除记录以便重试
        progress_data['processed_files'].pop(input_file, None)
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

# =============================================================================#
# 单文件处理核心函数                                                    #
# =============================================================================#

def process_single_file_worker(input_file: str, output_file: str, 
                             debug: bool = False) -> Dict:
    """
    处理单个HDF5文件的worker函数（用于多进程）
    
    参数:
        input_file: 输入HDF5文件路径
        output_file: 输出HDF5文件路径
        debug: 是否打印调试信息
        
    返回:
        处理结果字典
    """
    result = {
        'input_file': input_file,
        'output_file': output_file,
        'success': False,
        'error': None,
        'stats': None,
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        if debug:
            print(f"🔧 开始处理文件: {input_file}")
        
        # 创建PyBullet环境（每个进程独立创建）
        pybullet_manager = PyBulletManager()
        mapper = OptimizedHumanToRobotMapper(pybullet_manager)
        
        # 打开输入文件
        with h5py.File(input_file, 'r') as hdf5_file:
            # 检查必需的数据集
            if 'actions_48d' not in hdf5_file:
                raise ValueError("输入文件中缺少 'actions_48d' 数据集")
            
            if '/transforms/leftThumbIntermediateBase' not in hdf5_file:
                raise ValueError("输入文件中缺少变换数据")
            
            num_frames = hdf5_file['actions_48d'].shape[0]
            actions_48d = hdf5_file['actions_48d'][:]
            
            if debug:
                print(f"📊 输入数据: {num_frames} 帧, 48维动作")
            
            # 检查指尖数据
            has_left_thumb = 'leftThumbTip' in hdf5_file['transforms']
            has_left_index = 'leftIndexFingerTip' in hdf5_file['transforms']
            has_right_thumb = 'rightThumbTip' in hdf5_file['transforms']
            has_right_index = 'rightIndexFingerTip' in hdf5_file['transforms']
            
            # 计算坐标变换
            human_init_transforms = {
                'left': hdf5_file['/transforms/leftThumbIntermediateBase'][0],
                'right': hdf5_file['/transforms/rightThumbIntermediateBase'][0]
            }
            mapper.calculate_coordinate_transform(human_init_transforms)
            
            # 计算动态夹爪范围
            gripper_ranges = mapper.calculate_dynamic_gripper_ranges(hdf5_file, num_frames)
            
            if debug:
                print(f"🔧 指尖数据可用性: 左拇指={has_left_thumb}, 左食指={has_left_index}, 右拇指={has_right_thumb}, 右食指={has_right_index}")
                print(f"🔧 左手夹爪动态范围: {gripper_ranges['left']['min']:.4f} - {gripper_ranges['left']['max']:.4f}")
                print(f"🔧 右手夹爪动态范围: {gripper_ranges['right']['min']:.4f} - {gripper_ranges['right']['max']:.4f}")
            
            # 初始化输出数组
            actions_62d = np.zeros((num_frames, 62), dtype=np.float32)
            actions_14d = np.zeros((num_frames, 14), dtype=np.float32)
            
            processing_stats = {
                'total_frames': num_frames,
                'successful_frames': 0,
                'failed_frames': 0,
                'ik_failures': 0,
                'transform_failures': 0
            }
            
            # 逐帧处理
            for frame_idx in range(num_frames):
                try:
                    # 读取手部变换
                    left_transform = hdf5_file['/transforms/leftThumbIntermediateBase'][frame_idx]
                    right_transform = hdf5_file['/transforms/rightThumbIntermediateBase'][frame_idx]
                    
                    # 坐标变换得到目标位置
                    left_pos, left_orn = mapper.transform_human_to_robot_target(left_transform, 'left')
                    right_pos, right_orn = mapper.transform_human_to_robot_target(right_transform, 'right')
                    
                    # IK求解
                    left_joint_angles = pybullet_manager.solve_ik(left_pos, left_orn, is_left_arm=True)
                    right_joint_angles = pybullet_manager.solve_ik(right_pos, right_orn, is_left_arm=False)
                    
                    if left_joint_angles is None or right_joint_angles is None:
                        processing_stats['ik_failures'] += 1
                        processing_stats['failed_frames'] += 1
                        continue
                    
                    # 计算夹爪状态
                    left_gripper_state = 0.5
                    right_gripper_state = 0.5
                    
                    try:
                        if has_left_thumb and has_left_index:
                            left_thumb_pos = hdf5_file['/transforms/leftThumbTip'][frame_idx][:3, 3]
                            left_index_pos = hdf5_file['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                            left_gripper_state = mapper.calculate_gripper_state_from_tips(
                                left_thumb_pos, left_index_pos,
                                gripper_ranges['left']['min'],
                                gripper_ranges['left']['max']
                            )
                        
                        if has_right_thumb and has_right_index:
                            right_thumb_pos = hdf5_file['/transforms/rightThumbTip'][frame_idx][:3, 3]
                            right_index_pos = hdf5_file['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                            right_gripper_state = mapper.calculate_gripper_state_from_tips(
                                right_thumb_pos, right_index_pos,
                                gripper_ranges['right']['min'],
                                gripper_ranges['right']['max']
                            )
                    except (KeyError, IndexError):
                        pass
                    
                    # 组装14维动作
                    action_14d = np.concatenate([
                        left_joint_angles,           # 前6维：左臂关节
                        [left_gripper_state],       # 第7维：左夹爪
                        right_joint_angles,          # 后6维：右臂关节
                        [right_gripper_state]       # 第14维：右夹爪
                    ])
                    
                    actions_14d[frame_idx] = action_14d
                    processing_stats['successful_frames'] += 1
                    
                except Exception as e:
                    processing_stats['transform_failures'] += 1
                    processing_stats['failed_frames'] += 1
                    continue
            
            # 合并48维和14维为62维
            actions_62d[:, :48] = actions_48d
            actions_62d[:, 48:] = actions_14d
            
            # 关闭PyBullet环境
            pybullet_manager.close()
        
        # 创建输出目录
        create_output_directory(output_file)
        
        # 保存到新的HDF5文件
        with h5py.File(output_file, 'w') as output_hdf5:
            output_hdf5.create_dataset('actions_62d', data=actions_62d, compression='gzip')
            
            # 保存处理统计信息作为属性
            output_hdf5.attrs['processing_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
            output_hdf5.attrs['input_file'] = input_file
            output_hdf5.attrs['total_frames'] = processing_stats['total_frames']
            output_hdf5.attrs['successful_frames'] = processing_stats['successful_frames']
            output_hdf5.attrs['failed_frames'] = processing_stats['failed_frames']
            output_hdf5.attrs['ik_failures'] = processing_stats['ik_failures']
            output_hdf5.attrs['transform_failures'] = processing_stats['transform_failures']
            success_rate = processing_stats['successful_frames'] / processing_stats['total_frames'] * 100
            output_hdf5.attrs['success_rate'] = success_rate
        
        result['success'] = True
        result['stats'] = processing_stats
        
        if debug:
            print(f"✅ 处理完成: {input_file}")
            print(f"   输出文件: {output_file}")
            print(f"   成功率: {processing_stats['successful_frames']}/{processing_stats['total_frames']} ({success_rate:.1f}%)")
            print(f"   IK失败: {processing_stats['ik_failures']}, 变换失败: {processing_stats['transform_failures']}")
    
    except Exception as e:
        result['error'] = str(e)
        if debug:
            print(f"❌ 处理失败: {input_file}")
            print(f"   错误: {e}")
    
    result['processing_time'] = time.time() - start_time
    return result

# =============================================================================#
# 批量处理器类                                                            #
# =============================================================================#

class EgodexTo62DProcessor:
    """EgoDex到62维机械臂标签的批量处理器"""
    
    def __init__(self, num_processes: int = None, debug: bool = False):
        self.num_processes = num_processes or mp.cpu_count()
        self.debug = debug
        self.progress_file = "processing_progress.json"
        
        print(f"🚀 初始化批量处理器")
        print(f"   CPU核心数: {self.num_processes}")
        print(f"   调试模式: {'开启' if debug else '关闭'}")
    
    def scan_input_files(self, input_root: str, file_pattern: str = "*.hdf5") -> List[str]:
        """扫描输入目录下的所有HDF5文件"""
        print(f"🔍 扫描输入目录: {input_root}")
        
        hdf5_files = []
        for root, dirs, files in os.walk(input_root):
            for file in files:
                if fnmatch.fnmatch(file, file_pattern):
                    file_path = os.path.join(root, file)
                    # 检查是否包含actions_48d数据集
                    try:
                        with h5py.File(file_path, 'r') as f:
                            if 'actions_48d' in f:
                                hdf5_files.append(file_path)
                    except:
                        continue
        
        print(f"📊 找到 {len(hdf5_files)} 个包含actions_48d的HDF5文件")
        return hdf5_files
    
    def process_directory(self, input_root: str, output_root: str, 
                         file_pattern: str = "*.hdf5", resume: bool = True) -> Dict:
        """
        批量处理目录下的所有HDF5文件
        
        参数:
            input_root: 输入根目录
            output_root: 输出根目录
            file_pattern: 文件匹配模式
            resume: 是否断点续传
            
        返回:
            批量处理统计信息
        """
        import fnmatch
        
        # 扫描输入文件
        input_files = self.scan_input_files(input_root, file_pattern)
        
        if not input_files:
            print(f"❌ 在目录 {input_root} 中未找到匹配的HDF5文件")
            return {'total_files': 0, 'processed_files': 0, 'failed_files': 0}
        
        # 准备输出路径
        file_pairs = []
        for input_file in input_files:
            output_file = get_output_path(input_file, input_root, output_root)
            file_pairs.append((input_file, output_file))
        
        # 过滤已处理的文件（如果启用断点续传）
        if resume:
            unprocessed_pairs = []
            for input_file, output_file in file_pairs:
                if not is_file_processed(input_file, self.progress_file):
                    unprocessed_pairs.append((input_file, output_file))
            
            print(f"📋 断点续传: 跳过 {len(file_pairs) - len(unprocessed_pairs)} 个已处理文件")
            file_pairs = unprocessed_pairs
        
        print(f"🎯 将处理 {len(file_pairs)} 个文件")
        
        # 创建进度条
        pbar = tqdm(total=len(file_pairs), desc="批量处理进度", unit="文件")
        
        # 处理统计
        batch_stats = {
            'total_files': len(file_pairs),
            'processed_files': 0,
            'failed_files': 0,
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'total_processing_time': 0,
            'file_results': []
        }
        
        # 多进程处理
        if self.num_processes > 1:
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(process_single_file_worker, input_file, output_file, self.debug): (input_file, output_file)
                    for input_file, output_file in file_pairs
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_file):
                    input_file, output_file = future_to_file[future]
                    
                    try:
                        result = future.result()
                        self._handle_file_result(result, batch_stats)
                        
                        # 更新进度文件
                        if result['success']:
                            update_progress_file(input_file, self.progress_file, success=True)
                        
                    except Exception as e:
                        error_result = {
                            'input_file': input_file,
                            'output_file': output_file,
                            'success': False,
                            'error': f"进程异常: {str(e)}",
                            'processing_time': 0
                        }
                        self._handle_file_result(error_result, batch_stats)
                    
                    pbar.update(1)
        
        else:
            # 单进程处理（用于调试）
            for input_file, output_file in file_pairs:
                result = process_single_file_worker(input_file, output_file, self.debug)
                self._handle_file_result(result, batch_stats)
                
                if result['success']:
                    update_progress_file(input_file, self.progress_file, success=True)
                
                pbar.update(1)
        
        pbar.close()
        
        # 打印最终统计
        self._print_final_stats(batch_stats)
        
        # 保存批量处理统计
        stats_file = os.path.join(output_root, 'batch_stats.json')
        os.makedirs(output_root, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        
        return batch_stats
    
    def process_single_file(self, input_file: str, output_root: str) -> Dict:
        """
        处理单个文件（保留此功能用于调试）
        
        参数:
            input_file: 输入文件路径
            output_root: 输出根目录
            
        返回:
            处理结果
        """
        input_root = os.path.dirname(os.path.dirname(input_file))  # 假设输入文件在 partX/basic_fold/ 下
        output_file = get_output_path(input_file, input_root, output_root)
        
        print(f"🎯 处理单个文件: {input_file}")
        print(f"   输出文件: {output_file}")
        
        result = process_single_file_worker(input_file, output_file, debug=True)
        
        if result['success']:
            print(f"✅ 单文件处理成功!")
            print(f"   输出文件: {result['output_file']}")
            print(f"   处理时间: {result['processing_time']:.2f}秒")
            if result['stats']:
                stats = result['stats']
                success_rate = stats['successful_frames'] / stats['total_frames'] * 100
                print(f"   成功率: {stats['successful_frames']}/{stats['total_frames']} ({success_rate:.1f}%)")
        else:
            print(f"❌ 单文件处理失败: {result['error']}")
        
        return result
    
    def _handle_file_result(self, result: Dict, batch_stats: Dict):
        """处理单个文件的结果"""
        batch_stats['file_results'].append(result)
        
        if result['success']:
            batch_stats['processed_files'] += 1
            if result['stats']:
                batch_stats['total_frames'] += result['stats']['total_frames']
                batch_stats['successful_frames'] += result['stats']['successful_frames']
                batch_stats['failed_frames'] += result['stats']['failed_frames']
        else:
            batch_stats['failed_files'] += 1
        
        batch_stats['total_processing_time'] += result['processing_time']
    
    def _print_final_stats(self, batch_stats: Dict):
        """打印最终统计信息"""
        print(f"\n🎉 批量处理完成!")
        print(f"   总文件数: {batch_stats['total_files']}")
        print(f"   处理成功: {batch_stats['processed_files']}")
        print(f"   处理失败: {batch_stats['failed_files']}")
        
        if batch_stats['processed_files'] > 0:
            success_rate = batch_stats['processed_files'] / batch_stats['total_files'] * 100
            print(f"   文件成功率: {success_rate:.1f}%")
        
        if batch_stats['total_frames'] > 0:
            frame_success_rate = batch_stats['successful_frames'] / batch_stats['total_frames'] * 100
            print(f"   总帧数: {batch_stats['total_frames']}")
            print(f"   成功帧数: {batch_stats['successful_frames']}")
            print(f"   失败帧数: {batch_stats['failed_frames']}")
            print(f"   帧成功率: {frame_success_rate:.1f}%")
        
        total_time = batch_stats['total_processing_time']
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"   总处理时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if batch_stats['processed_files'] > 0:
            avg_time = total_time / batch_stats['processed_files']
            print(f"   平均每文件: {avg_time:.2f}秒")

# =============================================================================#
# 信号处理函数                                                            #
# =============================================================================#

def signal_handler(sig, frame):
    """处理Ctrl+C信号，优雅退出"""
    print(f"\n⚠️  收到中断信号，正在清理...")
    sys.exit(0)

# =============================================================================#
# 主函数                                                                  #
# =============================================================================#

def main():
    """主函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="批量处理EgoDex HDF5文件生成62维机械臂伪标签")
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='输入HDF5文件根目录')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='输出HDF5文件根目录')
    parser.add_argument('--file_pattern', '-p', type=str, default='*.hdf5',
                        help='文件匹配模式 (默认: *.hdf5)')
    parser.add_argument('--single_file', '-s', type=str, default=None,
                        help='处理单个文件 (可选)')
    parser.add_argument('--num_processes', '-n', type=int, default=None,
                        help='进程数 (默认: 使用所有CPU核心)')
    parser.add_argument('--no_resume', action='store_true',
                        help='禁用断点续传')
    parser.add_argument('--debug', action='store_true',
                        help='启用详细调试输出')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = EgodexTo62DProcessor(
        num_processes=args.num_processes,
        debug=args.debug
    )
    
    try:
        if args.single_file:
            # 处理单个文件
            result = processor.process_single_file(args.single_file, args.output_dir)
            
            if not result['success']:
                sys.exit(1)
        
        else:
            # 批量处理目录
            print(f"🚀 开始批量处理")
            print(f"   输入目录: {args.input_dir}")
            print(f"   输出目录: {args.output_dir}")
            print(f"   文件模式: {args.file_pattern}")
            print(f"   进程数: {processor.num_processes}")
            print(f"   断点续传: {'禁用' if args.no_resume else '启用'}")
            
            batch_stats = processor.process_directory(
                args.input_dir,
                args.output_dir,
                args.file_pattern,
                resume=not args.no_resume
            )
            
            # 如果有失败文件，返回非零退出码
            if batch_stats['failed_files'] > 0:
                print(f"⚠️  有 {batch_stats['failed_files']} 个文件处理失败")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()