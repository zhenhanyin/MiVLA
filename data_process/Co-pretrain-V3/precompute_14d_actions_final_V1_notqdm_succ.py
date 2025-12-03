#!/usr/bin/env python3
"""
最终版本的14D动作预计算脚本
递归遍历Egodex-full目录结构，在Egodex-14d-actions-joint中生成对应的action_14d文件
"""

import pybullet as p
import numpy as np
import h5py
import os
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Dict, Any
import glob
import multiprocessing
import argparse

# 配置类
class RobotConfig:
    URDF_PATH = "data_process/robot/aloha-agilex/urdf/arx5_description_isaac.urdf"
    LEFT_ARM_JOINTS = [18, 19, 20, 21, 22, 23]
    RIGHT_ARM_JOINTS = [28, 29, 30, 31, 32, 33]
    LEFT_END_EFFECTOR_LINK = "fl_link6"
    RIGHT_END_EFFECTOR_LINK = "fr_link6"
    IK_MAX_ITERATIONS = 100
    IK_THRESHOLD = 1e-5
    WORKSPACE_SCALE = 1.0

class PyBulletManager:
    """PyBullet环境管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PyBulletManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not PyBulletManager._initialized:
            self.robot_id = None
            self.link_indices = {}
            self.initial_poses = {}
            self.physics_client = None
            self._setup_pybullet()
            PyBulletManager._initialized = True
    
    def _setup_pybullet(self):
        """设置PyBullet环境"""
        self.physics_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(RobotConfig.URDF_PATH, useFixedBase=True)
        self.link_indices = self._get_link_indices()
        self._set_initial_poses()
    
    def _get_link_indices(self) -> Dict[str, int]:
        """获取链接索引"""
        link_indices = {}
        link_indices['left_end_effector'] = 23
        link_indices['right_end_effector'] = 33
        return link_indices
    
    def _set_initial_poses(self):
        """设置机器人初始姿态"""
        left_initial_angles = [0, -1.57, 0, 0, 0, 0]
        for i, angle in enumerate(left_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.LEFT_ARM_JOINTS[i], angle)
        
        right_initial_angles = [0, -1.57, 0, 0, 0, 0]
        for i, angle in enumerate(right_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.RIGHT_ARM_JOINTS[i], angle)
        
        left_pos, left_orn = p.getLinkState(self.robot_id, self.link_indices['left_end_effector'])[:2]
        right_pos, right_orn = p.getLinkState(self.robot_id, self.link_indices['right_end_effector'])[:2]
        
        self.initial_poses = {
            'left': {'position': np.array(left_pos), 'orientation': np.array(left_orn)},
            'right': {'position': np.array(right_pos), 'orientation': np.array(right_orn)}
        }
    
    def get_robot_info(self) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, np.ndarray]]]:
        """获取机器人信息"""
        return self.robot_id, self.link_indices, self.initial_poses
    
    def close(self):
        """关闭PyBullet连接"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)
            PyBulletManager._initialized = False
            PyBulletManager._instance = None

class OptimizedHumanToRobotMapper:
    """人类到机器人映射器"""
    
    def __init__(self, pybullet_manager: PyBulletManager):
        self.pybullet_manager = pybullet_manager
        self.robot_id, self.link_indices, self.initial_poses = pybullet_manager.get_robot_info()
        self.coordinate_transform = None
        self.human_init_transforms = None
    
    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """计算坐标系变换矩阵"""
        left_human_init_pos = human_init_transforms['left'][:3, 3]
        right_human_init_pos = human_init_transforms['right'][:3, 3]
        
        left_robot_init_pos = self.initial_poses['left']['position']
        right_robot_init_pos = self.initial_poses['right']['position']
        
        self.coordinate_transform = {
            'left_human_init': left_human_init_pos,
            'right_human_init': right_human_init_pos,
            'left_robot_init': left_robot_init_pos,
            'right_robot_init': right_robot_init_pos,
            'scale': RobotConfig.WORKSPACE_SCALE
        }
        
        self.human_init_transforms = human_init_transforms
    
    def transform_human_to_robot(self, human_transform: np.ndarray, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """将人类手部变换转换为机器人目标位姿"""
        if self.coordinate_transform is None:
            raise ValueError("坐标系变换未计算")
        
        human_pos = human_transform[:3, 3]
        
        if arm == 'left':
            human_init_pos = self.coordinate_transform['left_human_init']
            human_delta = human_pos - human_init_pos
            scaled_delta = human_delta * self.coordinate_transform['scale']
            robot_pos = self.coordinate_transform['left_robot_init'] + scaled_delta
        else:
            human_init_pos = self.coordinate_transform['right_human_init']
            human_delta = human_pos - human_init_pos
            scaled_delta = human_delta * self.coordinate_transform['scale']
            robot_pos = self.coordinate_transform['right_robot_init'] + scaled_delta
        
        if arm == 'left':
            human_init_rot_matrix = self.human_init_transforms['left'][:3, :3]
        else:
            human_init_rot_matrix = self.human_init_transforms['right'][:3, :3]
        
        human_current_rot_matrix = human_transform[:3, :3]
        delta_rotation = Rotation.from_matrix(human_current_rot_matrix @ human_init_rot_matrix.T)
        
        robot_init_orientation_quat = self.initial_poses[arm]['orientation']
        robot_init_rotation = Rotation.from_quat(robot_init_orientation_quat)
        robot_target_rotation = delta_rotation * robot_init_rotation
        robot_orn = robot_target_rotation.as_quat()
        
        return robot_pos, robot_orn
    
    def solve_ik(self, target_position: np.ndarray, target_orientation: np.ndarray,
                is_left_arm: bool = True) -> Optional[np.ndarray]:
        """求解逆运动学"""
        try:
            arm_joints_indices = RobotConfig.LEFT_ARM_JOINTS if is_left_arm else RobotConfig.RIGHT_ARM_JOINTS
            end_effector_link = (self.link_indices['left_end_effector'] if is_left_arm
                            else self.link_indices['right_end_effector'])

            num_joints = p.getNumJoints(self.robot_id)
            movable_joint_indices = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

            joint_angles_movable = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                maxNumIterations=RobotConfig.IK_MAX_ITERATIONS,
                residualThreshold=RobotConfig.IK_THRESHOLD
            )

            if joint_angles_movable is None:
                return None

            movable_joints_map = dict(zip(movable_joint_indices, joint_angles_movable))
            arm_angles = [movable_joints_map.get(i, 0.0) for i in arm_joints_indices]

            return np.array(arm_angles, dtype=np.float32)

        except Exception:
            return None
    
    def calculate_gripper_state_from_tips(self, thumb_pos: np.ndarray, index_pos: np.ndarray, 
                                         min_distance: float = 0.01, max_distance: float = 0.2) -> float:
        """根据拇指和食指尖端位置计算夹爪状态"""
        try:
            distance = np.linalg.norm(thumb_pos - index_pos)
            gripper_state = np.clip((distance - min_distance) / (max_distance - min_distance), 0, 1)
            return gripper_state
        except Exception:
            return 0.5

def process_hdf5_file_final(input_path: str, output_path: str, 
                           pybullet_manager: PyBulletManager) -> bool:
    """最终版本HDF5文件处理 - 只输出action_14d数据"""
    try:
        mapper = OptimizedHumanToRobotMapper(pybullet_manager)
        
        with h5py.File(input_path, 'r') as f:
            # 检查必要的数据集
            if 'transforms' not in f or 'leftHand' not in f['transforms'] or 'rightHand' not in f['transforms']:
                return False
            
            left_hand_transforms = f['transforms']['leftHand'][:]
            right_hand_transforms = f['transforms']['rightHand'][:]
            
            has_left_thumb = 'leftThumbTip' in f['transforms']
            has_left_index = 'leftIndexFingerTip' in f['transforms']
            has_right_thumb = 'rightThumbTip' in f['transforms']
            has_right_index = 'rightIndexFingerTip' in f['transforms']
            
            num_frames = len(left_hand_transforms)
            
            human_init_transforms = {
                'left': left_hand_transforms[0],
                'right': right_hand_transforms[0]
            }
            mapper.calculate_coordinate_transform(human_init_transforms)
            
            # 计算指尖距离动态范围
            left_distances = []
            right_distances = []
            
            if has_left_thumb and has_left_index:
                for frame_idx in range(num_frames):
                    try:
                        left_thumb_pos = f['transforms']['leftThumbTip'][frame_idx][:3, 3]
                        left_index_pos = f['transforms']['leftIndexFingerTip'][frame_idx][:3, 3]
                        distance = np.linalg.norm(left_thumb_pos - left_index_pos)
                        left_distances.append(distance)
                    except (KeyError, IndexError):
                        continue
            
            if has_right_thumb and has_right_index:
                for frame_idx in range(num_frames):
                    try:
                        right_thumb_pos = f['transforms']['rightThumbTip'][frame_idx][:3, 3]
                        right_index_pos = f['transforms']['rightIndexFingerTip'][frame_idx][:3, 3]
                        distance = np.linalg.norm(right_thumb_pos - right_index_pos)
                        right_distances.append(distance)
                    except (KeyError, IndexError):
                        continue
            
            # 计算动态范围
            if left_distances:
                left_min_dist = np.min(left_distances)
                left_max_dist = np.max(left_distances)
                left_range = left_max_dist - left_min_dist
                if left_range < 1e-6:
                    left_min_dist = left_min_dist - 0.005
                    left_max_dist = left_min_dist + 0.005
            else:
                left_min_dist, left_max_dist = 0.01, 0.2
            
            if right_distances:
                right_min_dist = np.min(right_distances)
                right_max_dist = np.max(right_distances)
                right_range = right_max_dist - right_min_dist
                if right_range < 1e-6:
                    right_min_dist = right_min_dist - 0.005
                    right_max_dist = right_min_dist + 0.005
            else:
                right_min_dist, right_max_dist = 0.01, 0.2
            
            # 预分配数组
            actions_14d = np.zeros((num_frames, 14), dtype=np.float32)
            
            # LOCF策略
            last_successful_action = None
            
            for frame_idx in range(num_frames):
                try:
                    left_transform = left_hand_transforms[frame_idx]
                    right_transform = right_hand_transforms[frame_idx]
                    
                    left_target_pos, left_target_orn = mapper.transform_human_to_robot(left_transform, 'left')
                    right_target_pos, right_target_orn = mapper.transform_human_to_robot(right_transform, 'right')
                    
                    left_arm_angles = mapper.solve_ik(left_target_pos, left_target_orn, is_left_arm=True)
                    right_arm_angles = mapper.solve_ik(right_target_pos, right_target_orn, is_left_arm=False)
                    
                    # 计算夹爪状态
                    try:
                        left_thumb_pos = f['transforms']['leftThumbTip'][frame_idx][:3, 3]
                        left_index_pos = f['transforms']['leftIndexFingerTip'][frame_idx][:3, 3]
                        left_gripper = mapper.calculate_gripper_state_from_tips(
                            left_thumb_pos, left_index_pos, left_min_dist, left_max_dist
                        )
                    except (KeyError, IndexError):
                        left_gripper = 0.5
                    
                    try:
                        right_thumb_pos = f['transforms']['rightThumbTip'][frame_idx][:3, 3]
                        right_index_pos = f['transforms']['rightIndexFingerTip'][frame_idx][:3, 3]
                        right_gripper = mapper.calculate_gripper_state_from_tips(
                            right_thumb_pos, right_index_pos, right_min_dist, right_max_dist
                        )
                    except (KeyError, IndexError):
                        right_gripper = 0.5
                    
                    if left_arm_angles is not None and right_arm_angles is not None:
                        current_action = np.concatenate([
                            left_arm_angles, [left_gripper],
                            right_arm_angles, [right_gripper]
                        ])
                        
                        actions_14d[frame_idx] = current_action
                        last_successful_action = current_action
                    else:
                        if last_successful_action is not None:
                            actions_14d[frame_idx] = last_successful_action
                        else:
                            default_action = np.array([0.0] * 12 + [0.5, 0.5])
                            actions_14d[frame_idx] = default_action
                    
                except Exception:
                    if last_successful_action is not None:
                        actions_14d[frame_idx] = last_successful_action
                    else:
                        default_action = np.array([0.0] * 12 + [0.5, 0.5])
                        actions_14d[frame_idx] = default_action
                    continue
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果 - 只包含action_14d数据集
            with h5py.File(output_path, 'w') as out_f:
                out_f.create_dataset('action_14d', data=actions_14d)
                # 保存必要的元数据
                out_f.attrs['total_frames'] = num_frames
                out_f.attrs['left_gripper_min_distance'] = left_min_dist
                out_f.attrs['left_gripper_max_distance'] = left_max_dist
                out_f.attrs['right_gripper_min_distance'] = right_min_dist
                out_f.attrs['right_gripper_max_distance'] = right_max_dist
                out_f.attrs['normalization_method'] = 'dynamic_per_trajectory'
                out_f.attrs['source_file'] = input_path
            
            return True
            
    except Exception:
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
        # --- 核心修改：在这里排除目录 ---
        # 如果当前正在遍历的目录是根目录的直接子目录
        if root == root_dir:
            # 创建一个要删除的目录列表的副本，以免在迭代时修改列表
            dirs_to_remove = [d for d in dirs if d in exclude_dirs]
            for d in dirs_to_remove:
                print(f"  -> 忽略目录: {d}")
                dirs.remove(d)
        # -----------------------------

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
    output_filename = f"{filename}_14d.hdf5"
    # 构造输出路径
    output_path = os.path.join(output_root, os.path.dirname(relative_path), output_filename)
    return output_path

def worker_process(file_list, worker_id, input_root, output_root):
    """工作进程"""
    pybullet_manager = PyBulletManager()
    
    try:
        success_count = 0
        for input_file in file_list:
            output_file = map_input_to_output_path(input_file, input_root, output_root)
            
            success = process_hdf5_file_final(input_file, output_file, pybullet_manager)
            if success:
                success_count += 1
        
        return success_count
        
    except Exception:
        return 0
    finally:
        pybullet_manager.close()

def main_final_process():
    """最终版本主函数"""
    parser = argparse.ArgumentParser(description='14D动作预计算（最终版本 - 递归目录处理）')
    parser.add_argument('--input_root', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-full",
                       help='输入根目录 (例如: dataset/Egodex-full)')
    parser.add_argument('--output_root', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Egodex-14d-actions-joint_12_2",
                       help='输出根目录 (例如: dataset/Egodex-14d-actions-joint)')
    parser.add_argument('--num_processes', type=int, default=32,
                       help='进程数量')
    
    args = parser.parse_args()

    # <<<--- 新增：定义要排除的目录 ---<<<
    # EXCLUDE_DIRECTORIES = ['part2', 'part5','test']


    
    if not os.path.exists(args.input_root):
        print(f"❌ 输入根目录不存在: {args.input_root}")
        return
    
    print(f"🚀 开始递归处理目录: {args.input_root}")
    print(f"📁 输出根目录: {args.output_root}")
    
    # 递归查找所有HDF5文件
    print("🔍 搜索HDF5文件...")
    #hdf5_files = find_hdf5_files_recursive(args.input_root,exclude_dirs=EXCLUDE_DIRECTORIES)
    hdf5_files = find_hdf5_files_recursive(args.input_root)
    
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
    
    with multiprocessing.Pool(processes=actual_processes) as pool:
        results = []
        for i, file_list in enumerate(file_lists):
            if file_list:
                print(f"👷 启动工作进程 {i}，处理 {len(file_list)} 个文件")
                result = pool.apply_async(worker_process, (file_list, i, args.input_root, args.output_root))
                results.append(result)
        
        print("⏳ 等待所有工作进程完成...")
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