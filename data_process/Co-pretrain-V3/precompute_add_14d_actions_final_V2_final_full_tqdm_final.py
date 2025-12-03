#!/usr/bin/env python3
"""
最终版本的14D动作预计算脚本
递归遍历Egodex-full目录结构，在Egodex-full-pretrain中生成对应的action_62d文件
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
from tqdm import tqdm

# 配置类 - 与batch_process_14d_labels.py保持一致
class RobotConfig:
    URDF_PATH = "/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/yinzhenhan/data_process/robot/aloha-agilex/urdf/arx5_description_isaac.urdf"
    LEFT_ARM_JOINTS = [18, 19, 20, 21, 22, 23]
    RIGHT_ARM_JOINTS = [28, 29, 30, 31, 32, 33]
    LEFT_END_EFFECTOR_LINK = "fl_link6"
    RIGHT_END_EFFECTOR_LINK = "fr_link6"
    IK_MAX_ITERATIONS = 150
    IK_THRESHOLD = 1e-3



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
            
            self.movable_joint_indices = []
            self.joint_ll = []
            self.joint_ul = []
            self.joint_jr = []

            self._setup_pybullet()
            PyBulletManager._initialized = True
    
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
        # 使用自然下垂姿态
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
        self.coordinate_transform_matrix = None
        self.human_init_transforms = None
        
        # 存储起始位置
        self.human_start_pos = {}
        self.robot_start_pos = {}
        
        # 从PyBulletManager获取关节信息
        self.movable_joint_indices = pybullet_manager.movable_joint_indices
        self.joint_ll = pybullet_manager.joint_ll
        self.joint_ul = pybullet_manager.joint_ul
        self.joint_jr = pybullet_manager.joint_jr
    
    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """计算坐标变换校准 - 与batch_process_14d_labels.py保持一致"""
        # 记录人类在Egodex坐标系下的起始位置
        self.human_start_pos = {}
        self.human_start_pos['left'] = human_init_transforms['left'][:3, 3]
        self.human_start_pos['right'] = human_init_transforms['right'][:3, 3]

        # 记录机器人在PyBullet坐标系下的起始位置
        self.robot_start_pos = {}
        self.robot_start_pos['left'] = self.initial_poses['left']['position']
        self.robot_start_pos['right'] = self.initial_poses['right']['position']
        
        # 设置坐标变换矩阵
        self.coordinate_transform_matrix = np.array([
            [0, 0, -1],  # 人类-Z -> 机器人X
            [-1, 0, 0],  # 人类-X -> 机器人Y
            [0, 1, 0]    # 人类Y -> 机器人Z
        ])
        
        self.human_init_transforms = human_init_transforms
    
    def transform_human_to_robot(self, human_transform: np.ndarray, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """变换人类手部位姿到机器人目标位姿 """
        # 位置变换
        human_current_pos = human_transform[:3, 3]
        human_delta = human_current_pos - self.human_start_pos[arm]
        robot_delta = self.coordinate_transform_matrix @ human_delta
        robot_target_pos = self.robot_start_pos[arm] + robot_delta

        # 姿态变换
        human_rot_matrix = human_transform[:3, :3]
        robot_rot_matrix = self.coordinate_transform_matrix @ human_rot_matrix
        robot_rot = Rotation.from_matrix(robot_rot_matrix)
        
        return robot_target_pos, robot_rot
    
    def solve_ik(self, target_position: np.ndarray, target_orientation: np.ndarray,
                is_left_arm: bool = True) -> Optional[np.ndarray]:
        """求解逆运动学 - """
        try:
            arm_joints_indices = RobotConfig.LEFT_ARM_JOINTS if is_left_arm else RobotConfig.RIGHT_ARM_JOINTS
            end_effector_link = (self.link_indices['left_end_effector'] if is_left_arm
                            else self.link_indices['right_end_effector'])
            
            joint_states = p.getJointStates(self.robot_id, self.movable_joint_indices)
            # current_positions = [state[0] for state in joint_states]


            joint_angles_movable = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=self.joint_ll,
                upperLimits=self.joint_ul,
                jointRanges=self.joint_jr,
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

def process_hdf5_file_final(input_path: str, output_path: str, 
                           pybullet_manager: PyBulletManager) -> bool:
    """最终版本HDF5文件处理 - 使用batch_process_14d_labels.py的算法"""
    try:
        mapper = OptimizedHumanToRobotMapper(pybullet_manager)
        
        with h5py.File(input_path, 'r') as f:
            # 检查必要的数据集 - 使用与batch_process_14d_labels.py相同的数据结构
            # if 'transforms' not in f or 'leftThumbIntermediateBase' not in f['transforms'] or 'rightThumbIntermediateBase' not in f['transforms']:
            #     return False
            
            # # 检查是否有48维动作数据
            # if 'actions_48d' not in f:
            #     return False
            
            # 读取手部变换数据 - 使用绝对路径
            left_hand_transforms = f['/transforms/leftThumbIntermediateBase'][:]
            right_hand_transforms = f['/transforms/rightThumbIntermediateBase'][:]
            
            # 读取原48维动作数据
            actions_48d = f['actions_48d'][:]
            
            # 检查指尖数据是否存在
            has_left_thumb = 'leftThumbTip' in f['transforms']
            has_left_index = 'leftIndexFingerTip' in f['transforms']
            has_right_thumb = 'rightThumbTip' in f['transforms']
            has_right_index = 'rightIndexFingerTip' in f['transforms']
            
            num_frames = len(left_hand_transforms)
            
            # 用第0帧计算坐标变换
            human_init_transforms = {
                'left': left_hand_transforms[0],
                'right': right_hand_transforms[0]
            }
            mapper.calculate_coordinate_transform(human_init_transforms)
            
            # 计算动态夹爪范围参数 - 与batch_process_14d_labels.py保持一致
            left_distances = []
            right_distances = []
            
            for frame_idx in range(num_frames):
                try:
                    # 左手距离
                    if has_left_thumb and has_left_index:
                        left_thumb_pos = f['/transforms/leftThumbTip'][frame_idx][:3, 3]
                        left_index_pos = f['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                        left_distance = np.linalg.norm(left_thumb_pos - left_index_pos)
                        left_distances.append(left_distance)
                    
                    # 右手距离
                    if has_right_thumb and has_right_index:
                        right_thumb_pos = f['/transforms/rightThumbTip'][frame_idx][:3, 3]
                        right_index_pos = f['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                        right_distance = np.linalg.norm(right_thumb_pos - right_index_pos)
                        right_distances.append(right_distance)
                        
                except (KeyError, IndexError):
                    continue
            
            # 计算动态范围
            if left_distances:
                left_min_dist = np.min(left_distances)
                left_max_dist = np.max(left_distances)
                left_range = left_max_dist - left_min_dist
                if left_range < 1e-6:  # 防止范围过小
                    left_min_dist = left_min_dist - 0.005
                    left_max_dist = left_min_dist + 0.005
            else:
                left_min_dist, left_max_dist = 0.01, 0.2
            
            if right_distances:
                right_min_dist = np.min(right_distances)
                right_max_dist = np.max(right_distances)
                right_range = right_max_dist - right_min_dist
                if right_range < 1e-6:  # 防止范围过小
                    right_min_dist = right_min_dist - 0.005
                    right_max_dist = right_min_dist + 0.005
            else:
                right_min_dist, right_max_dist = 0.01, 0.2
            
            # 预分配数组
            actions_14d = np.zeros((num_frames, 14), dtype=np.float32)
            actions_62d = np.zeros((num_frames, 62), dtype=np.float32)
            
            # LOCF策略 - 与V1_notqdm.py保持一致
            last_successful_action = None
            
            # 逐帧处理 - 使用与batch_process_14d_labels.py相同的逻辑
            for frame_idx in range(num_frames):
                try:
                    # 读取手部变换
                    left_transform = left_hand_transforms[frame_idx]
                    right_transform = right_hand_transforms[frame_idx]
                    
                    # 坐标变换得到目标位置
                    left_pos, left_orn = mapper.transform_human_to_robot(left_transform, 'left')
                    right_pos, right_orn = mapper.transform_human_to_robot(right_transform, 'right')
                    
                    # IK求解
                    left_joint_angles = mapper.solve_ik(left_pos, left_orn, is_left_arm=True)
                    right_joint_angles = mapper.solve_ik(right_pos, right_orn, is_left_arm=False)
                    
                    # 计算夹爪状态
                    left_gripper_state = 0.5
                    right_gripper_state = 0.5
                    
                    try:
                        # 使用动态归一化计算夹爪状态
                        if has_left_thumb and has_left_index:
                            left_thumb_pos = f['/transforms/leftThumbTip'][frame_idx][:3, 3]
                            left_index_pos = f['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                            left_gripper_state = mapper.calculate_gripper_state_from_tips(
                                left_thumb_pos, left_index_pos,
                                left_min_dist, left_max_dist
                            )
                        
                        if has_right_thumb and has_right_index:
                            right_thumb_pos = f['/transforms/rightThumbTip'][frame_idx][:3, 3]
                            right_index_pos = f['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                            right_gripper_state = mapper.calculate_gripper_state_from_tips(
                                right_thumb_pos, right_index_pos,
                                right_min_dist, right_max_dist
                            )
                    except (KeyError, IndexError):
                        pass
                    
                    if left_joint_angles is not None and right_joint_angles is not None:
                        # 组装14维动作
                        current_action = np.concatenate([
                            left_joint_angles,           # 前6维：左臂关节
                            [left_gripper_state],       # 第7维：左夹爪
                            right_joint_angles,          # 后6维：右臂关节
                            [right_gripper_state]       # 第14维：右夹爪
                        ])
                        
                        actions_14d[frame_idx] = current_action
                        last_successful_action = current_action
                        
                        # 合并48维和14维为62维
                        actions_62d[frame_idx, :48] = actions_48d[frame_idx]
                        actions_62d[frame_idx, 48:62] = current_action
                    else:
                        # LOCF策略：使用最后一次成功的动作
                        if last_successful_action is not None:
                            actions_14d[frame_idx] = last_successful_action
                            actions_62d[frame_idx, :48] = actions_48d[frame_idx]
                            actions_62d[frame_idx, 48:62] = last_successful_action
                        else:
                            # 没有成功记录时的默认动作
                            default_action = np.array([0.0] * 12 + [0.5, 0.5])
                            actions_14d[frame_idx] = default_action
                            actions_62d[frame_idx, :48] = actions_48d[frame_idx]
                            actions_62d[frame_idx, 48:62] = default_action
                    
                except Exception:
                    # 异常情况也使用LOCF策略
                    if last_successful_action is not None:
                        actions_14d[frame_idx] = last_successful_action
                        actions_62d[frame_idx, :48] = actions_48d[frame_idx]
                        actions_62d[frame_idx, 48:62] = last_successful_action
                    else:
                        default_action = np.array([0.0] * 12 + [0.5, 0.5])
                        actions_14d[frame_idx] = default_action
                        actions_62d[frame_idx, :48] = actions_48d[frame_idx]
                        actions_62d[frame_idx, 48:62] = default_action
                    continue
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果 - 与batch_process_62d_optimized_back.py一致的格式
            with h5py.File(output_path, 'w') as out_f:
                # 保存62维动作数据
                out_f.create_dataset('actions_62d', data=actions_62d)
                # 同时保存14维动作数据（可选）
                out_f.create_dataset('actions_14d', data=actions_14d)
                # 保存成功率
                success_rate = np.sum(actions_14d[:, 0] != 0) / num_frames  # 简单的成功率计算
                out_f.create_dataset('success_rate', data=success_rate)
                
                # 保存必要的元数据
                out_f.attrs['source_file'] = input_path
                out_f.attrs['total_frames'] = num_frames
                # out_f.attrs['left_gripper_min_distance'] = left_min_dist
                # out_f.attrs['left_gripper_max_distance'] = left_max_dist
                # out_f.attrs['right_gripper_min_distance'] = right_min_dist
                # out_f.attrs['right_gripper_max_distance'] = right_max_dist
                # out_f.attrs['normalization_method'] = 'dynamic_per_trajectory'
            
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
    output_filename = f"{filename}.hdf5"
    # 构造输出路径
    output_path = os.path.join(output_root, os.path.dirname(relative_path), output_filename)
    return output_path

def worker_process(file_list, worker_id, input_root, output_root, progress_queue):
    """工作进程"""
    pybullet_manager = PyBulletManager()
    
    try:
        success_count = 0
        for input_file in file_list:
            output_file = map_input_to_output_path(input_file, input_root, output_root)
            
            success = process_hdf5_file_final(input_file, output_file, pybullet_manager)
            if success:
                success_count += 1
            
            # --- 核心修改：每处理完一个文件，就向队列发送一个信号 ---
            progress_queue.put(1)
        
        return success_count
        
    except Exception:
        # 即使出错，也要确保已经处理的文件进度被报告
        return 0
    finally:
        pybullet_manager.close()

def main_final_process():
    """使用pool.apply_async和Manager.Queue的主函数"""
    parser = argparse.ArgumentParser(description='62D动作预计算 (精确进度条版)')
    parser.add_argument('--input_root', type=str, default="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/yinzhenhan/data/dataset/Egodex-full", help='输入根目录')
    parser.add_argument('--output_root', type=str, required=True, help='输出根目录')
    parser.add_argument('--num_processes', type=int, default=10,
                       help='进程数量')
    
    args = parser.parse_args()


    # <<<--- 新增：定义要排除的目录 ---<<<
    EXCLUDE_DIRECTORIES = ['extra','part3','part4','part1','test']
    
    if not os.path.exists(args.input_root):
        print(f"❌ 输入根目录不存在: {args.input_root}")
        return
    
    print(f"🚀 开始递归处理目录: {args.input_root}")
    print(f"📁 输出根目录: {args.output_root}")
    
    # 递归查找所有HDF5文件
    print("🔍 搜索HDF5文件...")
    hdf5_files = find_hdf5_files_recursive(args.input_root,exclude_dirs=EXCLUDE_DIRECTORIES)
    
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
    
    # --- 核心修改：使用 Manager.Queue 和 tqdm ---
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
                pbar.update(1) # 更新进度条

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