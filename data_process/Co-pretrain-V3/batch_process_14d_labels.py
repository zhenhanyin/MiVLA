#!/usr/bin/env python3
"""
批量处理脚本：从EgoDex HDF5文件生成14维机械臂伪标签

输出格式：
- 前6维：左臂6个关节角度
- 第7维：左臂夹爪开合值
- 后7维：右臂6个关节角度 + 右臂夹爪开合值

使用与visualize_14d_final_thumbmoptim_delat_back.py相同的坐标变换和IK求解逻辑
"""

import numpy as np
import h5py
import os
import argparse
import pybullet as p
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple, List
from tqdm import tqdm
import json

# =============================================================================
# 核心配置类（从原脚本复制）
# =============================================================================

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
    
    def solve_ik(self, target_position, target_orientation, is_left_arm=True):
        """求解逆运动学"""
        try:
            arm_joints_indices = RobotConfig.LEFT_ARM_JOINTS if is_left_arm else RobotConfig.RIGHT_ARM_JOINTS
            end_effector_link = (self.link_indices['left_end_effector'] if is_left_arm
                            else self.link_indices['right_end_effector'])
            
            joint_states = p.getJointStates(self.robot_id, self.movable_joint_indices)
            #current_positions = [state[0] for state in joint_states]

            joint_angles_movable = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=self.joint_ll,
                upperLimits=self.joint_ul,
                jointRanges=self.joint_jr,
                #restPoses=current_positions, # 使用当前姿态作为参考，实现平滑过渡
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
        
        print(f"🔧 指尖数据可用性: 左拇指={has_left_thumb}, 左食指={has_left_index}, 右拇指={has_right_thumb}, 右食指={has_right_index}")
        
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
            print(f"🔧 左手夹爪动态范围: {left_min_dist:.4f} - {left_max_dist:.4f} (范围: {left_range:.4f})")
        else:
            gripper_ranges['left'] = {'min': 0.01, 'max': 0.2}
            print(f"🔧 左手夹爪使用默认范围: 0.01 - 0.2")
        
        # 右手夹爪范围
        if right_distances:
            right_min_dist = np.min(right_distances)
            right_max_dist = np.max(right_distances)
            right_range = right_max_dist - right_min_dist
            if right_range < 1e-6:  # 防止范围过小
                right_min_dist = right_min_dist - 0.005
                right_max_dist = right_min_dist + 0.005
            gripper_ranges['right'] = {'min': right_min_dist, 'max': right_max_dist}
            print(f"🔧 右手夹爪动态范围: {right_min_dist:.4f} - {right_max_dist:.4f} (范围: {right_range:.4f})")
        else:
            gripper_ranges['right'] = {'min': 0.01, 'max': 0.2}
            print(f"🔧 右手夹爪使用默认范围: 0.01 - 0.2")
        
        return gripper_ranges

# =============================================================================
# 批量处理核心类
# =============================================================================

class EgodexTo14DProcessor:
    """EgoDex到14维机械臂标签的批量处理器"""
    
    def __init__(self):
        self.pybullet_manager = PyBulletManager()
        self.mapper = OptimizedHumanToRobotMapper(self.pybullet_manager)
        
    def process_single_file(self, hdf5_file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        处理单个HDF5文件，生成14维标签序列
        
        返回:
            actions_14d: (num_frames, 14) 的14维动作数组
            metadata: 处理元数据
        """
        try:
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                num_frames = hdf5_file['/transforms/leftThumbIntermediateBase'].shape[0]
                
                # 检查指尖数据是否存在
                has_left_thumb = 'leftThumbTip' in hdf5_file['transforms']
                has_left_index = 'leftIndexFingerTip' in hdf5_file['transforms']
                has_right_thumb = 'rightThumbTip' in hdf5_file['transforms']
                has_right_index = 'rightIndexFingerTip' in hdf5_file['transforms']
                
                # 用第0帧计算坐标变换
                human_init_transforms = {
                    'left': hdf5_file['/transforms/leftThumbIntermediateBase'][0],
                    'right': hdf5_file['/transforms/rightThumbIntermediateBase'][0]
                }
                self.mapper.calculate_coordinate_transform(human_init_transforms)
                
                # 计算动态夹爪范围参数
                print("🔧 计算动态夹爪范围参数...")
                self.gripper_ranges = self.mapper.calculate_dynamic_gripper_ranges(hdf5_file, num_frames)
                
                # 初始化输出数组
                actions_14d = np.zeros((num_frames, 14), dtype=np.float32)
                processing_stats = {
                    'total_frames': num_frames,
                    'successful_frames': 0,
                    'failed_frames': 0,
                    'has_finger_data': {
                        'left_thumb': has_left_thumb,
                        'left_index': has_left_index,
                        'right_thumb': has_right_thumb,
                        'right_index': has_right_index
                    }
                }
                
                # 逐帧处理
                for frame_idx in range(num_frames):
                    try:
                        # 读取手部变换
                        left_transform = hdf5_file['/transforms/leftThumbIntermediateBase'][frame_idx]
                        right_transform = hdf5_file['/transforms/rightThumbIntermediateBase'][frame_idx]
                        
                        # 坐标变换得到目标位置
                        left_pos, left_orn = self.mapper.transform_human_to_robot_target(left_transform, 'left')
                        right_pos, right_orn = self.mapper.transform_human_to_robot_target(right_transform, 'right')
                        
                        # IK求解
                        left_joint_angles = self.pybullet_manager.solve_ik(left_pos, left_orn, is_left_arm=True)
                        right_joint_angles = self.pybullet_manager.solve_ik(right_pos, right_orn, is_left_arm=False)
                        
                        if left_joint_angles is None or right_joint_angles is None:
                            processing_stats['failed_frames'] += 1
                            continue
                        
                        # 计算夹爪状态
                        left_gripper_state = 0.5
                        right_gripper_state = 0.5
                        
                        try:
                            # 使用动态归一化计算夹爪状态
                            if has_left_thumb and has_left_index:
                                left_thumb_pos = hdf5_file['/transforms/leftThumbTip'][frame_idx][:3, 3]
                                left_index_pos = hdf5_file['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                                left_gripper_state = self.mapper.calculate_gripper_state_from_tips(
                                    left_thumb_pos, left_index_pos,
                                    self.gripper_ranges['left']['min'],
                                    self.gripper_ranges['left']['max']
                                )
                            
                            if has_right_thumb and has_right_index:
                                right_thumb_pos = hdf5_file['/transforms/rightThumbTip'][frame_idx][:3, 3]
                                right_index_pos = hdf5_file['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                                right_gripper_state = self.mapper.calculate_gripper_state_from_tips(
                                    right_thumb_pos, right_index_pos,
                                    self.gripper_ranges['right']['min'],
                                    self.gripper_ranges['right']['max']
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
                        processing_stats['failed_frames'] += 1
                        continue
                
                return actions_14d, processing_stats
                
        except Exception as e:
            print(f"处理文件失败 {hdf5_file_path}: {e}")
            return None, {'error': str(e)}
    
    def process_directory(self, input_dir: str, output_dir: str, file_pattern: str = "*.hdf5") -> Dict:
        """
        批量处理目录下的所有HDF5文件
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            
        返回:
            批量处理统计信息
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        hdf5_files = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not hdf5_files:
            print(f"在目录 {input_dir} 中未找到匹配 {file_pattern} 的文件")
            return {'total_files': 0, 'processed_files': 0, 'failed_files': 0}
        
        print(f"找到 {len(hdf5_files)} 个HDF5文件")
        
        batch_stats = {
            'total_files': len(hdf5_files),
            'processed_files': 0,
            'failed_files': 0,
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'file_results': []
        }
        
        for hdf5_file in tqdm(hdf5_files, desc="批量处理进度"):
            try:
                # 生成14维标签
                actions_14d, stats = self.process_single_file(hdf5_file)
                
                if actions_14d is None:
                    batch_stats['failed_files'] += 1
                    batch_stats['file_results'].append({
                        'file': hdf5_file,
                        'status': 'failed',
                        'error': stats.get('error', 'Unknown error')
                    })
                    continue
                
                # 保存结果
                filename = os.path.basename(hdf5_file).replace('.hdf5', '_14d.npy')
                output_path = os.path.join(output_dir, filename)
                np.save(output_path, actions_14d)
                
                # 保存统计信息
                stats_path = os.path.join(output_dir, filename.replace('.npy', '_stats.json'))
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                batch_stats['processed_files'] += 1
                batch_stats['total_frames'] += stats['total_frames']
                batch_stats['successful_frames'] += stats['successful_frames']
                batch_stats['failed_frames'] += stats['failed_frames']
                
                batch_stats['file_results'].append({
                    'file': hdf5_file,
                    'status': 'success',
                    'output_file': output_path,
                    'stats': stats
                })
                
            except Exception as e:
                batch_stats['failed_files'] += 1
                batch_stats['file_results'].append({
                    'file': hdf5_file,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # 保存批量处理统计
        batch_stats_path = os.path.join(output_dir, 'batch_stats.json')
        with open(batch_stats_path, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        
        return batch_stats
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'pybullet_manager'):
            self.pybullet_manager.close()

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量处理EgoDex HDF5文件生成14维机械臂伪标签")
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='输入HDF5文件目录')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--file_pattern', '-p', type=str, default='*.hdf5',
                        help='文件匹配模式 (默认: *.hdf5)')
    parser.add_argument('--single_file', '-s', type=str, default=None,
                        help='处理单个文件 (可选)')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = EgodexTo14DProcessor()
    
    try:
        if args.single_file:
            # 处理单个文件
            print(f"处理单个文件: {args.single_file}")
            actions_14d, stats = processor.process_single_file(args.single_file)
            
            if actions_14d is not None:
                # 保存结果
                filename = os.path.basename(args.single_file).replace('.hdf5', '_14d.npy')
                output_path = os.path.join(args.output_dir, filename)
                os.makedirs(args.output_dir, exist_ok=True)
                np.save(output_path, actions_14d)
                
                # 保存统计信息
                stats_path = os.path.join(args.output_dir, filename.replace('.npy', '_stats.json'))
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"✅ 处理完成!")
                print(f"   输出文件: {output_path}")
                print(f"   帧数: {actions_14d.shape[0]}")
                print(f"   成功率: {stats['successful_frames']}/{stats['total_frames']} "
                      f"({stats['successful_frames']/stats['total_frames']*100:.1f}%)")
            else:
                print(f"❌ 处理失败: {stats.get('error', 'Unknown error')}")
        
        else:
            # 批量处理目录
            print(f"批量处理目录: {args.input_dir}")
            print(f"输出目录: {args.output_dir}")
            print(f"文件模式: {args.file_pattern}")
            
            batch_stats = processor.process_directory(
                args.input_dir, 
                args.output_dir, 
                args.file_pattern
            )
            
            print(f"\n🎉 批量处理完成!")
            print(f"   处理文件: {batch_stats['processed_files']}/{batch_stats['total_files']}")
            print(f"   总帧数: {batch_stats['total_frames']}")
            print(f"   成功帧数: {batch_stats['successful_frames']}")
            print(f"   失败帧数: {batch_stats['failed_frames']}")
            if batch_stats['total_frames'] > 0:
                success_rate = batch_stats['successful_frames'] / batch_stats['total_frames'] * 100
                print(f"   整体成功率: {success_rate:.1f}%")
    
    finally:
        processor.close()

if __name__ == "__main__":
    main()