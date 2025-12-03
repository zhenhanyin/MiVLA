#!/usr/bin/env python3
"""
基于直接坐标变换的机器人可视化脚本
按照visualize_robot_14d_simple.py的模式，生成图像帧

EgoDex坐标系定义（以脚部为原点）：
+X: 右
+Y: 上  
-Z: 前

PyBullet右手坐标系定义：
+X: 右
+Y: 前
+Z: 上

坐标系变换关系：
EgoDex (x, y, z) -> PyBullet (x, -z, y)
"""

import pybullet as p
import pybullet_data
import numpy as np
import h5py
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2
from typing import Dict, Tuple, Optional

# ==============================================================================
# 从visualize_14d_final.py移植的核心组件
# ==============================================================================

class RobotConfig:
    """机器人配置类"""
    URDF_PATH = "/home/yinzhenhan/Human_Pre/dataset/robot/aloha-agilex/urdf/arx5_description_isaac.urdf"
    LEFT_ARM_JOINTS = [18, 19, 20, 21, 22, 23]
    RIGHT_ARM_JOINTS = [28, 29, 30, 31, 32, 33]
    LEFT_END_EFFECTOR_LINK = "fl_link6"
    RIGHT_END_EFFECTOR_LINK = "fr_link6"
    IK_MAX_ITERATIONS = 200  # 增加迭代次数以提高精度
    IK_THRESHOLD = 1e-3      # 降低阈值以提高精度
    WORKSPACE_SCALE = 1.2

class PyBulletManager:
    """真实的PyBullet环境管理器"""
    
    def __init__(self):
        self.robot_id = None
        self.link_indices = {}
        self.initial_poses = {}
        self.physics_client = None
        #--- [新增] 存储IK约束参数 ---
        self.movable_joint_indices = []
        self.joint_ll = []  # Lower Limits
        self.joint_ul = []  # Upper Limits
        self.joint_jr = []  # Joint Ranges
        
        self._setup_pybullet()
    
    def _setup_pybullet(self):
        """设置PyBullet环境"""
        self.physics_client = p.connect(p.DIRECT)  # 无头模式
        self.robot_id = p.loadURDF(RobotConfig.URDF_PATH, useFixedBase=True)
        self.link_indices = self._get_link_indices()
        
        # --- [新增] 初始化时获取并存储约束 ---
        self._get_joint_constraints()
        
        self._set_initial_poses()
    
    def _get_link_indices(self):
        """获取链接索引"""
        link_indices = {}
        link_indices['left_end_effector'] = 23
        link_indices['right_end_effector'] = 33
        return link_indices
    
    def _set_initial_poses(self):
        """设置机器人初始姿态"""
        # =========== [可选的初始姿态配置] ===========
        # 你可以修改这些关节角度来改变机器人的初始姿态
        # 关节角度顺序：[关节1, 关节2, 关节3, 关节4, 关节5, 关节6]
        # 角度单位：弧度
        
        # 选项1：自然下垂姿态 (当前使用的)
        left_initial_angles = [-0.6, 0, 0, 0, 0, 0]
        right_initial_angles = [0.6, 0, 0, 0, 0, 0]
        
        # 选项2：工作准备姿态 (手臂稍微抬起)
        # left_initial_angles = [0, -0.5, 0, 0, 0, 0]
        # right_initial_angles = [0, -0.5, 0, 0, 0, 0]
        
        # 选项3：前伸姿态 (手臂向前伸展)
        # left_initial_angles = [0, -1.2, 0, 0, 0, 0]
        # right_initial_angles = [0, -1.2, 0, 0, 0, 0]
        
        # 选项4：侧向姿态 (手臂向两侧展开)
        # left_initial_angles = [0, 0, 0, 0, 0, 0.8]
        # right_initial_angles = [0, 0, 0, 0, 0, -0.8]
        
        # 选项5：自定义姿态 (根据你的需求调整)
        # left_initial_angles = [关节1, 关节2, 关节3, 关节4, 关节5, 关节6]
        # right_initial_angles = [关节1, 关节2, 关节3, 关节4, 关节5, 关节6]
        
        # 应用关节角度
        for i, angle in enumerate(left_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.LEFT_ARM_JOINTS[i], angle)
        
        for i, angle in enumerate(right_initial_angles):
            p.resetJointState(self.robot_id, RobotConfig.RIGHT_ARM_JOINTS[i], angle)
        
        # 获取初始末端位姿
        left_pos, left_orn = p.getLinkState(self.robot_id, self.link_indices['left_end_effector'])[:2]
        right_pos, right_orn = p.getLinkState(self.robot_id, self.link_indices['right_end_effector'])[:2]
        
        # 打印初始姿态信息
        print(f"🔧 机器人初始姿态配置:")
        print(f"   左臂关节角度: {left_initial_angles}")
        print(f"   右臂关节角度: {right_initial_angles}")
        print(f"   左臂末端位置: {left_pos}")
        print(f"   右臂末端位置: {right_pos}")
        
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
            
            # 使用PyBullet的IK求解器 
            #全局求解: 它会为所有这些可动关节计算出一个目标姿态，以满足单个末端执行器的约束。
            joint_angles_movable = p.calculateInverseKinematics(
                self.robot_id,
                end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                
                # --- [核心优化] ---
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
            print(f"IK求解失败: {e}")
            return None
    
    def _get_joint_constraints(self):
        """一次性获取所有可动关节的约束信息并存储"""
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                self.movable_joint_indices.append(i)
                self.joint_ll.append(info[8])
                self.joint_ul.append(info[9])
                self.joint_jr.append(info[9] - info[8])
        print(f"🔧 成功获取并缓存了 {len(self.movable_joint_indices)} 个可动关节的约束。")
    
    def get_joint_positions(self, joint_angles, is_left_arm=True):
        """根据关节角度计算所有关节位置"""
        try:
            arm_joints_indices = RobotConfig.LEFT_ARM_JOINTS if is_left_arm else RobotConfig.RIGHT_ARM_JOINTS
            
            # 设置关节角度
            for i, angle in enumerate(joint_angles):
                p.resetJointState(self.robot_id, arm_joints_indices[i], angle)
            
            # 获取关节位置
            joint_positions = []
            
            # 添加基座位置（假设基座在原点）
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            joint_positions.append(np.array(base_pos))
            
            # 获取每个臂关节的位置
            for joint_idx in arm_joints_indices:
                joint_state = p.getLinkState(self.robot_id, joint_idx)
                joint_positions.append(np.array(joint_state[0]))  # 位置
            
            # 获取末端执行器位置
            end_effector_link = (self.link_indices['left_end_effector'] if is_left_arm
                            else self.link_indices['right_end_effector'])
            end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
            joint_positions.append(np.array(end_effector_state[0]))
            
            return np.array(joint_positions)
            
        except Exception as e:
            print(f"计算关节位置失败: {e}")
            return None
    
    def close(self):
        """关闭PyBullet连接"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)

class OptimizedHumanToRobotMapper:
    """
    基于已知坐标系关系的直接变换映射器。
    使用已知的坐标系统关系直接转换，无需复杂的矩阵求解。
    """
    def __init__(self, pybullet_manager: PyBulletManager):
        # 从PyBullet管理器获取初始位姿
        _, _, self.initial_poses = pybullet_manager.get_robot_info()
        
        # 直接坐标变换公式：EgoDex (x,y,z) -> PyBullet (-z,-x,y)
        self.coordinate_transform_matrix = np.array([
            [0, 0, -1],  # 人类-Z -> 机器人X (人类-Z向前，机器人+X向前)
            [-1, 0, 0],  # 人类-X -> 机器人Y (人类X向右，机器人-Y向左)
            [0, 1, 0]    # 人类Y -> 机器人Z (人类Y向上，机器人+Z向上)
        ])
        
        # --- [新增] 用于存储起始位置的变量 ---
        self.human_start_pos = {}
        self.robot_start_pos = {}
        
        print(f"🔧 坐标系变换: EgoDex (x,y,z) -> PyBullet (-z,-x,y)")
        print("🔧 相对偏移映射器初始化完成。")

    def calculate_gripper_state_from_tips(self, thumb_pos: np.ndarray, index_pos: np.ndarray, 
                                         min_distance: float, max_distance: float) -> float:
        """根据拇指和食指尖端位置计算夹爪状态 - 使用动态归一化参数"""
        try:
            distance = np.linalg.norm(thumb_pos - index_pos)
            gripper_state = np.clip((distance - min_distance) / (max_distance - min_distance), 0, 1)
            return gripper_state
        except Exception:
            return 0.5

    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """
        第0帧的校准检查器或调试器
        """
        # 记录人类在Egodex坐标系下的起始位置
        self.human_start_pos['left'] = human_init_transforms['left'][:3, 3]
        self.human_start_pos['right'] = human_init_transforms['right'][:3, 3]
        
        # 记录机器人在PyBullet坐标系下的起始位置
        self.robot_start_pos['left'] = self.initial_poses['left']['position']
        self.robot_start_pos['right'] = self.initial_poses['right']['position']
        
        print("🔧 校准完成:")
        print(f"  - 人类左手起始点: {self.human_start_pos['left']}")
        print(f"  - 机器人左手起始点: {self.robot_start_pos['left']}")
        print(f"  - 人类右手起始点: {self.human_start_pos['right']}")
        print(f"  - 机器人右手起始点: {self.robot_start_pos['right']}")

    def transform_egodex_vector_to_pybullet(self, egodex_vec):
        """
        只对方向/位移向量进行坐标系旋转变换。
        """
        return self.coordinate_transform_matrix @ egodex_vec

    def transform_human_to_robot_target(self, human_transform: np.ndarray, arm: str) -> Tuple[np.ndarray, Rotation]:
        """
        应用直接坐标变换，将人类手部位姿转换为机器人目标位姿。
        """
        # --- 位置变换 ---
        # 1. 获取人类当前位置
        human_current_pos = human_transform[:3, 3]
        
        # 2. 计算人类相对于其起点的位移 (Delta)
        human_delta = human_current_pos - self.human_start_pos[arm]
        
        # 3. 将这个位移向量转换到机器人坐标系
        robot_delta = self.transform_egodex_vector_to_pybullet(human_delta)
        
        # 4. 将机器人位移应用到机器人的起始位置上，得到最终目标
        robot_target_pos = self.robot_start_pos[arm] + robot_delta

        # --- 姿态变换（使用直接坐标变换）---
        # 从旋转矩阵创建旋转对象
        human_rot_matrix = human_transform[:3, :3]
        
        # 应用相同的坐标变换到旋转矩阵
        robot_rot_matrix = self.coordinate_transform_matrix @ human_rot_matrix
        robot_rot = Rotation.from_matrix(robot_rot_matrix)
        
        return robot_target_pos, robot_rot

class DirectTransformRobotVisualizer:
    def __init__(self, headless=True):
        """
        基于直接坐标变换的机器人可视化器 - 使用优化IK求解
        """
        print("🤖 初始化高精度PyBullet环境...")
        
        # 使用移植的PyBulletManager进行IK求解
        self.pybullet_manager = PyBulletManager()
        self.robot_id = self.pybullet_manager.robot_id
        
        # 初始化坐标变换器
        self.mapper = OptimizedHumanToRobotMapper(self.pybullet_manager)
        
        # 设置可视化环境
        if headless:
            self.physics_client = p.connect(p.DIRECT)  # 用于渲染的独立连接
        else:
            self.physics_client = p.connect(p.GUI)
            print("使用 GUI 模式。")

        # 设置场景
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载地面（纯白背景）
        try:
            self.plane_id = p.loadURDF("plane.urdf")
            print("✅ 地面加载成功")
        except Exception as e:
            print(f"⚠️ 警告: 无法加载地面: {e}")
            self.plane_id = None

        # 初始化关节信息
        self.left_arm_joints = RobotConfig.LEFT_ARM_JOINTS
        self.right_arm_joints = RobotConfig.RIGHT_ARM_JOINTS
        self.left_gripper_joints = [25, 26]
        self.right_gripper_joints = [35, 36]
        
        print("✅ 高精度直接坐标变换机器人可视化器初始化完成")
        print("🔧 使用PyBullet真实IK求解，200次迭代，1e-3精度阈值，带关节约束")

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

    def set_robot_from_hdf5(self, hdf5_file, frame_idx):
        """
        从HDF5文件读取手部数据，使用优化IK求解设置机器人姿态
        """
        try:
            # 读取左右手变换矩阵 - 使用正确的数据源
            left_transform = hdf5_file['/transforms/leftThumbIntermediateBase'][frame_idx]
            right_transform = hdf5_file['/transforms/rightThumbIntermediateBase'][frame_idx]
            
            # 调试：显示原始人类手部位置
            left_hand_pos = left_transform[:3, 3]
            right_hand_pos = right_transform[:3, 3]
            
            if frame_idx == 0:
                print(f"🔍 调试信息 - 帧{frame_idx}:")
                print(f"   人类左手原始位置 (EgoDex): {left_hand_pos}")
                print(f"   人类右手原始位置 (EgoDex): {right_hand_pos}")
            
            # 使用相对偏移坐标变换和IK求解
            left_pos, left_orn = self.mapper.transform_human_to_robot_target(left_transform, 'left')
            right_pos, right_orn = self.mapper.transform_human_to_robot_target(right_transform, 'right')
            
            if frame_idx == 0:
                print(f"   变换后左臂目标位置 (PyBullet): {left_pos}")
                print(f"   变换后右臂目标位置 (PyBullet): {right_pos}")
                print(f"   PyBullet X方向(向前): {left_pos[0]:.3f}, {right_pos[0]:.3f}")
                print(f"   PyBullet Y方向(向左): {left_pos[1]:.3f}, {right_pos[1]:.3f}")
                print(f"   PyBullet Z方向(向上): {left_pos[2]:.3f}, {right_pos[2]:.3f}")
                print(f"   检查：X值>0表示向前运动，Y值表示左右位置")
            
            # 使用PyBullet进行优化IK求解
            left_joint_angles = self.pybullet_manager.solve_ik(left_pos, left_orn, is_left_arm=True)
            right_joint_angles = self.pybullet_manager.solve_ik(right_pos, right_orn, is_left_arm=False)
            
            if left_joint_angles is None or right_joint_angles is None:
                print(f"⚠️ 帧 {frame_idx}: IK求解失败")
                return None
            
            # 设置左臂关节角度
            for i, joint_idx in enumerate(self.left_arm_joints):
                if i < len(left_joint_angles):
                    p.resetJointState(self.robot_id, joint_idx, left_joint_angles[i])
            
            # 设置右臂关节角度
            for i, joint_idx in enumerate(self.right_arm_joints):
                if i < len(right_joint_angles):
                    p.resetJointState(self.robot_id, joint_idx, right_joint_angles[i])
            
            # 使用动态归一化计算夹爪状态
            left_gripper_state = 0.5  # 默认值
            right_gripper_state = 0.5  # 默认值
            
            # 检查是否传递了夹爪范围参数
            if hasattr(self, 'gripper_ranges'):
                try:
                    # 计算左手夹爪状态
                    if 'leftThumbTip' in hdf5_file['transforms'] and 'leftIndexFingerTip' in hdf5_file['transforms']:
                        left_thumb_pos = hdf5_file['/transforms/leftThumbTip'][frame_idx][:3, 3]
                        left_index_pos = hdf5_file['/transforms/leftIndexFingerTip'][frame_idx][:3, 3]
                        left_gripper_state = self.mapper.calculate_gripper_state_from_tips(
                            left_thumb_pos, left_index_pos, 
                            self.gripper_ranges['left']['min'], 
                            self.gripper_ranges['left']['max']
                        )
                    
                    # 计算右手夹爪状态
                    if 'rightThumbTip' in hdf5_file['transforms'] and 'rightIndexFingerTip' in hdf5_file['transforms']:
                        right_thumb_pos = hdf5_file['/transforms/rightThumbTip'][frame_idx][:3, 3]
                        right_index_pos = hdf5_file['/transforms/rightIndexFingerTip'][frame_idx][:3, 3]
                        right_gripper_state = self.mapper.calculate_gripper_state_from_tips(
                            right_thumb_pos, right_index_pos,
                            self.gripper_ranges['right']['min'],
                            self.gripper_ranges['right']['max']
                        )
                        
                except (KeyError, IndexError) as e:
                    if frame_idx < 5:
                        print(f"⚠️ 帧{frame_idx}无法计算夹爪状态: {e}")
            
            # 设置夹爪关节位置（基于URDF的实际参数）
            # URDF分析: 夹爪是prismatic关节，范围[0, 0.04765]米
            # 夹爪状态: 0=完全闭合, 1=完全张开
            max_gripper_distance = 0.04765  # 最大张开距离4.765厘米
            
            left_gripper_distance = left_gripper_state * max_gripper_distance
            right_gripper_distance = right_gripper_state * max_gripper_distance
            
            # 调试信息（仅在前几帧显示）
            if frame_idx < 5:
                print(f"🔧 帧{frame_idx}夹爪状态: 左={left_gripper_state:.3f}, 右={right_gripper_state:.3f}")
                print(f"   左夹爪距离: {left_gripper_distance*1000:.1f}mm, 右夹爪距离: {right_gripper_distance*1000:.1f}mm")
            
            # 设置左臂夹爪关节 (fl_joint7, fl_joint8)
            p.resetJointState(self.robot_id, self.left_gripper_joints[0], left_gripper_distance)  # fl_joint7: Y轴正方向
            p.resetJointState(self.robot_id, self.left_gripper_joints[1], left_gripper_distance)  # fl_joint8: Y轴负方向
            
            # 设置右臂夹爪关节 (fr_joint7, fr_joint8)  
            p.resetJointState(self.robot_id, self.right_gripper_joints[0], right_gripper_distance)  # fr_joint7: Y轴正方向
            p.resetJointState(self.robot_id, self.right_gripper_joints[1], right_gripper_distance)  # fr_joint8: Y轴负方向
            
            # 获取末端执行器实际位置
            left_end_effector_state = p.getLinkState(self.robot_id, 23)  # fl_link6
            right_end_effector_state = p.getLinkState(self.robot_id, 33)  # fr_link6
            
            left_end_pos = np.array(left_end_effector_state[0])
            right_end_pos = np.array(right_end_effector_state[0])
            
            # 计算位置误差
            left_error = np.linalg.norm(left_end_pos - left_pos)
            right_error = np.linalg.norm(right_end_pos - right_pos)
            
            return {
                'left_target_pos': left_pos,
                'right_target_pos': right_pos,
                'left_actual_pos': left_end_pos,
                'right_actual_pos': right_end_pos,
                'left_joint_angles': left_joint_angles,
                'right_joint_angles': right_joint_angles,
                'left_error': left_error,
                'right_error': right_error
            }
            
        except Exception as e:
            print(f"设置机器人姿态失败: {e}")
            return None

    def render_to_frames(self, hdf5_file_path, output_dir, fps=30, max_frames=None):
        """渲染HDF5数据为图像帧序列"""
        
        # 检查文件是否存在
        if not os.path.exists(hdf5_file_path):
            print(f"❌ HDF5文件不存在: {hdf5_file_path}")
            return False
        
        # 打开HDF5文件
        try:
            hdf5_file = h5py.File(hdf5_file_path, 'r')
            num_frames = hdf5_file['/transforms/leftThumbIntermediateBase'].shape[0]
            
            if max_frames:
                num_frames = min(num_frames, max_frames)
            
            print(f"📊 HDF5文件总帧数: {hdf5_file['/transforms/leftThumbIntermediateBase'].shape[0]}")
            print(f"🎬 将渲染 {num_frames} 帧")
            
            # 用第0帧计算该视频专属的"翻译规则"
            human_init_transforms = {
                'left': hdf5_file['/transforms/leftThumbIntermediateBase'][0],
                'right': hdf5_file['/transforms/rightThumbIntermediateBase'][0]
            }
            self.mapper.calculate_coordinate_transform(human_init_transforms)
            
            # 计算动态夹爪范围参数
            print("🔧 计算动态夹爪范围参数...")
            self.gripper_ranges = self.calculate_dynamic_gripper_ranges(hdf5_file, num_frames)
            
        except Exception as e:
            print(f"❌ 打开HDF5文件失败: {e}")
            return False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图像参数
        width, height = 800, 600
        
        print(f"🎬 开始渲染帧序列...")
        print(f"📁 输出目录: {output_dir}")
        
        successful_frames = 0
        
        for frame_idx in tqdm(range(num_frames), desc="渲染进度"):
            try:
                # 从HDF5数据设置机器人姿态
                pose_info = self.set_robot_from_hdf5(hdf5_file, frame_idx)
                
                if pose_info is None:
                    continue
                
                # 设置第三视角相机参数 - 优化视角以显示向前运动
                camera_distance = 1.6
                camera_yaw = 180      # 60度偏航角，从侧面更好观察向前运动
                camera_pitch = -45   # 20度俯角，平视工作区域
                
                # 这里我们用标准的球坐标公式计算，先不考虑目标点偏移
                camera_x_offset = camera_distance * np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch))
                camera_y_offset = camera_distance * np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch))
                camera_z_offset = camera_distance * np.sin(np.radians(camera_pitch))
                
                # 调整相机目标点 - 聚焦在工作区域
                camera_target = [0.4, 0, 0.9]  # 指向机器人工作区域中心
                camera_pos = [
                    camera_target[0] + camera_x_offset,
                    camera_target[1] + camera_y_offset,
                    camera_target[2] - camera_z_offset  # 注意这里是减法，因为z_offset是负的
                ]

                # 5. 定义相机的“上”方向
                up_vector = [0, 0, 1] # Z轴向上
                
                # 调试信息 - 显示相机参数
                if frame_idx == 0:
                    print(f"📹 相机调试信息:")
                    print(f"   相机距离: {camera_distance}")
                    print(f"   偏航角: {camera_yaw}°")
                    print(f"   俯仰角: {camera_pitch}°")
                    print(f"   相机位置: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
                    print(f"   目标位置: [{camera_target[0]:.2f}, {camera_target[1]:.2f}, {camera_target[2]:.2f}]")
                    print(f"   上方向: {up_vector}")
                
                # 计算视图矩阵
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=camera_pos,
                    cameraTargetPosition=camera_target,
                    cameraUpVector=up_vector
                )
                
                # 投影矩阵
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=width/height, nearVal=0.1, farVal=50.0
                )
                
                # 获取图像
                img_arr = p.getCameraImage(
                    width=width, height=height,
                    viewMatrix=view_matrix, 
                    projectionMatrix=proj_matrix,
                )
                
                # 检查图像数据
                if img_arr[2] is not None and len(img_arr[2]) > 0:
                    # 转换图像格式
                    rgb_image = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
                    
                    if rgb_image is None or rgb_image.size == 0:
                        print(f"警告: 帧 {frame_idx} 图像数据为空")
                        continue
                    
                    # 保存帧
                    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    successful_frames += 1
                    
                    # 只在第一帧和每20帧打印详细信息
                    if frame_idx == 0 or frame_idx % 20 == 0:
                        print(f"\n📹 帧 {frame_idx} - 优化IK求解:")
                        print(f"  左臂目标位置: ({pose_info['left_target_pos'][0]:.3f}, {pose_info['left_target_pos'][1]:.3f}, {pose_info['left_target_pos'][2]:.3f})")
                        print(f"  右臂目标位置: ({pose_info['right_target_pos'][0]:.3f}, {pose_info['right_target_pos'][1]:.3f}, {pose_info['right_target_pos'][2]:.3f})")
                        print(f"  左臂实际位置: ({pose_info['left_actual_pos'][0]:.3f}, {pose_info['left_actual_pos'][1]:.3f}, {pose_info['left_actual_pos'][2]:.3f})")
                        print(f"  右臂实际位置: ({pose_info['right_actual_pos'][0]:.3f}, {pose_info['right_actual_pos'][1]:.3f}, {pose_info['right_actual_pos'][2]:.3f})")
                        print(f"  左臂误差: {pose_info['left_error']:.6f}m")
                        print(f"  右臂误差: {pose_info['right_error']:.6f}m")
                        print(f"  左臂关节角度: {[f'{a:.3f}' for a in pose_info['left_joint_angles']]}")
                        print(f"  右臂关节角度: {[f'{a:.3f}' for a in pose_info['right_joint_angles']]}")
                        
            except Exception as e:
                print(f"渲染帧 {frame_idx} 时出错: {e}")
                continue
        
        # 关闭HDF5文件
        hdf5_file.close()
        
        print(f"🎉 渲染完成！成功渲染 {successful_frames}/{num_frames} 帧")
        print(f"📁 帧序列保存至: {output_dir}")
        
        return successful_frames > 0

    def close(self):
        """关闭所有PyBullet连接"""
        print("🔌 关闭PyBullet环境")
        if hasattr(self, 'pybullet_manager') and self.pybullet_manager is not None:
            self.pybullet_manager.close()
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)


def main():
    """主函数"""
    print("🚀 启动高精度机器人轨迹复现可视化器")
    
    # 用户配置
    hdf5_file = "/home/yinzhenhan/Human_Pre/dataset/Egodex-full/part2/basic_fold/0.hdf5"
    output_dir = "./visual_test/high_precision_frames_basic_final-0"
    fps = 30
    max_frames = None  # 不限制渲染帧数，渲染全部帧
    use_gui = False  # 使用无头模式
    
    # 检查文件
    if not os.path.exists(hdf5_file):
        print(f"❌ HDF5文件不存在: {hdf5_file}")
        return
    
    print(f"📂 HDF5文件: {hdf5_file}")
    print(f"🎯 输出目录: {output_dir}")
    print(f"🎬 渲染帧数: {max_frames}")
    print(f"📐 坐标变换: EgoDex (x,y,z) -> PyBullet (-z,-x,y)")
    
    # 创建高精度可视化器
    visualizer = DirectTransformRobotVisualizer(headless=not use_gui)
    
    if visualizer.robot_id is None:
        print("❌ 机器人加载失败，退出")
        return
    
    # 渲染帧序列
    success = visualizer.render_to_frames(hdf5_file, output_dir, fps=fps, max_frames=max_frames)
    
    if success:
        print("✅ 高精度机器人轨迹复现成功！")
        print("🔧 技术特点：")
        print("   - 基于相对偏移的精确变换")
        print("   - EgoDex坐标系: +X右, +Y上, -Z前")
        print("   - PyBullet坐标系: +X前, +Y左, +Z上")
        print("   - 变换公式: EgoDex (x,y,z) -> PyBullet (-z,-x,y)")
        print("   - 使用PyBullet真实URDF模型")
        print("   - 优化IK求解: 200次迭代，1e-3阈值，带关节约束")
        print("   - 实时误差监控和显示")
        print("   - 🆕 夹爪动态归一化计算（基于URDF修正）")
        print("     * 基于拇指和食指尖端距离计算夹爪开合程度")
        print("     * 每个轨迹使用动态归一化范围，充分利用[0,1]区间")
        print("     * URDF分析: 夹爪为prismatic关节，范围[0, 0.04765]米")
        print("     * 正确映射: 夹爪状态0=完全闭合(0mm), 1=完全张开(47.65mm)")
        print("     * 左右夹爪分别使用Y轴正/负方向的两个关节实现开合")
    else:
        print("❌ 可视化失败")
    
    # 关闭环境
    visualizer.close()


if __name__ == "__main__":
    main()