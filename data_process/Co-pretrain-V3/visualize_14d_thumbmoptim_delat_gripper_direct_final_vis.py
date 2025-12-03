#!/usr/bin/env python3
"""
独立的可视化预览脚本。

功能：
1. 读取一个原始的Egodex HDF5文件。
2. 应用与最终数据生成脚本完全相同的坐标系对齐和变换逻辑。
3. 在一个交互式的3D图表中，动画展示转换后得到的、将在PyBullet世界坐标系中呈现的
   机器人双臂末端执行器的目标运动轨迹。

这个脚本是数据生成流程的“质检工具”和“预览器”。
"""

import plotly.graph_objects as go
import numpy as np
import h5py
import os
import argparse
import pybullet as p
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple

# ==============================================================================
# 步骤 1: 复用核心逻辑和配置
# 为了让此脚本独立，我们将数据生成脚本中的核心类和配置直接复制过来。
# ==============================================================================

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

    # --- [新增] 获取并存储关节约束的方法 ---
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
        

    
    def _get_link_indices(self):
        """获取末端执行器链接索引"""
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
        


        # 左臂目标位置: [0.35028314 0.21704594 0.94555875]
        # 右臂目标位置: [ 0.35024546 -0.22481277  0.94455875]


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

            # # --- [诊断打印 1] 检查列表长度 ---
            # print(f"--- IK 求解诊断 (臂: {'左' if is_left_arm else '右'}) ---")
            # print(f"目标位置: {[f'{v:.3f}' for v in target_position]}")
            # print(f"总可动关节数: {len(self.movable_joint_indices)}")
            # print(f"  - lowerLimits 长度: {len(self.joint_ll)}")
            # print(f"  - upperLimits 长度: {len(self.joint_ul)}")
            # print(f"  - jointRanges 长度: {len(self.joint_jr)}")               #38
            # print(f"  - restPoses   长度: {len(current_positions)}")
            # ------------------------------------


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

# PyBulletManagerMock类已删除，现在使用真实的PyBulletManager

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

class OptimizedHumanToRobotMapper:
    """
    基于已知坐标系关系的直接变换映射器。
    使用已知的坐标系统关系直接转换，无需复杂的矩阵求解。
    """
    def __init__(self, pybullet_manager: PyBulletManager):

        # 从真实的PyBulletManager获取初始位姿
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




    def calculate_coordinate_transform(self, human_init_transforms: Dict[str, np.ndarray]):
        """
        第0帧的校准检查器”或“调试器
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

    # def transform_egodex_to_pybullet(self, egodex_pos):
    #     """
    #     应用直接坐标变换：EgoDex (x,y,z) -> PyBullet (-z,-x,y)
    #     """
    #     x, y, z = egodex_pos
    #     pybullet_pos = np.array([-z, -x, y])
    #     return pybullet_pos

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

    def calculate_gripper_state_from_tips(self, thumb_pos: np.ndarray, index_pos: np.ndarray, 
                                         min_distance: float = 0.01, max_distance: float = 0.2) -> float:
        """根据拇指和食指尖端位置计算夹爪状态"""
        try:
            distance = np.linalg.norm(thumb_pos - index_pos)
            gripper_state = np.clip((distance - min_distance) / (max_distance - min_distance), 0, 1)
            return gripper_state
        except Exception:
            return 0.5

# ==============================================================================
# 步骤 2: Plotly 可视化主逻辑
# ==============================================================================

def main(human_file: str):
    """主函数，执行数据加载、转换和可视化。"""
    
    if not os.path.exists(human_file):
        print(f"❌ 错误: 文件不存在 '{human_file}'")
        return

    human_root = h5py.File(human_file, 'r')
    num_frames = human_root['/transforms/leftThumbIntermediateBase'].shape[0]
    
    print(f"📊 正在加载和转换 {num_frames} 帧人类数据从: {human_file}")
    print(f"🔧 使用数据源: leftThumbIntermediateBase 和 rightThumbIntermediateBase")

    # --- 1. [核心] 计算转换后的机器人目标轨迹 ---
    # 使用真实的PyBulletManager进行IK求解和初始位姿计算
    pybullet_manager = PyBulletManager()
    
    # 现在使用真实的PyBulletManager，它会计算实际的初始位姿
    mapper = OptimizedHumanToRobotMapper(pybullet_manager)
    
    # 用第0帧计算该视频专属的"翻译规则"
    human_init_transforms = {
        'left': human_root['/transforms/leftThumbIntermediateBase'][0],
        'right': human_root['/transforms/rightThumbIntermediateBase'][0]
    }
    mapper.calculate_coordinate_transform(human_init_transforms)
    
    # 调试：显示第0帧的转换结果
    left_pos_0, _ = mapper.transform_human_to_robot_target(human_init_transforms['left'], 'left')
    right_pos_0, _ = mapper.transform_human_to_robot_target(human_init_transforms['right'], 'right')
    print(f"🔧 第0帧转换结果:")
    print(f"   左臂: {left_pos_0}")
    print(f"   右臂: {right_pos_0}")
    print(f"🔧 PyBulletManager配置:")
    print(f"   左臂: {mapper.initial_poses['left']['position']}")
    print(f"   右臂: {mapper.initial_poses['right']['position']}")
    
    # 调试：检查直接坐标变换矩阵
    print(f"🔍 直接坐标变换矩阵调试:")
    print(f"   坐标变换矩阵: {mapper.coordinate_transform_matrix}")
    print(f"   变换公式: EgoDex (x,y,z) -> PyBullet (-z,-x,y)")
    
    # 验证第0帧转换结果
    print(f"🔧 第0帧验证:")
    human_left_pos = human_init_transforms['left'][:3, 3]
    human_right_pos = human_init_transforms['right'][:3, 3]
    print(f"   人类左手原始位置: {human_left_pos}")
    print(f"   人类右手原始位置: {human_right_pos}")
    print(f"   机器人左手转换后: {left_pos_0}")
    print(f"   机器人右手转换后: {right_pos_0}")
    
    # 逐帧转换整个人类轨迹，得到机器人的目标轨迹
    robot_target_frames = []
    joint_positions_frames = []  # 存储每帧的关节位置
    gripper_states = []  # 存储每帧的夹爪状态和指尖位置
    
    print(f"🔧 开始IK求解和关节位置计算...")
    print(f"📊 总共需要处理 {num_frames} 帧，这可能需要一些时间...")
    
    # 检查指尖数据是否存在
    has_left_thumb = 'leftThumbTip' in human_root['transforms']
    has_left_index = 'leftIndexFingerTip' in human_root['transforms']
    has_right_thumb = 'rightThumbTip' in human_root['transforms']
    has_right_index = 'rightIndexFingerTip' in human_root['transforms']
    
    print(f"🔧 指尖数据可用性:")
    print(f"   左拇指: {has_left_thumb}")
    print(f"   左食指: {has_left_index}")
    print(f"   右拇指: {has_right_thumb}")
    print(f"   右食指: {has_right_index}")
    
    # 处理所有帧，确保关节位置计算完整
    # 添加IK成功率统计
    ik_success_count = 0
    ik_failed_frames = []
    
    for i in range(num_frames):
        # 显示进度
        if i % 20 == 0 or i == num_frames - 1:
            print(f"   进度: {i+1}/{num_frames} 帧 ({(i+1)/num_frames*100:.1f}%)")
        left_transform = human_root['/transforms/leftThumbIntermediateBase'][i]
        right_transform = human_root['/transforms/rightThumbIntermediateBase'][i]
        
        # 坐标变换得到目标位置
        left_pos, left_orn = mapper.transform_human_to_robot_target(left_transform, 'left')
        right_pos, right_orn = mapper.transform_human_to_robot_target(right_transform, 'right')
        
        # 使用PyBullet进行IK求解
        left_joint_angles = pybullet_manager.solve_ik(left_pos, left_orn, is_left_arm=True)
        right_joint_angles = pybullet_manager.solve_ik(right_pos, right_orn, is_left_arm=False)
        
        # 统计IK成功率
        if left_joint_angles is not None and right_joint_angles is not None:
            ik_success_count += 1
        else:
            ik_failed_frames.append(i)
        
        # 计算关节位置
        left_joint_positions = None
        right_joint_positions = None
        
        if left_joint_angles is not None:
            left_joint_positions = pybullet_manager.get_joint_positions(left_joint_angles, is_left_arm=True)
            # 计算末端执行器实际位置与目标位置的差异
            if left_joint_positions is not None and len(left_joint_positions) > 0:
                actual_end_effector_pos = left_joint_positions[-1]  # 最后一个是末端执行器
                position_error = np.linalg.norm(actual_end_effector_pos - left_pos)
                if i < 5:  # 只显示前5帧的调试信息
                    print(f"   左臂末端执行器误差: {position_error:.6f}")
                    print(f"     目标位置: {left_pos}")
                    print(f"     实际位置: {actual_end_effector_pos}")
        
        if right_joint_angles is not None:
            right_joint_positions = pybullet_manager.get_joint_positions(right_joint_angles, is_left_arm=False)
            # 计算末端执行器实际位置与目标位置的差异
            if right_joint_positions is not None and len(right_joint_positions) > 0:
                actual_end_effector_pos = right_joint_positions[-1]  # 最后一个是末端执行器
                position_error = np.linalg.norm(actual_end_effector_pos - right_pos)
                if i < 5:  # 只显示前5帧的调试信息
                    print(f"   右臂末端执行器误差: {position_error:.6f}")
                    print(f"     目标位置: {right_pos}")
                    print(f"     实际位置: {actual_end_effector_pos}")
        
        # 存储结果
        robot_target_frames.append(np.array([left_pos, right_pos]))
        joint_positions_frames.append({
            'left': left_joint_positions,
            'right': right_joint_positions,
            'left_angles': left_joint_angles,
            'right_angles': right_joint_angles
        })
        
        # 计算夹爪状态和指尖位置
        frame_gripper_data = {
            'left_gripper_state': 0.5,
            'right_gripper_state': 0.5,
            'left_thumb_pos': None,
            'left_index_pos': None,
            'right_thumb_pos': None,
            'right_index_pos': None
        }
        
        # 读取指尖数据并计算夹爪状态
        try:
            # 左手夹爪状态
            if has_left_thumb and has_left_index:
                left_thumb_pos = human_root['/transforms/leftThumbTip'][i][:3, 3]
                left_index_pos = human_root['/transforms/leftIndexFingerTip'][i][:3, 3]


                # --- [核心修正] ---
                # 同样使用相对偏移逻辑来计算指尖在机器人空间中的位置
                # 1. 计算指尖相对于人类手部起点的偏移
                left_thumb_delta = left_thumb_pos - mapper.human_start_pos['left']
                left_index_delta = left_index_pos - mapper.human_start_pos['left']
                
                # 2. 转换偏移向量
                left_thumb_robot_delta = mapper.transform_egodex_vector_to_pybullet(left_thumb_delta)
                left_index_robot_delta = mapper.transform_egodex_vector_to_pybullet(left_index_delta)
                
                # 3. 应用到机器人手部起点
                left_thumb_robot_pos = mapper.robot_start_pos['left'] + left_thumb_robot_delta
                left_index_robot_pos = mapper.robot_start_pos['left'] + left_index_robot_delta


                # 计算夹爪状态
                frame_gripper_data['left_gripper_state'] = mapper.calculate_gripper_state_from_tips(
                    left_thumb_pos, left_index_pos
                )
                frame_gripper_data['left_thumb_pos'] = left_thumb_robot_pos
                frame_gripper_data['left_index_pos'] = left_index_robot_pos
            
            # 右手夹爪状态
            if has_right_thumb and has_right_index:
                right_thumb_pos = human_root['/transforms/rightThumbTip'][i][:3, 3]
                right_index_pos = human_root['/transforms/rightIndexFingerTip'][i][:3, 3]

                # --- [核心修正] ---
                # 同样使用相对偏移逻辑
                right_thumb_delta = right_thumb_pos - mapper.human_start_pos['right']
                right_index_delta = right_index_pos - mapper.human_start_pos['right']
                
                right_thumb_robot_delta = mapper.transform_egodex_vector_to_pybullet(right_thumb_delta)
                right_index_robot_delta = mapper.transform_egodex_vector_to_pybullet(right_index_delta)
                
                right_thumb_robot_pos = mapper.robot_start_pos['right'] + right_thumb_robot_delta
                right_index_robot_pos = mapper.robot_start_pos['right'] + right_index_robot_delta

                # 计算夹爪状态
                frame_gripper_data['right_gripper_state'] = mapper.calculate_gripper_state_from_tips(
                    right_thumb_pos, right_index_pos
                )
                frame_gripper_data['right_thumb_pos'] = right_thumb_robot_pos
                frame_gripper_data['right_index_pos'] = right_index_robot_pos
                
        except (KeyError, IndexError) as e:
            if i < 5:  # 只显示前5帧的调试信息
                print(f"   警告: 无法读取指尖数据: {e}")
        
        gripper_states.append(frame_gripper_data)
        
        # 调试前几帧
        if i < 5:
            print(f"🔧 帧{i}:")
            print(f"   左臂目标位置: {left_pos}")
            print(f"   右臂目标位置: {right_pos}")
            if left_joint_angles is not None:
                print(f"   左臂关节角度: {left_joint_angles}")
            if right_joint_angles is not None:
                print(f"   右臂关节角度: {right_joint_angles}")
            print(f"   左夹爪状态: {frame_gripper_data['left_gripper_state']:.3f}")
            print(f"   右夹爪状态: {frame_gripper_data['right_gripper_state']:.3f}")
            if frame_gripper_data['left_thumb_pos'] is not None:
                print(f"   左拇指位置: {frame_gripper_data['left_thumb_pos']}")
                print(f"   左食指位置: {frame_gripper_data['left_index_pos']}")
    
    print(f"🔧 IK求解完成，处理了 {num_frames} 帧")
    
    # 计算IK成功率
    ik_success_rate = ik_success_count / num_frames * 100 if num_frames > 0 else 0
    print(f"📊 IK求解统计:")
    print(f"   成功帧数: {ik_success_count}/{num_frames}")
    print(f"   成功率: {ik_success_rate:.1f}%")
    if ik_failed_frames:
        print(f"   失败帧: {ik_failed_frames[:10]}{'...' if len(ik_failed_frames) > 10 else ''}")
    else:
        print(f"   失败帧: 无")
        
    # --- 2. 创建并显示 Plotly 3D 动画图表 ---
    print("📈 正在生成交互式预览图表...")
    fig = go.Figure()
    
    # 绘制初始帧 (第0帧) 的数据作为静态背景
    fig.add_trace(go.Scatter3d(
        x=robot_target_frames[0][:, 0], y=robot_target_frames[0][:, 1], z=robot_target_frames[0][:, 2],
        mode='markers',
        name='Robot Target EEF Positions',
        marker=dict(size=8, color='red', symbol='cross')
    ))
    
    # 如果有有效关节位置，绘制初始帧的关节点
    if joint_positions_frames[0]['left'] is not None:
        left_joints = joint_positions_frames[0]['left']
        left_angles = joint_positions_frames[0]['left_angles']
        # 创建悬停文本
        left_hover_text = []
        for j in range(len(left_joints)):
            if j == 0:
                left_hover_text.append(f"左臂基座<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})")
            elif j == len(left_joints) - 1:
                joint_value = left_angles[j-1] if left_angles is not None and j-1 < len(left_angles) else 0.0
                left_hover_text.append(f"左臂末端执行器<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
            else:
                joint_value = left_angles[j-1] if left_angles is not None and j-1 < len(left_angles) else 0.0
                left_hover_text.append(f"左臂关节{j}<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
        
        fig.add_trace(go.Scatter3d(
            x=left_joints[:, 0], y=left_joints[:, 1], z=left_joints[:, 2],
            mode='markers+lines',
            name='Left Arm Joints ',
            marker=dict(size=6, color='blue'),
            line=dict(color='blue', width=2),
            text=left_hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    if joint_positions_frames[0]['right'] is not None:
        right_joints = joint_positions_frames[0]['right']
        right_angles = joint_positions_frames[0]['right_angles']
        # 创建悬停文本
        right_hover_text = []
        for j in range(len(right_joints)):
            if j == 0:
                right_hover_text.append(f"右臂基座<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})")
            elif j == len(right_joints) - 1:
                joint_value = right_angles[j-1] if right_angles is not None and j-1 < len(right_angles) else 0.0
                right_hover_text.append(f"右臂末端执行器<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
            else:
                joint_value = right_angles[j-1] if right_angles is not None and j-1 < len(right_angles) else 0.0
                right_hover_text.append(f"右臂关节{j}<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
        
        fig.add_trace(go.Scatter3d(
            x=right_joints[:, 0], y=right_joints[:, 1], z=right_joints[:, 2],
            mode='markers+lines',
            name='Right Arm Joints ',
            marker=dict(size=6, color='green'),
            line=dict(color='green', width=2),
            text=right_hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # 绘制初始帧的指尖位置
    gripper_data_0 = gripper_states[0]
    
    # 左手指尖
    if gripper_data_0['left_thumb_pos'] is not None and gripper_data_0['left_index_pos'] is not None:
        left_thumb_pos = gripper_data_0['left_thumb_pos']
        left_index_pos = gripper_data_0['left_index_pos']
        
        # 左拇指
        fig.add_trace(go.Scatter3d(
            x=[left_thumb_pos[0]], y=[left_thumb_pos[1]], z=[left_thumb_pos[2]],
            mode='markers',
            name='Left Thumb ',
            marker=dict(size=8, color='orange', symbol='diamond'),
            text=[f"左拇指<br>坐标: ({left_thumb_pos[0]:.3f}, {left_thumb_pos[1]:.3f}, {left_thumb_pos[2]:.3f})<br>夹爪状态: {gripper_data_0['left_gripper_state']:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # 左食指
        fig.add_trace(go.Scatter3d(
            x=[left_index_pos[0]], y=[left_index_pos[1]], z=[left_index_pos[2]],
            mode='markers',
            name='Left Index ',
            marker=dict(size=8, color='purple', symbol='diamond'),
            text=[f"左食指<br>坐标: ({left_index_pos[0]:.3f}, {left_index_pos[1]:.3f}, {left_index_pos[2]:.3f})<br>夹爪状态: {gripper_data_0['left_gripper_state']:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # 右手指尖
    if gripper_data_0['right_thumb_pos'] is not None and gripper_data_0['right_index_pos'] is not None:
        right_thumb_pos = gripper_data_0['right_thumb_pos']
        right_index_pos = gripper_data_0['right_index_pos']
        
        # 右拇指
        fig.add_trace(go.Scatter3d(
            x=[right_thumb_pos[0]], y=[right_thumb_pos[1]], z=[right_thumb_pos[2]],
            mode='markers',
            name='Right Thumb ',
            marker=dict(size=8, color='red', symbol='diamond'),
            text=[f"右拇指<br>坐标: ({right_thumb_pos[0]:.3f}, {right_thumb_pos[1]:.3f}, {right_thumb_pos[2]:.3f})<br>夹爪状态: {gripper_data_0['right_gripper_state']:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # 右食指
        fig.add_trace(go.Scatter3d(
            x=[right_index_pos[0]], y=[right_index_pos[1]], z=[right_index_pos[2]],
            mode='markers',
            name='Right Index',
            marker=dict(size=8, color='cyan', symbol='diamond'),
            text=[f"右食指<br>坐标: ({right_index_pos[0]:.3f}, {right_index_pos[1]:.3f}, {right_index_pos[2]:.3f})<br>夹爪状态: {gripper_data_0['right_gripper_state']:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # 自动计算合适的坐标轴范围以完整显示整个轨迹
    all_points = np.concatenate(robot_target_frames)
    min_coords, max_coords = np.min(all_points, axis=0), np.max(all_points, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.6
    
    # Z轴从地面开始显示（机械臂底座高度为0.85）
    z_min = 0  # 
    z_max = max_coords[2] + 0.2  # 顶部留一些空间
    
    # Y轴范围扩充到(-0.8, 0.8)
    y_min = -0.8
    y_max = 0.8
    
    # 设置图表布局和相机
    # 显示IK成功率信息
    failed_frames_text = f", 失败帧: {ik_failed_frames[:5]}{'...' if len(ik_failed_frames) > 5 else ''}" if ik_failed_frames else ""
    
    fig.update_layout(
        title=f"Preview of Retargeted Trajectory in PyBullet's World Frame<br>"
              f"<sub>IK求解成功率: {ik_success_rate:.1f}% ({ik_success_count}/{num_frames}帧{failed_frames_text})</sub>",
        scene=dict(
            # 隐藏所有坐标轴标题，让画面更简洁
            # xaxis_title='X (Robot Forward)',
            # yaxis_title='Y (Robot Left)',
            # zaxis_title='Z (Robot Up)',
            xaxis=dict(
                range=[center[0] - max_range, center[0] + max_range+0.5],
                showgrid=False,      # 隐藏X轴网格
                showbackground=False, # 隐藏X轴背景
                zeroline=False,      # 隐藏X轴零线
                showticklabels=False, # 隐藏X轴数值
                showline=False,      # 隐藏X轴线
                showspikes=False     # 隐藏X轴尖峰
            ),
            yaxis=dict(
                range=[y_min, y_max],
                showgrid=False,      # 隐藏Y轴网格
                showbackground=False, # 隐藏Y轴背景
                zeroline=False,      # 隐藏Y轴零线
                showticklabels=False, # 隐藏Y轴数值
                showline=False,      # 隐藏Y轴线
                showspikes=False     # 隐藏Y轴尖峰
            ),
            zaxis=dict(
                range=[z_min, z_max],
                showgrid=False,      # 隐藏Z轴网格
                showbackground=False, # 隐藏Z轴背景
                zeroline=False,      # 隐藏Z轴零线
                showticklabels=False, # 隐藏Z轴数值
                showline=False,      # 隐藏Z轴线
                showspikes=False     # 隐藏Z轴尖峰
            ),
            aspectmode='cube', # 保持XYZ轴比例为1:1:1，无变形
            camera=dict( # 设置一个舒适的初始观察视角
                up=dict(x=0, y=0, z=1), 
                eye=dict(x=center[0] - 1.2, y=center[1] - 1.2, z=center[2] + 1.2)
            )
        ),
        # legend=dict(x=0.01, y=0.99),  # 隐藏图例，让画面更简洁
        # 添加播放/暂停按钮（30帧每秒 = 33.33ms每帧）
        updatemenus=[{
            'type': 'buttons', 'direction': 'left', 'showactive': True,
            'buttons': [
                {'label': 'Play (30 FPS)', 'method': 'animate', 'args': [None, {'frame': {'duration': 33.33, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 'transition': {'duration': 0}}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }],
        # 添加时间轴滑块
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '}, 'pad': {'t': 50},
            'steps': [{'label': str(i), 'method': 'animate', 'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}]} for i in range(num_frames)]
        }]
    )
    
    # 创建动画的每一帧
    fig_frames = []
    for i in range(num_frames):
        frame_data = robot_target_frames[i]
        joint_data = joint_positions_frames[i]
        gripper_data = gripper_states[i]
        
        # 每帧的数据包含：末端执行器位置 + 关节位置
        frame_traces = [
            go.Scatter3d(x=frame_data[:, 0], y=frame_data[:, 1], z=frame_data[:, 2])  # 末端执行器
        ]
        
        # 添加关节位置（如果有）
        if joint_data['left'] is not None:
            left_joints = joint_data['left']
            left_angles = joint_data['left_angles']
            # 创建悬停文本
            left_hover_text = []
            for j in range(len(left_joints)):
                if j == 0:
                    left_hover_text.append(f"左臂基座<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})")
                elif j == len(left_joints) - 1:
                    joint_value = left_angles[j-1] if left_angles is not None and j-1 < len(left_angles) else 0.0
                    left_hover_text.append(f"左臂末端执行器<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
                else:
                    joint_value = left_angles[j-1] if left_angles is not None and j-1 < len(left_angles) else 0.0
                    left_hover_text.append(f"左臂关节{j}<br>坐标: ({left_joints[j, 0]:.3f}, {left_joints[j, 1]:.3f}, {left_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
            
            frame_traces.append(go.Scatter3d(
                x=left_joints[:, 0], y=left_joints[:, 1], z=left_joints[:, 2],
                mode='markers+lines',
                text=left_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        if joint_data['right'] is not None:
            right_joints = joint_data['right']
            right_angles = joint_data['right_angles']
            # 创建悬停文本
            right_hover_text = []
            for j in range(len(right_joints)):
                if j == 0:
                    right_hover_text.append(f"右臂基座<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})")
                elif j == len(right_joints) - 1:
                    joint_value = right_angles[j-1] if right_angles is not None and j-1 < len(right_angles) else 0.0
                    right_hover_text.append(f"右臂末端执行器<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
                else:
                    joint_value = right_angles[j-1] if right_angles is not None and j-1 < len(right_angles) else 0.0
                    right_hover_text.append(f"右臂关节{j}<br>坐标: ({right_joints[j, 0]:.3f}, {right_joints[j, 1]:.3f}, {right_joints[j, 2]:.3f})<br>关节值: {joint_value:.3f}")
            
            frame_traces.append(go.Scatter3d(
                x=right_joints[:, 0], y=right_joints[:, 1], z=right_joints[:, 2],
                mode='markers+lines',
                text=right_hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # 添加指尖位置（如果有）
        # 左手指尖
        if gripper_data['left_thumb_pos'] is not None and gripper_data['left_index_pos'] is not None:
            left_thumb_pos = gripper_data['left_thumb_pos']
            left_index_pos = gripper_data['left_index_pos']
            
            # 左拇指
            frame_traces.append(go.Scatter3d(
                x=[left_thumb_pos[0]], y=[left_thumb_pos[1]], z=[left_thumb_pos[2]],
                mode='markers',
                marker=dict(size=8, color='orange', symbol='diamond'),
                text=[f"左拇指<br>坐标: ({left_thumb_pos[0]:.3f}, {left_thumb_pos[1]:.3f}, {left_thumb_pos[2]:.3f})<br>夹爪状态: {gripper_data['left_gripper_state']:.3f}"],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # 左食指
            frame_traces.append(go.Scatter3d(
                x=[left_index_pos[0]], y=[left_index_pos[1]], z=[left_index_pos[2]],
                mode='markers',
                marker=dict(size=8, color='purple', symbol='diamond'),
                text=[f"左食指<br>坐标: ({left_index_pos[0]:.3f}, {left_index_pos[1]:.3f}, {left_index_pos[2]:.3f})<br>夹爪状态: {gripper_data['left_gripper_state']:.3f}"],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # 右手指尖
        if gripper_data['right_thumb_pos'] is not None and gripper_data['right_index_pos'] is not None:
            right_thumb_pos = gripper_data['right_thumb_pos']
            right_index_pos = gripper_data['right_index_pos']
            
            # 右拇指
            frame_traces.append(go.Scatter3d(
                x=[right_thumb_pos[0]], y=[right_thumb_pos[1]], z=[right_thumb_pos[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                text=[f"右拇指<br>坐标: ({right_thumb_pos[0]:.3f}, {right_thumb_pos[1]:.3f}, {right_thumb_pos[2]:.3f})<br>夹爪状态: {gripper_data['right_gripper_state']:.3f}"],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # 右食指
            frame_traces.append(go.Scatter3d(
                x=[right_index_pos[0]], y=[right_index_pos[1]], z=[right_index_pos[2]],
                mode='markers',
                marker=dict(size=8, color='cyan', symbol='diamond'),
                text=[f"右食指<br>坐标: ({right_index_pos[0]:.3f}, {right_index_pos[1]:.3f}, {right_index_pos[2]:.3f})<br>夹爪状态: {gripper_data['right_gripper_state']:.3f}"],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig_frames.append(go.Frame(
            data=frame_traces,
            name=str(i)  # 帧的名称，用于按钮和滑块索引
        ))
    fig.frames = fig_frames
    
    # 在浏览器中显示图表
    fig.show()
    
    # 清理PyBullet资源
    pybullet_manager.close()


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Preview retargeted trajectory in the robot's world frame.")
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='Path to the original Egodex HDF5 file (e.g., /path/to/egodex/data/0.hdf5)')
    args = parser.parse_args()
    
    main(args.file)