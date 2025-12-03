#!/usr/bin/env python3
"""
基于14D预处理器数据的机械臂直接控制可视化脚本

功能：
1. 读取预处理后的HDF5文件中的actions_62d数据
2. 提取后14维机械臂控制数据（左臂6关节+左夹爪+右臂6关节+右夹爪）
3. 直接使用这些数据控制机械臂，逐帧渲染成图片

14D数据格式：
- 维度 0-5:   左臂6个关节角度
- 维度 6:     左臂夹爪开合状态 (0=闭合, 1=张开)
- 维度 7-12:  右臂6个关节角度
- 维度 13:    右臂夹爪开合状态 (0=闭合, 1=张开)
"""

import pybullet as p
import pybullet_data
import numpy as np
import h5py
import os
import cv2
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# ==============================================================================
# 机器人配置和PyBullet管理器
# ==============================================================================

class RobotConfig:
    """机器人配置类"""
    URDF_PATH = "/home/yinzhenhan/Human_Pre/dataset/robot/aloha-agilex/urdf/arx5_description_isaac.urdf"
    LEFT_ARM_JOINTS = [18, 19, 20, 21, 22, 23]
    RIGHT_ARM_JOINTS = [28, 29, 30, 31, 32, 33]
    LEFT_GRIPPER_JOINTS = [25, 26]  # fl_joint7, fl_joint8
    RIGHT_GRIPPER_JOINTS = [35, 36]  # fr_joint7, fr_joint8
    LEFT_END_EFFECTOR_LINK = "fl_link6"
    RIGHT_END_EFFECTOR_LINK = "fr_link6"

class Direct14DRobotController:
    """基于14D数据的直接机械臂控制器"""
    
    def __init__(self, headless=True):
        """
        初始化机械臂控制器
        """
        print("🤖 初始化14D直接控制机械臂环境...")
        
        # 设置PyBullet环境
        if headless:
            self.physics_client = p.connect(p.DIRECT)  # 无头模式
        else:
            self.physics_client = p.connect(p.GUI)
            print("使用 GUI 模式。")
        
        # 设置场景
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载机器人
        self.robot_id = p.loadURDF(RobotConfig.URDF_PATH, useFixedBase=True)
        
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
        self.left_gripper_joints = RobotConfig.LEFT_GRIPPER_JOINTS
        self.right_gripper_joints = RobotConfig.RIGHT_GRIPPER_JOINTS
        
        # 夹爪参数
        self.max_gripper_distance = 0.04765  # 最大张开距离4.765厘米
        
        print("✅ 14D直接控制机械臂初始化完成")
        print("🔧 将直接使用预处理的14D数据控制机械臂")
    
    def load_14d_data(self, hdf5_file_path: str) -> Tuple[np.ndarray, int]:
        """
        从预处理文件中加载14D数据
        
        参数:
            hdf5_file_path: 预处理后的HDF5文件路径
            
        返回:
            actions_14d: 14D数据数组 (num_frames, 14)
            num_frames: 帧数
        """
        try:
            with h5py.File(hdf5_file_path, 'r') as hdf5_file:
                if 'actions_62d' not in hdf5_file:
                    raise ValueError("预处理文件中缺少 'actions_62d' 数据集")
                
                actions_62d = hdf5_file['actions_62d'][:]
                num_frames = actions_62d.shape[0]
                
                # 提取后14维数据
                actions_14d = actions_62d[:, 48:62]  # 后14维是机械臂控制数据
                
                print(f"📊 成功加载14D数据:")
                print(f"   总帧数: {num_frames}")
                print(f"   数据形状: {actions_14d.shape}")
                print(f"   数据类型: {actions_14d.dtype}")
                
                # 显示数据统计信息
                print(f"📊 14D数据统计:")
                print(f"   左臂关节范围: [{np.min(actions_14d[:, 0:6]):.3f}, {np.max(actions_14d[:, 0:6]):.3f}]")
                print(f"   左夹爪范围: [{np.min(actions_14d[:, 6]):.3f}, {np.max(actions_14d[:, 6]):.3f}]")
                print(f"   右臂关节范围: [{np.min(actions_14d[:, 7:13]):.3f}, {np.max(actions_14d[:, 7:13]):.3f}]")
                print(f"   右夹爪范围: [{np.min(actions_14d[:, 13]):.3f}, {np.max(actions_14d[:, 13]):.3f}]")
                
                return actions_14d, num_frames
                
        except Exception as e:
            print(f"❌ 加载14D数据失败: {e}")
            return None, 0
    
    def set_robot_from_14d(self, actions_14d: np.ndarray, frame_idx: int) -> Dict:
        """
        使用14D数据设置机械臂姿态
        
        参数:
            actions_14d: 14D数据数组
            frame_idx: 当前帧索引
            
        返回:
            包含设置信息的字典
        """
        try:
            # 获取当前帧的14D数据
            frame_data = actions_14d[frame_idx]
            
            # 分解14D数据
            left_joint_angles = frame_data[0:6]    # 左臂6关节
            left_gripper_state = frame_data[6]    # 左夹爪状态
            right_joint_angles = frame_data[7:13]  # 右臂6关节
            right_gripper_state = frame_data[13]   # 右夹爪状态
            
            # 设置左臂关节角度
            for i, joint_idx in enumerate(self.left_arm_joints):
                p.resetJointState(self.robot_id, joint_idx, left_joint_angles[i])
            
            # 设置右臂关节角度
            for i, joint_idx in enumerate(self.right_arm_joints):
                p.resetJointState(self.robot_id, joint_idx, right_joint_angles[i])
            
            # 设置左臂夹爪
            left_gripper_distance = left_gripper_state * self.max_gripper_distance
            p.resetJointState(self.robot_id, self.left_gripper_joints[0], left_gripper_distance)
            p.resetJointState(self.robot_id, self.left_gripper_joints[1], left_gripper_distance)
            
            # 设置右臂夹爪
            right_gripper_distance = right_gripper_state * self.max_gripper_distance
            p.resetJointState(self.robot_id, self.right_gripper_joints[0], right_gripper_distance)
            p.resetJointState(self.robot_id, self.right_gripper_joints[1], right_gripper_distance)
            
            # 获取末端执行器实际位置
            left_end_effector_state = p.getLinkState(self.robot_id, 23)  # fl_link6
            right_end_effector_state = p.getLinkState(self.robot_id, 33)  # fr_link6
            
            left_end_pos = np.array(left_end_effector_state[0])
            right_end_pos = np.array(right_end_effector_state[0])
            
            # 调试信息（仅在前几帧显示）
            if frame_idx < 5:
                print(f"🔧 帧{frame_idx} 14D控制数据:")
                print(f"   左臂关节角度: {[f'{a:.3f}' for a in left_joint_angles]}")
                print(f"   左夹爪状态: {left_gripper_state:.3f} (距离: {left_gripper_distance*1000:.1f}mm)")
                print(f"   右臂关节角度: {[f'{a:.3f}' for a in right_joint_angles]}")
                print(f"   右夹爪状态: {right_gripper_state:.3f} (距离: {right_gripper_distance*1000:.1f}mm)")
                print(f"   左臂末端位置: ({left_end_pos[0]:.3f}, {left_end_pos[1]:.3f}, {left_end_pos[2]:.3f})")
                print(f"   右臂末端位置: ({right_end_pos[0]:.3f}, {right_end_pos[1]:.3f}, {right_end_pos[2]:.3f})")
            
            return {
                'left_joint_angles': left_joint_angles,
                'right_joint_angles': right_joint_angles,
                'left_gripper_state': left_gripper_state,
                'right_gripper_state': right_gripper_state,
                'left_end_pos': left_end_pos,
                'right_end_pos': right_end_pos,
                'left_gripper_distance': left_gripper_distance,
                'right_gripper_distance': right_gripper_distance
            }
            
        except Exception as e:
            print(f"设置机械臂姿态失败: {e}")
            return None
    
    def render_to_frames(self, hdf5_file_path: str, output_dir: str, fps=30, max_frames=None):
        """
        渲染14D数据控制的机械臂运动为图像帧序列
        
        参数:
            hdf5_file_path: 预处理后的HDF5文件路径
            output_dir: 输出目录
            fps: 帧率
            max_frames: 最大渲染帧数
        """
        
        # 检查文件是否存在
        if not os.path.exists(hdf5_file_path):
            print(f"❌ HDF5文件不存在: {hdf5_file_path}")
            return False
        
        # 加载14D数据
        actions_14d, num_frames = self.load_14d_data(hdf5_file_path)
        if actions_14d is None:
            return False
        
        if max_frames:
            num_frames = min(num_frames, max_frames)
        
        print(f"🎬 将渲染 {num_frames} 帧")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图像参数
        width, height = 800, 600
        
        print(f"🎬 开始渲染14D直接控制的机械臂运动...")
        print(f"📁 输出目录: {output_dir}")
        
        successful_frames = 0
        
        for frame_idx in tqdm(range(num_frames), desc="渲染进度"):
            try:
                # 使用14D数据设置机械臂姿态
                control_info = self.set_robot_from_14d(actions_14d, frame_idx)
                
                if control_info is None:
                    continue
                
                # 设置相机参数（与direct_transform_visualizer.py保持一致）
                camera_distance = 1.6
                camera_yaw = 180
                camera_pitch = -45
                
                # 计算相机位置
                camera_x_offset = camera_distance * np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch))
                camera_y_offset = camera_distance * np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch))
                camera_z_offset = camera_distance * np.sin(np.radians(camera_pitch))
                
                # 相机目标点
                camera_target = [0.4, 0, 0.9]
                camera_pos = [
                    camera_target[0] + camera_x_offset,
                    camera_target[1] + camera_y_offset,
                    camera_target[2] - camera_z_offset
                ]
                
                up_vector = [0, 0, 1]
                
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
                        print(f"\\n📹 帧 {frame_idx} - 14D直接控制:")
                        print(f"  左臂关节角度: {[f'{a:.3f}' for a in control_info['left_joint_angles']]}")
                        print(f"  右臂关节角度: {[f'{a:.3f}' for a in control_info['right_joint_angles']]}")
                        print(f"  左夹爪状态: {control_info['left_gripper_state']:.3f} (距离: {control_info['left_gripper_distance']*1000:.1f}mm)")
                        print(f"  右夹爪状态: {control_info['right_gripper_state']:.3f} (距离: {control_info['right_gripper_distance']*1000:.1f}mm)")
                        print(f"  左臂末端位置: ({control_info['left_end_pos'][0]:.3f}, {control_info['left_end_pos'][1]:.3f}, {control_info['left_end_pos'][2]:.3f})")
                        print(f"  右臂末端位置: ({control_info['right_end_pos'][0]:.3f}, {control_info['right_end_pos'][1]:.3f}, {control_info['right_end_pos'][2]:.3f})")
                        
            except Exception as e:
                print(f"渲染帧 {frame_idx} 时出错: {e}")
                continue
        
        print(f"🎉 渲染完成！成功渲染 {successful_frames}/{num_frames} 帧")
        print(f"📁 帧序列保存至: {output_dir}")
        
        return successful_frames > 0
    
    def close(self):
        """关闭PyBullet连接"""
        print("🔌 关闭PyBullet环境")
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    print("🚀 启动14D直接控制机械臂可视化器")
    
    # 用户配置
    hdf5_file = "/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain/part2/basic_fold/0.hdf5"
    output_dir = "./visual_test/14d_direct_control_frames—lid-jump-small-o"
    fps = 30
    max_frames = None  # 不限制渲染帧数，渲染全部帧
    use_gui = False  # 使用无头模式
    
    # 检查文件
    if not os.path.exists(hdf5_file):
        print(f"❌ HDF5文件不存在: {hdf5_file}")
        return
    
    print(f"📂 预处理HDF5文件: {hdf5_file}")
    print(f"🎯 输出目录: {output_dir}")
    print(f"🎬 渲染帧数: {max_frames}")
    
    # 创建14D直接控制器
    controller = Direct14DRobotController(headless=not use_gui)
    
    if controller.robot_id is None:
        print("❌ 机器人加载失败，退出")
        return
    
    # 渲染帧序列
    success = controller.render_to_frames(hdf5_file, output_dir, fps=fps, max_frames=max_frames)
    
    if success:
        print("✅ 14D直接控制机械臂可视化成功！")
        print("🔧 技术特点：")
        print("   - 直接使用预处理后的14D机械臂控制数据")
        print("   - 14D格式: [左臂6关节+左夹爪+右臂6关节+右夹爪]")
        print("   - 无需IK求解，直接关节控制")
        print("   - 使用PyBullet真实URDF模型")
        print("   - 夹爪基于URDF实际参数: [0, 0.04765]米")
        print("   - 实时显示关节角度和夹爪状态")
        print("   - 🆕 与IK求解结果对比，验证预处理数据质量")
    else:
        print("❌ 可视化失败")
    
    # 关闭环境
    controller.close()

if __name__ == "__main__":
    main()