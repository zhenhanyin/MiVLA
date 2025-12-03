#!/usr/bin/env python3
"""
RoboTwin到人手映射可视化脚本

功能：
1. 继承原有RoboTwin脚本的所有功能，显示双臂末端执行器位姿和坐标系
2. 将末端执行器位姿映射到人手拇指根关节
3. 根据拇指根关节位姿计算手腕位置
4. 基于手腕位姿计算5个手指指尖位置
5. 在3D可视化中显示末端执行器、手腕和手指指尖位置
"""

import plotly.graph_objects as go
import numpy as np
import h5py
import os
import argparse
import pybullet as p
from typing import Dict, List, Tuple
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
        # 手腕在拇指根关节的后方
        self.wrist_offset_from_thumb_base = np.array([-0.050, 0.030, 0.000])  # 向后50mm，向右30mm

        # 手腕到各手指根关节的位置 (在手腕局部坐标系中，XOY平面)
        # X轴: 手指朝向(正前方), Y轴: 手指排列方向(左负右正), Z轴: 0 (在XOY平面)
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

    def wrist_to_fingertips(self, wrist_pos, wrist_quat):
        """
        根据手腕位姿计算5个手指指尖位置

        Args:
            wrist_pos: 手腕位置 [x, y, z]
            wrist_quat: 手腕四元数 [qw, qx, qy, qz]

        Returns:
            Dict[str, np.ndarray]: 各手指指尖位置
        """
        # 将四元数转换为旋转矩阵
        rotation = R.from_quat([wrist_quat[1], wrist_quat[2],
                               wrist_quat[3], wrist_quat[0]])
        rot_matrix = rotation.as_matrix()

        fingertips = {}

        for finger_name, base_offset in self.finger_base_offsets.items():
            # 计算手指根关节位置 (世界坐标系)
            finger_base_pos = wrist_pos + rot_matrix @ base_offset

            # 手指方向：在手腕局部坐标系中，统一为正X轴方向
            finger_direction_local = np.array([1.0, 0.0, 0.0])  # 正X轴方向

            # 将手指方向从手腕局部坐标系转换到世界坐标系
            world_direction = rot_matrix @ finger_direction_local

            # 计算指尖位置 = 根关节位置 + 手指长度 * 手指方向
            fingertip_pos = finger_base_pos + world_direction * self.finger_lengths[finger_name]

            fingertips[finger_name] = fingertip_pos

        return fingertips

# =============================================================================
# RoboTwin数据集结构分析器
# =============================================================================

class RoboTwinDataLoader:
    """RoboTwin数据集加载器"""

    def __init__(self, hdf5_file_path: str):
        self.file_path = hdf5_file_path
        self.data = None
        self.num_frames = 0
        self._load_data()

    def _load_data(self):
        """加载HDF5数据"""
        try:
            self.data = h5py.File(self.file_path, 'r')
            self.num_frames = self.data['joint_action/left_arm'].shape[0]
            print(f"📊 成功加载RoboTwin数据: {self.num_frames} 帧")
            self._print_data_info()
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            raise

    def _print_data_info(self):
        """打印数据集信息"""
        print("🔧 数据集结构:")
        print(f"   总帧数: {self.num_frames}")
        print(f"   左臂关节: {self.data['joint_action/left_arm'].shape}")
        print(f"   右臂关节: {self.data['joint_action/right_arm'].shape}")
        print(f"   左臂末端位姿: {self.data['endpose/left_endpose'].shape}")
        print(f"   右臂末端位姿: {self.data['endpose/right_endpose'].shape}")
        print(f"   左夹爪状态: {self.data['endpose/left_gripper'].shape}")
        print(f"   右夹爪状态: {self.data['endpose/right_gripper'].shape}")

    def get_frame_data(self, frame_idx: int) -> Dict:
        """获取指定帧的所有数据"""
        if frame_idx >= self.num_frames:
            raise IndexError(f"帧索引 {frame_idx} 超出范围 (0-{self.num_frames-1})")

        return {
            'left_joints': self.data['joint_action/left_arm'][frame_idx],
            'right_joints': self.data['joint_action/right_arm'][frame_idx],
            'left_endpose': self.data['endpose/left_endpose'][frame_idx],
            'right_endpose': self.data['endpose/right_endpose'][frame_idx],
            'left_gripper': self.data['endpose/left_gripper'][frame_idx],
            'right_gripper': self.data['endpose/right_gripper'][frame_idx]
        }

    def get_all_data(self) -> Dict:
        """获取所有帧数据"""
        return {
            'left_joints': self.data['joint_action/left_arm'][:],
            'right_joints': self.data['joint_action/right_arm'][:],
            'left_endpose': self.data['endpose/left_endpose'][:],
            'right_endpose': self.data['endpose/right_endpose'][:],
            'left_gripper': self.data['endpose/left_gripper'][:],
            'right_gripper': self.data['endpose/right_gripper'][:]
        }

# =============================================================================
# 末端执行器数据处理器
# =============================================================================

def create_orientation_arrows(position: np.ndarray, quaternion: np.ndarray,
                             arrow_length: float = 0.05,
                             prefix: str = "") -> Dict[str, go.Scatter3d]:
    """
    创建表示末端执行器朝向的箭头（坐标系）

    Args:
        position: 3D位置 [x, y, z]
        quaternion: 四元数 [qw, qx, qy, qz]
        arrow_length: 箭头长度
        prefix: 坐标轴前缀名称

    Returns:
        包含X、Y、Z三个轴箭头的字典
    """
    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rot_matrix = rotation.as_matrix()

    # 定义三个坐标轴方向
    axes_directions = {
        'X': np.array([1, 0, 0]),  # 红色 - X轴
        'Y': np.array([0, 1, 0]),  # 绿色 - Y轴
        'Z': np.array([0, 0, 1])   # 蓝色 - Z轴
    }

    axes_colors = {
        'X': 'red',
        'Y': 'green',
        'Z': 'blue'
    }

    arrows = {}

    for axis_name, direction in axes_directions.items():
        # 旋转方向向量
        rotated_direction = rot_matrix @ direction
        end_point = position + rotated_direction * arrow_length

        # 创建箭头线
        arrows[f'{prefix}{axis_name}_axis'] = go.Scatter3d(
            x=[position[0], end_point[0]],
            y=[position[1], end_point[1]],
            z=[position[2], end_point[2]],
            mode='lines',
            name=f'{prefix}{axis_name}轴',
            line=dict(color=axes_colors[axis_name], width=4),
            hovertemplate=f'{prefix}{axis_name}轴<br>起点: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})<br>终点: ({end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f})<extra></extra>'
        )

    return arrows

# =============================================================================
# RoboTwin到人手映射可视化主逻辑
# =============================================================================

def visualize_robotwin_to_hand_trajectory(hdf5_file: str, use_arm: str = 'left'):
    """主函数：可视化RoboTwin轨迹和人手映射"""

    if not os.path.exists(hdf5_file):
        print(f"❌ 错误: 文件不存在 '{hdf5_file}'")
        return

    # 1. 加载数据
    print("🚀 开始可视化RoboTwin轨迹和人手映射...")
    loader = RoboTwinDataLoader(hdf5_file)
    all_data = loader.get_all_data()

    # 2. 初始化手部模型
    hand_model = HandModel()
    print("🤚 初始化手部模型...")

    # 3. 选择使用哪个臂的末端执行器作为拇指根关节
    endpose_key = f'{use_arm}_endpose'
    endposes = all_data[endpose_key]

    # 4. 计算所有帧的手部位置
    print("📊 计算手部位置...")
    wrist_positions = []
    fingertip_positions = {finger: [] for finger in hand_model.finger_lengths.keys()}

    for i in range(loader.num_frames):
        # 将末端执行器位姿作为拇指根关节位姿
        thumb_base_pos = endposes[i][:3]
        thumb_base_quat = endposes[i][3:7]

        # 计算手腕位姿
        wrist_pos, wrist_quat = hand_model.thumb_base_to_wrist(thumb_base_pos, thumb_base_quat)
        wrist_positions.append(wrist_pos)

        # 计算指尖位置
        fingertips = hand_model.wrist_to_fingertips(wrist_pos, wrist_quat)
        for finger_name, fingertip_pos in fingertips.items():
            fingertip_positions[finger_name].append(fingertip_pos)

    # 转换为numpy数组
    wrist_positions = np.array(wrist_positions)
    for finger_name in fingertip_positions:
        fingertip_positions[finger_name] = np.array(fingertip_positions[finger_name])

    # 5. 创建Plotly 3D图表
    print("📈 生成交互式3D图表...")
    fig = go.Figure()

    # 6. 绘制初始帧（第0帧）- 继承原脚本的所有功能
    initial_frame = 0

    # 原有的双臂末端执行器显示
    left_endpose_0 = all_data['left_endpose'][initial_frame]
    right_endpose_0 = all_data['right_endpose'][initial_frame]
    left_gripper_0 = all_data['left_gripper'][initial_frame]
    right_gripper_0 = all_data['right_gripper'][initial_frame]

    # 提取位置和四元数
    left_pos_0 = left_endpose_0[:3]
    left_quat_0 = left_endpose_0[3:7]
    right_pos_0 = right_endpose_0[:3]
    right_quat_0 = right_endpose_0[3:7]

    # 左末端执行器（拇指根关节）
    fig.add_trace(go.Scatter3d(
        x=[left_pos_0[0]],
        y=[left_pos_0[1]],
        z=[left_pos_0[2]],
        mode='markers',
        name='左末端执行器(拇指根关节)' if use_arm == 'left' else '左末端执行器',
        marker=dict(size=10, color='darkred', symbol='diamond'),
        text=[f"左末端执行器{'(拇指根关节)' if use_arm == 'left' else ''}<br>坐标: ({left_pos_0[0]:.3f}, {left_pos_0[1]:.3f}, {left_pos_0[2]:.3f})<br>四元数: ({left_quat_0[0]:.3f}, {left_quat_0[1]:.3f}, {left_quat_0[2]:.3f}, {left_quat_0[3]:.3f})<br>夹爪: {left_gripper_0:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 右末端执行器
    fig.add_trace(go.Scatter3d(
        x=[right_pos_0[0]],
        y=[right_pos_0[1]],
        z=[right_pos_0[2]],
        mode='markers',
        name='右末端执行器(拇指根关节)' if use_arm == 'right' else '右末端执行器',
        marker=dict(size=10, color='darkorange', symbol='diamond'),
        text=[f"右末端执行器{'(拇指根关节)' if use_arm == 'right' else ''}<br>坐标: ({right_pos_0[0]:.3f}, {right_pos_0[1]:.3f}, {right_pos_0[2]:.3f})<br>四元数: ({right_quat_0[0]:.3f}, {right_quat_0[1]:.3f}, {right_quat_0[2]:.3f}, {right_quat_0[3]:.3f})<br>夹爪: {right_gripper_0:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 添加末端执行器朝向箭头（坐标系）
    print("🧭 添加末端执行器朝向指示...")
    left_orientation_arrows = create_orientation_arrows(left_pos_0, left_quat_0, arrow_length=0.08, prefix="左_")
    right_orientation_arrows = create_orientation_arrows(right_pos_0, right_quat_0, arrow_length=0.08, prefix="右_")

    # 添加朝向箭头到图表
    for arrow_trace in left_orientation_arrows.values():
        fig.add_trace(arrow_trace)
    for arrow_trace in right_orientation_arrows.values():
        fig.add_trace(arrow_trace)

    # 添加人手模型的可视化
    print("🖐️ 添加人手模型...")

    # 计算手指根关节位置（用于可视化）
    finger_base_positions = {finger: [] for finger in hand_model.finger_lengths.keys()}

    # 使用第0帧的手腕位姿计算根关节位置
    wrist_pos_0 = wrist_positions[initial_frame]
    endpose_0 = endposes[initial_frame]
    thumb_base_pos_0 = endpose_0[:3]
    thumb_base_quat_0 = endpose_0[3:7]

    # 计算手腕位姿
    wrist_pos, wrist_quat = hand_model.thumb_base_to_wrist(thumb_base_pos_0, thumb_base_quat_0)

    # 将四元数转换为旋转矩阵
    rotation = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])
    rot_matrix = rotation.as_matrix()

    # 计算各手指根关节位置
    for finger_name in hand_model.finger_lengths.keys():
        base_offset = hand_model.finger_base_offsets[finger_name]
        finger_base_pos = wrist_pos + rot_matrix @ base_offset
        finger_base_positions[finger_name] = finger_base_pos

    # 手腕位置
    fig.add_trace(go.Scatter3d(
        x=[wrist_positions[initial_frame][0]],
        y=[wrist_positions[initial_frame][1]],
        z=[wrist_positions[initial_frame][2]],
        mode='markers',
        name='手腕',
        marker=dict(size=12, color='black', symbol='circle'),
        text=[f"手腕<br>坐标: ({wrist_positions[initial_frame][0]:.3f}, {wrist_positions[initial_frame][1]:.3f}, {wrist_positions[initial_frame][2]:.3f})"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 手指颜色和中文名称
    finger_colors = {
        'thumb': 'magenta',
        'index': 'cyan',
        'middle': 'lime',
        'ring': 'yellow',
        'pinky': 'pink'
    }

    finger_names_chinese = {
        'thumb': '拇指',
        'index': '食指',
        'middle': '中指',
        'ring': '无名指',
        'pinky': '小指'
    }

    # 绘制手指根关节
    for finger_name, base_pos in finger_base_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[base_pos[0]],
            y=[base_pos[1]],
            z=[base_pos[2]],
            mode='markers',
            name=finger_names_chinese[finger_name] + '根关节',
            marker=dict(size=6, color=finger_colors[finger_name], symbol='square'),
            text=[f"{finger_names_chinese[finger_name]}根关节<br>坐标: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

    # 绘制手指指尖
    for finger_name, positions in fingertip_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[positions[initial_frame][0]],
            y=[positions[initial_frame][1]],
            z=[positions[initial_frame][2]],
            mode='markers',
            name=finger_names_chinese[finger_name] + '指尖',
            marker=dict(size=8, color=finger_colors[finger_name], symbol='diamond'),
            text=[f"{finger_names_chinese[finger_name]}指尖<br>坐标: ({positions[initial_frame][0]:.3f}, {positions[initial_frame][1]:.3f}, {positions[initial_frame][2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

    # 绘制连接线：手腕 -> 手指根关节 -> 指尖
    print("🔗 绘制手指连接线...")
    for finger_name in hand_model.finger_lengths.keys():
        base_pos = finger_base_positions[finger_name]
        tip_pos = fingertip_positions[finger_name][initial_frame]

        # 手腕到根关节的线
        fig.add_trace(go.Scatter3d(
            x=[wrist_positions[initial_frame][0], base_pos[0]],
            y=[wrist_positions[initial_frame][1], base_pos[1]],
            z=[wrist_positions[initial_frame][2], base_pos[2]],
            mode='lines',
            name=f'手腕-{finger_names_chinese[finger_name]}根关节',
            line=dict(color=finger_colors[finger_name], width=3, dash='solid'),
            hoverinfo='skip'
        ))

        # 根关节到指尖的线
        fig.add_trace(go.Scatter3d(
            x=[base_pos[0], tip_pos[0]],
            y=[base_pos[1], tip_pos[1]],
            z=[base_pos[2], tip_pos[2]],
            mode='lines',
            name=f'{finger_names_chinese[finger_name]}根关节-指尖',
            line=dict(color=finger_colors[finger_name], width=4, dash='solid'),
            hoverinfo='skip'
        ))

    # 7. 计算坐标轴范围
    all_endposes = np.concatenate([all_data['left_endpose'][:, :3],
                                   all_data['right_endpose'][:, :3],
                                   wrist_positions] + list(fingertip_positions.values()))
    min_coords, max_coords = np.min(all_endposes, axis=0), np.max(all_endposes, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.8

    # 8. 设置图表布局
    fig.update_layout(
        title=f"RoboTwin到人手映射可视化<br><sub>数据文件: {os.path.basename(hdf5_file)} | 使用{('左' if use_arm == 'left' else '右')}臂映射到拇指根关节 | 总帧数: {loader.num_frames}</sub>",
        scene=dict(
            xaxis_title='X (右)',
            yaxis_title='Y (前)',
            zaxis_title='Z (上)',
            xaxis=dict(range=[center[0] - max_range, center[0] + max_range]),
            yaxis=dict(range=[center[1] - max_range, center[1] + max_range]),
            zaxis=dict(range=[max(0, center[2] - max_range), center[2] + max_range]),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=center[0] - 1.5, y=center[1] - 1.5, z=center[2] + 1.0)
            )
        ),
        legend=dict(x=0.01, y=0.99),
        # 添加播放控制按钮
        updatemenus=[{
            'type': 'buttons', 'direction': 'left', 'showactive': True,
            'buttons': [
                {'label': '播放 (30 FPS)', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 33.33, 'redraw': True},
                                 'fromcurrent': True, 'mode': 'immediate',
                                 'transition': {'duration': 0}}]},
                {'label': '暂停', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                 'mode': 'immediate', 'transition': {'duration': 0}}]}
            ]
        }],
        # 添加时间轴滑块
        sliders=[{
            'currentvalue': {'prefix': '帧: '}, 'pad': {'t': 50},
            'steps': [{'label': str(i), 'method': 'animate',
                      'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True},
                                        'mode': 'immediate', 'transition': {'duration': 0}}]}
                     for i in range(loader.num_frames)]
        }]
    )

    # 9. 创建动画帧
    print("🎬 创建动画帧...")
    frames = []

    for i in range(loader.num_frames):
        frame_traces = []

        # 末端执行器完整位姿数据（位置 + 四元数）
        left_endpose_full = all_data['left_endpose'][i]
        right_endpose_full = all_data['right_endpose'][i]
        left_gripper = all_data['left_gripper'][i]
        right_gripper = all_data['right_gripper'][i]

        # 提取位置和四元数
        left_pos = left_endpose_full[:3]
        left_quat = left_endpose_full[3:7]
        right_pos = right_endpose_full[:3]
        right_quat = right_endpose_full[3:7]

        # 末端执行器位置点
        frame_traces.append(go.Scatter3d(
            x=[left_pos[0]],
            y=[left_pos[1]],
            z=[left_pos[2]],
            mode='markers',
            marker=dict(size=10, color='darkred', symbol='diamond'),
            text=[f"左末端执行器{'(拇指根关节)' if use_arm == 'left' else ''}<br>坐标: ({left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f})<br>四元数: ({left_quat[0]:.3f}, {left_quat[1]:.3f}, {left_quat[2]:.3f}, {left_quat[3]:.3f})<br>夹爪: {left_gripper:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))

        frame_traces.append(go.Scatter3d(
            x=[right_pos[0]],
            y=[right_pos[1]],
            z=[right_pos[2]],
            mode='markers',
            marker=dict(size=10, color='darkorange', symbol='diamond'),
            text=[f"右末端执行器{'(拇指根关节)' if use_arm == 'right' else ''}<br>坐标: ({right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f})<br>四元数: ({right_quat[0]:.3f}, {right_quat[1]:.3f}, {right_quat[2]:.3f}, {right_quat[3]:.3f})<br>夹爪: {right_gripper:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))

        # 添加朝向箭头
        left_orientation_arrows = create_orientation_arrows(left_pos, left_quat, arrow_length=0.08, prefix="左_")
        right_orientation_arrows = create_orientation_arrows(right_pos, right_quat, arrow_length=0.08, prefix="右_")

        for arrow_trace in left_orientation_arrows.values():
            frame_traces.append(arrow_trace)
        for arrow_trace in right_orientation_arrows.values():
            frame_traces.append(arrow_trace)

        # 手腕
        frame_traces.append(go.Scatter3d(
            x=[wrist_positions[i][0]],
            y=[wrist_positions[i][1]],
            z=[wrist_positions[i][2]],
            mode='markers',
            marker=dict(size=12, color='black', symbol='circle'),
            text=[f"手腕<br>坐标: ({wrist_positions[i][0]:.3f}, {wrist_positions[i][1]:.3f}, {wrist_positions[i][2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

        # 计算当前帧的手指根关节位置
        current_thumb_base_pos = endposes[i][:3]
        current_thumb_base_quat = endposes[i][3:7]
        current_wrist_pos, current_wrist_quat = hand_model.thumb_base_to_wrist(current_thumb_base_pos, current_thumb_base_quat)

        # 将四元数转换为旋转矩阵
        current_rotation = R.from_quat([current_wrist_quat[1], current_wrist_quat[2], current_wrist_quat[3], current_wrist_quat[0]])
        current_rot_matrix = current_rotation.as_matrix()

        # 计算当前帧各手指根关节位置
        current_finger_base_positions = {}
        for finger_name in hand_model.finger_lengths.keys():
            base_offset = hand_model.finger_base_offsets[finger_name]
            finger_base_pos = current_wrist_pos + current_rot_matrix @ base_offset
            current_finger_base_positions[finger_name] = finger_base_pos

        # 各手指根关节
        for finger_name, base_pos in current_finger_base_positions.items():
            frame_traces.append(go.Scatter3d(
                x=[base_pos[0]],
                y=[base_pos[1]],
                z=[base_pos[2]],
                mode='markers',
                marker=dict(size=6, color=finger_colors[finger_name], symbol='square'),
                text=[f"{finger_names_chinese[finger_name]}根关节<br>坐标: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})"],
                hovertemplate='%{text}<extra></extra>'
            ))

        # 各手指指尖
        for finger_name, positions in fingertip_positions.items():
            frame_traces.append(go.Scatter3d(
                x=[positions[i][0]],
                y=[positions[i][1]],
                z=[positions[i][2]],
                mode='markers',
                marker=dict(size=8, color=finger_colors[finger_name], symbol='diamond'),
                text=[f"{finger_names_chinese[finger_name]}指尖<br>坐标: ({positions[i][0]:.3f}, {positions[i][1]:.3f}, {positions[i][2]:.3f})"],
                hovertemplate='%{text}<extra></extra>'
            ))

        # 绘制连接线：手腕 -> 手指根关节 -> 指尖
        for finger_name in hand_model.finger_lengths.keys():
            base_pos = current_finger_base_positions[finger_name]
            tip_pos = fingertip_positions[finger_name][i]

            # 手腕到根关节的线
            frame_traces.append(go.Scatter3d(
                x=[current_wrist_pos[0], base_pos[0]],
                y=[current_wrist_pos[1], base_pos[1]],
                z=[current_wrist_pos[2], base_pos[2]],
                mode='lines',
                line=dict(color=finger_colors[finger_name], width=3, dash='solid'),
                hoverinfo='skip'
            ))

            # 根关节到指尖的线
            frame_traces.append(go.Scatter3d(
                x=[base_pos[0], tip_pos[0]],
                y=[base_pos[1], tip_pos[1]],
                z=[base_pos[2], tip_pos[2]],
                mode='lines',
                line=dict(color=finger_colors[finger_name], width=4, dash='solid'),
                hoverinfo='skip'
            ))

        frames.append(go.Frame(data=frame_traces, name=str(i)))

    fig.frames = frames

    # 10. 显示图表
    print("✅ 可视化完成！在浏览器中打开...")
    print("💡 提示:")
    print("   - 红色/橙色菱形: 末端执行器位置（作为拇指根关节）")
    print("   - 红绿蓝箭头: 末端执行器坐标系")
    print("   - 黑色圆点: 手腕位置")
    print("   - 彩色方块: 各手指根关节位置")
    print("   - 彩色菱形: 各手指指尖位置")
    print("   - 彩色线条: 手腕-根关节-指尖连接线")
    print("   - 使用播放按钮观看动画")
    print("   - 拖动时间轴滑块选择特定帧")
    print("   - 鼠标悬停查看详细坐标信息")

    fig.show()

    # 11. 关闭数据文件
    loader.data.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboTwin到人手映射可视化工具")
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='RoboTwin HDF5文件路径 (例如: /path/to/episode0.hdf5)')
    parser.add_argument('--arm', type=str, default='left', choices=['left', 'right'],
                        help='使用哪个臂的末端执行器数据作为拇指根关节 (默认: left)')

    args = parser.parse_args()

    visualize_robotwin_to_hand_trajectory(args.file, args.arm)