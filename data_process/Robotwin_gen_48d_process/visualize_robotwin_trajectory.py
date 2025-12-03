#!/usr/bin/env python3
"""
RoboTwin数据集可视化脚本

功能：
1. 读取RoboTwin HDF5文件，包含关节角度和末端执行器位姿数据
2. 在交互式3D图表中动画展示机器人双臂的运动轨迹
3. 显示末端执行器位置、朝向（四元数）和夹爪状态
4. 使用坐标系箭头可视化末端执行器的姿态朝向
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
# RoboTwin可视化主逻辑
# =============================================================================

def visualize_robotwin_trajectory(hdf5_file: str):
    """主函数：可视化RoboTwin轨迹"""
    
    if not os.path.exists(hdf5_file):
        print(f"❌ 错误: 文件不存在 '{hdf5_file}'")
        return
    
    # 1. 加载数据
    print("🚀 开始可视化RoboTwin轨迹...")
    loader = RoboTwinDataLoader(hdf5_file)
    all_data = loader.get_all_data()
    
    # 2. 直接使用末端执行器数据，无需计算关节位置
    print("📊 准备末端执行器数据...")
    print(f"   数据准备完成: {loader.num_frames} 帧")
    
    # 4. 创建Plotly 3D图表
    print("📈 生成交互式3D图表...")
    fig = go.Figure()
    
    # 5. 绘制初始帧（第0帧）末端执行器
    initial_frame = 0

    # 末端执行器位姿数据（位置 + 四元数）
    left_endpose_0 = all_data['left_endpose'][initial_frame]  # [x, y, z, qw, qx, qy, qz]
    right_endpose_0 = all_data['right_endpose'][initial_frame]
    left_gripper_0 = all_data['left_gripper'][initial_frame]
    right_gripper_0 = all_data['right_gripper'][initial_frame]

    # 提取位置和四元数
    left_pos_0 = left_endpose_0[:3]
    left_quat_0 = left_endpose_0[3:7]
    right_pos_0 = right_endpose_0[:3]
    right_quat_0 = right_endpose_0[3:7]

    # 左末端执行器
    fig.add_trace(go.Scatter3d(
        x=[left_pos_0[0]],
        y=[left_pos_0[1]],
        z=[left_pos_0[2]],
        mode='markers',
        name='左末端执行器',
        marker=dict(size=10, color='darkred', symbol='diamond'),
        text=[f"左末端执行器<br>坐标: ({left_pos_0[0]:.3f}, {left_pos_0[1]:.3f}, {left_pos_0[2]:.3f})<br>四元数: ({left_quat_0[0]:.3f}, {left_quat_0[1]:.3f}, {left_quat_0[2]:.3f}, {left_quat_0[3]:.3f})<br>夹爪: {left_gripper_0:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 右末端执行器
    fig.add_trace(go.Scatter3d(
        x=[right_pos_0[0]],
        y=[right_pos_0[1]],
        z=[right_pos_0[2]],
        mode='markers',
        name='右末端执行器',
        marker=dict(size=10, color='darkorange', symbol='diamond'),
        text=[f"右末端执行器<br>坐标: ({right_pos_0[0]:.3f}, {right_pos_0[1]:.3f}, {right_pos_0[2]:.3f})<br>四元数: ({right_quat_0[0]:.3f}, {right_quat_0[1]:.3f}, {right_quat_0[2]:.3f}, {right_quat_0[3]:.3f})<br>夹爪: {right_gripper_0:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 添加朝向箭头（坐标系）
    print("🧭 添加末端执行器朝向指示...")
    left_orientation_arrows = create_orientation_arrows(left_pos_0, left_quat_0, arrow_length=0.08, prefix="左_")
    right_orientation_arrows = create_orientation_arrows(right_pos_0, right_quat_0, arrow_length=0.08, prefix="右_")

    # 添加朝向箭头到图表
    for arrow_trace in left_orientation_arrows.values():
        fig.add_trace(arrow_trace)
    for arrow_trace in right_orientation_arrows.values():
        fig.add_trace(arrow_trace)
    
    # 6. 计算坐标轴范围
    all_endposes = np.concatenate([all_data['left_endpose'][:, :3], 
                                   all_data['right_endpose'][:, :3]])
    min_coords, max_coords = np.min(all_endposes, axis=0), np.max(all_endposes, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.8
    
    # 7. 设置图表布局
    fig.update_layout(
        title=f"RoboTwin机器人双臂轨迹可视化<br><sub>数据文件: {os.path.basename(hdf5_file)} | 总帧数: {loader.num_frames}</sub>",
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
    
    # 8. 创建动画帧
    print("🎬 创建动画帧...")
    frames = []
    
    for i in range(loader.num_frames):
        frame_traces = []

        # 末端执行器完整位姿数据（位置 + 四元数）
        left_endpose_full = all_data['left_endpose'][i]  # [x, y, z, qw, qx, qy, qz]
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
            text=[f"左末端执行器<br>坐标: ({left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f})<br>四元数: ({left_quat[0]:.3f}, {left_quat[1]:.3f}, {left_quat[2]:.3f}, {left_quat[3]:.3f})<br>夹爪: {left_gripper:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))

        frame_traces.append(go.Scatter3d(
            x=[right_pos[0]],
            y=[right_pos[1]],
            z=[right_pos[2]],
            mode='markers',
            marker=dict(size=10, color='darkorange', symbol='diamond'),
            text=[f"右末端执行器<br>坐标: ({right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f})<br>四元数: ({right_quat[0]:.3f}, {right_quat[1]:.3f}, {right_quat[2]:.3f}, {right_quat[3]:.3f})<br>夹爪: {right_gripper:.3f}"],
            hovertemplate='%{text}<extra></extra>'
        ))

        # 添加朝向箭头
        left_orientation_arrows = create_orientation_arrows(left_pos, left_quat, arrow_length=0.08, prefix="左_")
        right_orientation_arrows = create_orientation_arrows(right_pos, right_quat, arrow_length=0.08, prefix="右_")

        for arrow_trace in left_orientation_arrows.values():
            frame_traces.append(arrow_trace)
        for arrow_trace in right_orientation_arrows.values():
            frame_traces.append(arrow_trace)

        frames.append(go.Frame(data=frame_traces, name=str(i)))
    
    fig.frames = frames
    
    # 9. 显示图表
    print("✅ 可视化完成！在浏览器中打开...")
    print("💡 提示:")
    print("   - 使用播放按钮观看动画")
    print("   - 拖动时间轴滑块选择特定帧")
    print("   - 鼠标悬停查看详细坐标信息")
    print("   - 左键拖动旋转视角，右键拖动缩放")
    
    fig.show()
    
    # 10. 关闭数据文件
    loader.data.close()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="RoboTwin机器人轨迹可视化工具")
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='RoboTwin HDF5文件路径 (例如: /path/to/episode0.hdf5)')
    
    args = parser.parse_args()
    
    visualize_robotwin_trajectory(args.file)