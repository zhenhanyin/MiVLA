#!/usr/bin/env python3
"""
Egodex转换结果可视化脚本

功能：
1. 读取转换后的Egodex HDF5文件，其中包含62维动作数据（前48维为转换后的人手动作）
2. 提取左右手腕位置和五个指尖位置
3. 在交互式3D图表中展示双手运动轨迹
4. 显示手腕位置、五个指尖位置，并用连接线表示手掌结构
5. 使用动画展示时间序列变化
"""

import plotly.graph_objects as go
import numpy as np
import h5py
import os
import argparse
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 手指名称和颜色定义
# =============================================================================

# 手指名称定义（按照Egodex数据顺序）
FINGER_NAMES = ['拇指', '食指', '中指', '无名指', '小指']
HAND_NAMES = ['左手', '右手']

# 手指名称和颜色定义（中英文映射）
FINGER_NAME_MAPPING = {
    '拇指': 'thumb',
    '食指': 'index',
    '中指': 'middle',
    '无名指': 'ring',
    '小指': 'pinky'
}

# 颜色定义（使用英文键）
HAND_COLORS = {
    '左手': {
        'wrist': 'darkred',
        'thumb': 'red',
        'index': 'orange',
        'middle': 'yellow',
        'ring': 'green',
        'pinky': 'blue'
    },
    '右手': {
        'wrist': 'darkorange',
        'thumb': 'coral',
        'index': 'gold',
        'middle': 'lightgreen',
        'ring': 'cyan',
        'pinky': 'purple'
    }
}

# =============================================================================
# Egodex转换结果数据加载器
# =============================================================================

class EgodexDataLoader:
    """Egodex转换结果数据加载器"""

    def __init__(self, hdf5_file_path: str):
        self.file_path = hdf5_file_path
        self.data = None
        self.num_frames = 0
        self._load_data()

    def _load_data(self):
        """加载HDF5数据"""
        try:
            self.data = h5py.File(self.file_path, 'r')
            if 'actions_62d' not in self.data:
                raise ValueError("文件中未找到 'actions_62d' 数据集")

            self.num_frames = self.data['actions_62d'].shape[0]
            print(f"📊 成功加载Egodex转换数据: {self.num_frames} 帧")

            # 检查数据集属性
            actions_62d_attrs = self.data['actions_62d'].attrs
            print(f"🔧 数据集信息:")
            print(f"   坐标系统: {actions_62d_attrs.get('coordinate_system', 'Unknown')}")
            print(f"   描述: {actions_62d_attrs.get('description', 'No description')}")

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            raise

    def get_hand_data(self, frame_idx: int) -> Dict:
        """获取指定帧的双手数据"""
        if frame_idx >= self.num_frames:
            raise IndexError(f"帧索引 {frame_idx} 超出范围 (0-{self.num_frames-1})")

        actions_62d = self.data['actions_62d'][frame_idx]
        actions_48d = actions_62d[:48]  # 前48维是手部动作

        # 分解左右手
        left_24d = actions_48d[:24]
        right_24d = actions_48d[24:48]

        return {
            'left': self._parse_hand_data(left_24d),
            'right': self._parse_hand_data(right_24d)
        }

    def _parse_hand_data(self, hand_24d: np.ndarray) -> Dict:
        """解析单只手的24维数据"""
        # 分解 24D: wrist_pos(3), wrist_rot_6d(6), fingertips_pos(15)
        wrist_pos = hand_24d[0:3]
        wrist_rot_6d = hand_24d[3:9]
        fingertips_pos = hand_24d[9:].reshape(5, 3)  # 5个指尖位置

        return {
            'wrist_pos': wrist_pos,
            'wrist_rot_6d': wrist_rot_6d,
            'fingertips_pos': fingertips_pos
        }

    def get_all_hand_data(self) -> Dict:
        """获取所有帧的双手数据"""
        all_actions_48d = self.data['actions_62d'][:, :48]  # 所有帧的前48维

        num_frames = all_actions_48d.shape[0]

        # 分解左右手数据
        left_all = all_actions_48d[:, :24]  # 左手24维
        right_all = all_actions_48d[:, 24:48]  # 右手24维

        return {
            'left': {
                'wrist_pos': left_all[:, :3],
                'wrist_rot_6d': left_all[:, 3:9],
                'fingertips_pos': left_all[:, 9:].reshape(num_frames, 5, 3)
            },
            'right': {
                'wrist_pos': right_all[:, :3],
                'wrist_rot_6d': right_all[:, 3:9],
                'fingertips_pos': right_all[:, 9:].reshape(num_frames, 5, 3)
            }
        }

# =============================================================================
# 手部结构可视化组件
# =============================================================================

def create_hand_traces(hand_data: Dict, hand_name: str, frame_idx: int = None) -> List[go.Scatter3d]:
    """
    创建单只手的可视化轨迹

    Args:
        hand_data: 手部数据字典
        hand_name: 手的名称 ('左手' 或 '右手')
        frame_idx: 帧索引（用于显示信息）

    Returns:
        包含手腕和指尖的散点图轨迹列表
    """
    traces = []
    colors = HAND_COLORS[hand_name]

    wrist_pos = hand_data['wrist_pos']
    fingertips_pos = hand_data['fingertips_pos']

    # 手腕位置
    traces.append(go.Scatter3d(
        x=[wrist_pos[0]],
        y=[wrist_pos[1]],
        z=[wrist_pos[2]],
        mode='markers',
        name=f'{hand_name}手腕',
        marker=dict(size=12, color=colors['wrist'], symbol='diamond'),
        text=[f"{hand_name}手腕<br>坐标: ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f})"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 五个指尖位置
    for i, finger_name in enumerate(FINGER_NAMES):
        fingertip_pos = fingertips_pos[i]
        finger_key = FINGER_NAME_MAPPING[finger_name]  # 映射到英文键
        color = colors[finger_key]

        traces.append(go.Scatter3d(
            x=[fingertip_pos[0]],
            y=[fingertip_pos[1]],
            z=[fingertip_pos[2]],
            mode='markers',
            name=f'{hand_name}{finger_name}',
            marker=dict(size=8, color=color, symbol='circle'),
            text=[f"{hand_name}{finger_name}<br>坐标: ({fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}, {fingertip_pos[2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

        # 手腕到指尖的连接线
        traces.append(go.Scatter3d(
            x=[wrist_pos[0], fingertip_pos[0]],
            y=[wrist_pos[1], fingertip_pos[1]],
            z=[wrist_pos[2], fingertip_pos[2]],
            mode='lines',
            name=f'{hand_name}{finger_name}连接',
            line=dict(color=color, width=2, dash='dot'),
            hoverinfo='skip'
        ))

    return traces

def create_orientation_arrows(wrist_pos: np.ndarray, wrist_rot_6d: np.ndarray,
                           arrow_length: float = 0.05, hand_name: str = "") -> List[go.Scatter3d]:
    """
    创建表示手腕朝向的箭头（坐标系）

    Args:
        wrist_pos: 手腕位置 [x, y, z]
        wrist_rot_6d: 手腕6D旋转表示
        arrow_length: 箭头长度
        hand_name: 手的名称

    Returns:
        包含X、Y、Z三个轴箭头的列表
    """
    # 从6D表示重建旋转矩阵
    r1, r2 = wrist_rot_6d[:3], wrist_rot_6d[3:]
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3) if np.linalg.norm(r3) > 1e-6 else 1
    rot_matrix = np.column_stack([r1, r2, r3])

    # 定义三个坐标轴方向
    axes_directions = {
        'X': np.array([1, 0, 0]),  # 红色 - X轴
        'Y': np.array([0, 1, 0]),  # 绿色 - Y轴
        'Z': np.array([0, 0, 1])   # 蓝色 - Z轴
    }

    axes_colors = ['red', 'green', 'blue']
    arrows = []

    for i, (axis_name, direction) in enumerate(axes_directions.items()):
        # 旋转方向向量
        rotated_direction = rot_matrix @ direction
        end_point = wrist_pos + rotated_direction * arrow_length

        # 创建箭头线
        arrows.append(go.Scatter3d(
            x=[wrist_pos[0], end_point[0]],
            y=[wrist_pos[1], end_point[1]],
            z=[wrist_pos[2], end_point[2]],
            mode='lines',
            name=f'{hand_name}{axis_name}轴',
            line=dict(color=axes_colors[i], width=3),
            hovertemplate=f'{hand_name}{axis_name}轴<br>起点: ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f})<br>终点: ({end_point[0]:.3f}, {end_point[1]:.3f}, {end_point[2]:.3f})<extra></extra>'
        ))

    return arrows

# =============================================================================
# 主可视化函数
# =============================================================================

def visualize_egodex_hand_trajectory(hdf5_file: str):
    """主函数：可视化Egodex手部轨迹"""

    if not os.path.exists(hdf5_file):
        print(f"❌ 错误: 文件不存在 '{hdf5_file}'")
        return

    # 1. 加载数据
    print("🚀 开始可视化Egodex转换结果...")
    loader = EgodexDataLoader(hdf5_file)
    all_data = loader.get_all_hand_data()

    print("📊 准备手部轨迹数据...")
    print(f"   数据准备完成: {loader.num_frames} 帧")

    # 2. 创建Plotly 3D图表
    print("📈 生成交互式3D图表...")
    fig = go.Figure()

    # 3. 绘制初始帧（第0帧）
    initial_frame = 0
    initial_data = {
        'left': {key: all_data['left'][key][initial_frame] if key != 'fingertips_pos' else all_data['left'][key][initial_frame]
                for key in all_data['left'].keys()},
        'right': {key: all_data['right'][key][initial_frame] if key != 'fingertips_pos' else all_data['right'][key][initial_frame]
                 for key in all_data['right'].keys()}
    }

    # 添加左右手轨迹
    for hand_name in ['left', 'right']:
        hand_display_name = '左手' if hand_name == 'left' else '右手'
        hand_traces = create_hand_traces(initial_data[hand_name], hand_display_name, initial_frame)
        for trace in hand_traces:
            fig.add_trace(trace)

        # 添加手腕朝向箭头
        orientation_arrows = create_orientation_arrows(
            initial_data[hand_name]['wrist_pos'],
            initial_data[hand_name]['wrist_rot_6d'],
            arrow_length=0.06,
            hand_name=hand_display_name
        )
        for arrow in orientation_arrows:
            fig.add_trace(arrow)

    # 4. 计算坐标轴范围
    all_positions = []
    for hand_name in ['left', 'right']:
        all_positions.append(all_data[hand_name]['wrist_pos'])
        all_positions.append(all_data[hand_name]['fingertips_pos'].reshape(-1, 3))

    all_positions = np.concatenate(all_positions)
    min_coords, max_coords = np.min(all_positions, axis=0), np.max(all_positions, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.7

    # 5. 设置图表布局
    fig.update_layout(
        title=f"Egodex转换结果 - 双手轨迹可视化<br><sub>数据文件: {os.path.basename(hdf5_file)} | 总帧数: {loader.num_frames}</sub>",
        scene=dict(
            xaxis_title='X (右)',
            yaxis_title='Y (前)',
            zaxis_title='Z (上)',
            xaxis=dict(range=[center[0] - max_range, center[0] + max_range]),
            yaxis=dict(range=[center[1] - max_range, center[1] + max_range]),
            zaxis=dict(range=[center[2] - max_range, center[2] + max_range]),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=center[0] - 1.2, y=center[1] - 1.2, z=center[2] + 1.0)
            )
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
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

    # 6. 创建动画帧
    print("🎬 创建动画帧...")
    frames = []

    for i in range(loader.num_frames):
        frame_traces = []

        # 构造当前帧数据
        frame_data = {
            'left': {key: all_data['left'][key][i] if key != 'fingertips_pos' else all_data['left'][key][i]
                    for key in all_data['left'].keys()},
            'right': {key: all_data['right'][key][i] if key != 'fingertips_pos' else all_data['right'][key][i]
                     for key in all_data['right'].keys()}
        }

        # 添加左右手轨迹
        for hand_name in ['left', 'right']:
            hand_display_name = '左手' if hand_name == 'left' else '右手'
            hand_traces = create_hand_traces(frame_data[hand_name], hand_display_name, i)
            frame_traces.extend(hand_traces)

            # 添加手腕朝向箭头
            orientation_arrows = create_orientation_arrows(
                frame_data[hand_name]['wrist_pos'],
                frame_data[hand_name]['wrist_rot_6d'],
                arrow_length=0.06,
                hand_name=hand_display_name
            )
            frame_traces.extend(orientation_arrows)

        frames.append(go.Frame(data=frame_traces, name=str(i)))

    fig.frames = frames

    # 7. 显示图表
    print("✅ 可视化完成！在浏览器中打开...")
    print("💡 提示:")
    print("   - 使用播放按钮观看动画")
    print("   - 拖动时间轴滑块选择特定帧")
    print("   - 鼠标悬停查看详细坐标信息")
    print("   - 左键拖动旋转视角，右键拖动缩放")
    print("   - 红橙色系代表左手，黄绿色系代表右手")

    fig.show()

    # 8. 关闭数据文件
    loader.data.close()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Egodex转换结果可视化工具")
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='转换后的Egodex HDF5文件路径 (例如: /path/to/converted_episode0.hdf5)')

    args = parser.parse_args()

    visualize_egodex_hand_trajectory(args.file)