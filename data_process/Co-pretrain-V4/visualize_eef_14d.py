#!/usr/bin/env python3
"""
可视化HDF5文件中actions_76d数据集后14维的EEF点位置

功能：
- 读取HDF5文件中的actions_76d数据集
- 提取后14维（左臂7维 + 右臂7维）的EEF位姿信息
- 使用plotly可视化两个EEF点的当前位置
- 包含时间轴滑块和播放控制按钮
- 不显示轨迹线，只显示当前帧位置
"""

import h5py
import numpy as np
import plotly.graph_objects as go
import argparse
import os

def extract_eef_positions(actions_76d):
    """
    从actions_76d数据中提取EEF位置信息

    Args:
        actions_76d: (N, 76) 动作数据

    Returns:
        left_eef_pos: (N, 3) 左臂EEF位置
        right_eef_pos: (N, 3) 右臂EEF位置
    """
    # 后14维格式：[左臂x,y,z,qw,qx,qy,qz, 右臂x,y,z,qw,qx,qy,qz]
    eef_14d = actions_76d[:, -14:]  # (N, 14)

    # 提取位置信息（每7维的前3维）
    left_eef_pos = eef_14d[:, 0:3]   # 左臂位置 [x, y, z]
    right_eef_pos = eef_14d[:, 7:10] # 右臂位置 [x, y, z]

    return left_eef_pos, right_eef_pos

def visualize_eef_positions_with_animation(left_pos, right_pos, title="EEF Positions"):
    """
    可视化EEF点的当前位置，包含动画控制

    Args:
        left_pos: (N, 3) 左臂EEF位置
        right_pos: (N, 3) 右臂EEF位置
        title: 图表标题
    """
    fig = go.Figure()

    num_frames = len(left_pos)
    initial_frame = 0

    # 左臂EEF初始点
    fig.add_trace(go.Scatter3d(
        x=[left_pos[initial_frame, 0]],
        y=[left_pos[initial_frame, 1]],
        z=[left_pos[initial_frame, 2]],
        mode='markers',
        name='左臂EEF',
        marker=dict(size=15, color='blue', symbol='diamond'),
        text=[f"左臂EEF<br>帧: {initial_frame}<br>坐标: ({left_pos[initial_frame, 0]:.3f}, {left_pos[initial_frame, 1]:.3f}, {left_pos[initial_frame, 2]:.3f})"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 右臂EEF初始点
    fig.add_trace(go.Scatter3d(
        x=[right_pos[initial_frame, 0]],
        y=[right_pos[initial_frame, 1]],
        z=[right_pos[initial_frame, 2]],
        mode='markers',
        name='右臂EEF',
        marker=dict(size=15, color='red', symbol='diamond'),
        text=[f"右臂EEF<br>帧: {initial_frame}<br>坐标: ({right_pos[initial_frame, 0]:.3f}, {right_pos[initial_frame, 1]:.3f}, {right_pos[initial_frame, 2]:.3f})"],
        hovertemplate='%{text}<extra></extra>'
    ))

    # 计算坐标轴范围
    all_positions = np.concatenate([left_pos, right_pos])
    min_coords, max_coords = np.min(all_positions, axis=0), np.max(all_positions, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.8

    # 设置图表布局
    fig.update_layout(
        title=f"EEF位置可视化<br><sub>数据文件: {title} | 总帧数: {num_frames}</sub>",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
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
        width=1000,
        height=800
    )

    # 添加播放控制按钮
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'showactive': True,
            'pad': {'t': 20},
            'buttons': [
                {
                    'label': '播放 (30 FPS)',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 33.33, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '暂停',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }]
    )

    # 添加时间轴滑块
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': '帧: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.05,
            'steps': [
                {
                    'label': str(i),
                    'method': 'animate',
                    'args': [[str(i)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for i in range(num_frames)
            ]
        }]
    )

    # 创建动画帧
    print("🎬 创建动画帧...")
    frames = []

    for i in range(num_frames):
        frame_traces = []

        # 当前帧的左臂EEF点
        frame_traces.append(go.Scatter3d(
            x=[left_pos[i, 0]],
            y=[left_pos[i, 1]],
            z=[left_pos[i, 2]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='diamond'),
            showlegend=False,
            text=[f"左臂EEF<br>帧: {i}<br>坐标: ({left_pos[i, 0]:.3f}, {left_pos[i, 1]:.3f}, {left_pos[i, 2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

        # 当前帧的右臂EEF点
        frame_traces.append(go.Scatter3d(
            x=[right_pos[i, 0]],
            y=[right_pos[i, 1]],
            z=[right_pos[i, 2]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            showlegend=False,
            text=[f"右臂EEF<br>帧: {i}<br>坐标: ({right_pos[i, 0]:.3f}, {right_pos[i, 1]:.3f}, {right_pos[i, 2]:.3f})"],
            hovertemplate='%{text}<extra></extra>'
        ))

        frames.append(go.Frame(data=frame_traces, name=str(i)))

    fig.frames = frames

    return fig

def main():
    parser = argparse.ArgumentParser(description='可视化HDF5文件中的EEF位置')
    parser.add_argument('--file_path', type=str, required=True, help='HDF5文件路径')
    parser.add_argument('--dataset_name', type=str, default='actions_76d', help='数据集名称')
    parser.add_argument('--output_html', type=str, help='输出HTML文件路径（可选）')
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"❌ 文件不存在: {args.file_path}")
        return

    try:
        # 读取HDF5文件
        with h5py.File(args.file_path, 'r') as f:
            if args.dataset_name not in f:
                print(f"❌ 数据集 '{args.dataset_name}' 不存在于文件中")
                print(f"可用数据集: {list(f.keys())}")
                return

            actions_76d = f[args.dataset_name][:]
            print(f"📊 读取数据集 '{args.dataset_name}': {actions_76d.shape}")

            if actions_76d.shape[1] < 76:
                print(f"❌ 数据集维度不足，期望76维，实际{actions_76d.shape[1]}维")
                return

        # 提取EEF位置
        left_pos, right_pos = extract_eef_positions(actions_76d)
        print(f"🔍 提取EEF位置信息:")
        print(f"   - 左臂EEF: {left_pos.shape}")
        print(f"   - 右臂EEF: {right_pos.shape}")
        print(f"   - 轨迹长度: {len(left_pos)} 帧")

        # 打印位置范围信息
        print(f"📏 位置范围:")
        print(f"   - 左臂 X: [{left_pos[:, 0].min():.3f}, {left_pos[:, 0].max():.3f}]")
        print(f"   - 左臂 Y: [{left_pos[:, 1].min():.3f}, {left_pos[:, 1].max():.3f}]")
        print(f"   - 左臂 Z: [{left_pos[:, 2].min():.3f}, {left_pos[:, 2].max():.3f}]")
        print(f"   - 右臂 X: [{right_pos[:, 0].min():.3f}, {right_pos[:, 0].max():.3f}]")
        print(f"   - 右臂 Y: [{right_pos[:, 1].min():.3f}, {right_pos[:, 1].max():.3f}]")
        print(f"   - 右臂 Z: [{right_pos[:, 2].min():.3f}, {right_pos[:, 2].max():.3f}]")

        # 创建可视化图表
        title = os.path.basename(args.file_path)
        fig = visualize_eef_positions_with_animation(left_pos, right_pos, title)

        # 保存HTML文件
        if args.output_html:
            fig.write_html(args.output_html)
            print(f"💾 图表已保存到: {args.output_html}")

        # 显示图表
        print("✅ 可视化完成！在浏览器中打开...")
        print("💡 提示:")
        print("   - 使用播放按钮观看动画")
        print("   - 拖动时间轴滑块选择特定帧")
        print("   - 鼠标悬停查看详细坐标信息")
        print("   - 左键拖动旋转视角，右键拖动缩放")

        fig.show()

    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")

if __name__ == '__main__':
    main()