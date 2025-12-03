#!/usr/bin/env python3
"""
查看HDF5文件中62维数据的后14维机械臂关节数据
每30帧打印一次数值内容，便于分析数据分布和变化趋势
"""

import numpy as np
import h5py
import argparse
import sys

def print_14d_analysis(hdf5_file_path, frame_interval=30):
    """分析并打印62维数据中的后14维内容"""
    
    print("=" * 80)
    print(f"📊 分析文件: {hdf5_file_path}")
    print("=" * 80)
    
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 检查数据集
            if 'actions_62d' not in f:
                print(f"❌ 错误: 文件中未找到 'actions_62d' 数据集")
                print(f"可用数据集: {list(f.keys())}")
                return
            
            actions_62d = f['actions_62d'][:]
            num_frames, total_dims = actions_62d.shape
            
            print(f"📈 数据基本信息:")
            print(f"   总帧数: {num_frames}")
            print(f"   总维度: {total_dims}")
            print(f"   数据类型: {actions_62d.dtype}")
            print(f"   显示间隔: 每 {frame_interval} 帧")
            
            # 提取后14维数据
            robot_data = actions_62d[:, 48:]  # 后14维是机械臂关节数据
            
            print(f"\n🤖 后14维机械臂关节数据说明:")
            print(f"   维度 0-5:   左臂6个关节角度")
            print(f"   维度 6:     左臂夹爪开合状态 (0=闭合, 1=张开)")
            print(f"   维度 7-12:  右臂6个关节角度") 
            print(f"   维度 13:    右臂夹爪开合状态 (0=闭合, 1=张开)")
            
            # 计算统计信息
            print(f"\n📊 整体统计信息:")
            print(f"   数据范围: [{robot_data.min():.6f}, {robot_data.max():.6f}]")
            print(f"   平均值: {robot_data.mean():.6f}")
            print(f"   标准差: {robot_data.std():.6f}")
            
            # 分别统计左臂、右臂、夹爪
            left_arm = robot_data[:, :6]      # 左臂关节
            left_gripper = robot_data[:, 6]   # 左夹爪
            right_arm = robot_data[:, 7:13]   # 右臂关节
            right_gripper = robot_data[:, 13] # 右夹爪
            
            print(f"\n📊 分部分统计:")
            print(f"   左臂关节 (维0-5):   范围[{left_arm.min():.3f}, {left_arm.max():.3f}], 均值{left_arm.mean():.3f}")
            print(f"   左夹爪 (维6):       范围[{left_gripper.min():.3f}, {left_gripper.max():.3f}], 均值{left_gripper.mean():.3f}")
            print(f"   右臂关节 (维7-12):  范围[{right_arm.min():.3f}, {right_arm.max():.3f}], 均值{right_arm.mean():.3f}")
            print(f"   右夹爪 (维13):      范围[{right_gripper.min():.3f}, {right_gripper.max():.3f}], 均值{right_gripper.mean():.3f}")
            
            # 检查夹爪状态分布
            left_closed = np.sum(left_gripper < 0.3)
            left_open = np.sum(left_gripper > 0.7)
            right_closed = np.sum(right_gripper < 0.3)
            right_open = np.sum(right_gripper > 0.7)
            
            print(f"\n🤏 夹爪状态分析:")
            print(f"   左夹爪闭合状态(<0.3): {left_closed}帧 ({left_closed/num_frames*100:.1f}%)")
            print(f"   左夹爪张开状态(>0.7): {left_open}帧 ({left_open/num_frames*100:.1f}%)")
            print(f"   右夹爪闭合状态(<0.3): {right_closed}帧 ({right_closed/num_frames*100:.1f}%)")
            print(f"   右夹爪张开状态(>0.7): {right_open}帧 ({right_open/num_frames*100:.1f}%)")
            
            # 详细帧数据
            print(f"\n" + "=" * 80)
            print(f"🔍 详细帧数据 (每 {frame_interval} 帧):")
            print("=" * 80)
            
            # 表头
            header = "帧序号 | " + " | ".join([f"左臂{i}" for i in range(6)]) + " | 左夹爪 | " + \
                    " | ".join([f"右臂{i}" for i in range(6)]) + " | 右夹爪"
            print(header)
            print("-" * len(header))
            
            # 打印数据
            for frame_idx in range(0, num_frames, frame_interval):
                frame_data = robot_data[frame_idx]
                
                # 格式化输出
                row = f"{frame_idx:5d} | "
                row += " | ".join([f"{x:7.3f}" for x in frame_data[:6]]) + " | "
                row += f"{frame_data[6]:6.3f} | "
                row += " | ".join([f"{x:7.3f}" for x in frame_data[7:13]]) + " | "
                row += f"{frame_data[13]:6.3f}"
                
                print(row)
            
            # 最后5帧数据
            if num_frames > frame_interval:
                print(f"\n... (中间省略 {num_frames - frame_interval - 5} 帧) ...")
                print("\n最后5帧数据:")
                print("-" * len(header))
                
                for frame_idx in range(max(frame_interval, num_frames-5), num_frames):
                    frame_data = robot_data[frame_idx]
                    
                    row = f"{frame_idx:5d} | "
                    row += " | ".join([f"{x:7.3f}" for x in frame_data[:6]]) + " | "
                    row += f"{frame_data[6]:6.3f} | "
                    row += " | ".join([f"{x:7.3f}" for x in frame_data[7:13]]) + " | "
                    row += f"{frame_data[13]:6.3f}"
                    
                    print(row)
            
            # 异常值检测
            print(f"\n" + "=" * 80)
            print(f"⚠️  异常值检测:")
            print("=" * 80)
            
            # 检查关节角度是否超出合理范围
            joint_threshold = 3.0  # 弧度
            outlier_frames = []
            
            for frame_idx in range(num_frames):
                frame_data = robot_data[frame_idx]
                
                # 检查关节角度 (排除夹爪)
                joint_angles = np.concatenate([frame_data[:6], frame_data[7:13]])
                if np.any(np.abs(joint_angles) > joint_threshold):
                    outlier_frames.append(frame_idx)
            
            if outlier_frames:
                print(f"发现 {len(outlier_frames)} 帧有关节角度超出 ±{joint_threshold} 弧度:")
                for frame in outlier_frames[:10]:  # 显示前10个异常帧
                    max_joint = np.max(np.abs(np.concatenate([robot_data[frame, :6], robot_data[frame, 7:13]])))
                    print(f"   帧 {frame}: 最大关节角度 {max_joint:.3f} 弧度")
                if len(outlier_frames) > 10:
                    print(f"   ... 还有 {len(outlier_frames) - 10} 帧异常")
            else:
                print(f"✅ 未发现明显的关节角度异常值 (阈值: ±{joint_threshold} 弧度)")
            
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="查看HDF5文件中62维数据的后14维机械臂关节数据")
    parser.add_argument('file_path', type=str, help='HDF5文件路径')
    parser.add_argument('--interval', type=int, default=30, 
                       help='显示间隔帧数 (默认: 30)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    import os
    if not os.path.exists(args.file_path):
        print(f"❌ 文件不存在: {args.file_path}")
        sys.exit(1)
    
    # 分析文件
    success = print_14d_analysis(args.file_path, args.interval)
    
    if success:
        print(f"\n✅ 分析完成!")
    else:
        print(f"\n❌ 分析失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()