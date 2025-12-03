#!/usr/bin/env python3
"""
检查14D动作数据的具体内容和统计信息（新增：检测连续重复帧）
"""

import h5py
import numpy as np
import os
import sys
from pathlib import Path

def check_14d_data(hdf5_path):
    """检查14D数据的详细内容"""
    print(f"\n{'='*60}")
    print(f"检查14D数据文件: {hdf5_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(hdf5_path):
        print(f"❌ 文件不存在: {hdf5_path}")
        return
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查数据集结构
            print("\n📁 文件结构:")
            print_keys_recursive(f)
            
            # 检查action_14d数据集
            if 'action_14d' in f:
                action_14d = f['action_14d'][:]
                print(f"\n📊 action_14d数据集信息:")
                print(f"   形状: {action_14d.shape}")
                print(f"   数据类型: {action_14d.dtype}")
                print(f"   总帧数: {len(action_14d)}")
                
                # 检查维度结构
                print(f"\n🔍 14D维度结构分析:")
                print(f"   维度 0-5:  左臂6个关节角度")
                print(f"   维度 6-11: 右臂6个关节角度") 
                print(f"   维度 12:   左夹爪开合度")
                print(f"   维度 13:   右夹爪开合度")
                
                # 统计每个维度的数据范围
                print(f"\n📈 各维度数据统计:")
                for i in range(14):
                    data = action_14d[:, i]
                    print(f"   维度{i:2d}: 最小值={data.min():8.4f}, 最大值={data.max():8.4f}, "
                          f"平均值={data.mean():8.4f}, 标准差={data.std():8.4f}")
                
                # 特殊分析
                print(f"\n🔧 特殊分析:")
                
                # 左臂关节分析
                left_arm = action_14d[:, :6]
                print(f"   左臂关节范围:")
                for i in range(6):
                    print(f"     关节{i}: [{left_arm[:, i].min():6.3f}, {left_arm[:, i].max():6.3f}]")
                
                # 右臂关节分析
                right_arm = action_14d[:, 6:12]
                print(f"   右臂关节范围:")
                for i in range(6):
                    print(f"     关节{i}: [{right_arm[:, i].min():6.3f}, {right_arm[:, i].max():6.3f}]")
                
                # 夹爪分析
                left_gripper = action_14d[:, 12]
                right_gripper = action_14d[:, 13]
                print(f"   左夹爪: 范围=[{left_gripper.min():.3f}, {left_gripper.max():.3f}], "
                      f"平均值={left_gripper.mean():.3f}")
                print(f"   右夹爪: 范围=[{right_gripper.min():.3f}, {right_gripper.max():.3f}], "
                      f"平均值={right_gripper.mean():.3f}")
                
                # 数据变化分析
                print(f"\n📊 数据变化分析:")
                for i in range(14):
                    data = action_14d[:, i]
                    # 计算变化幅度
                    data_range = data.max() - data.min()
                    # 计算变化次数（相邻帧差异大于阈值）
                    changes = np.sum(np.abs(np.diff(data)) > 1e-6)
                    print(f"   维度{i:2d}: 变化幅度={data_range:8.4f}, 变化次数={changes:3d}/{len(data)-1}")
                
                # 显示前几帧的详细数据
                print(f"\n📋 前5帧详细数据:")
                for frame in range(min(5, len(action_14d))):
                    formatted_frame = [f"{x:.3f}" for x in action_14d[frame]]
                    print(f"   帧{frame}: [{', '.join(formatted_frame)}]")
                
                # 检查是否有异常值
                print(f"\n⚠️  异常值检查:")
                for i in range(14):
                    data = action_14d[:, i]
                    # 检查NaN或Inf
                    if np.any(np.isnan(data)):
                        print(f"   维度{i:2d}: 发现NaN值!")
                    if np.any(np.isinf(data)):
                        print(f"   维度{i:2d}: 发现Inf值!")
                    
                    # 检查关节角度是否在合理范围内
                    if i < 12:  # 关节角度
                        if np.any(np.abs(data) > np.pi):  # 超过±180度
                            print(f"   维度{i:2d}: 关节角度可能过大! 范围:[{data.min():.3f}, {data.max():.3f}]")
                    else:  # 夹爪
                        if np.any(data < 0) or np.any(data > 1):
                            print(f"   维度{i:2d}: 夹爪值超出[0,1]范围! 范围:[{data.min():.3f}, {data.max():.3f}]")
                
                # 检查连续性
                print(f"\n🔗 连续性检查:")
                total_gaps = 0
                for i in range(14):
                    data = action_14d[:, i]
                    # 检查是否有突变（相邻帧变化过大）
                    if len(data) > 1:
                        diffs = np.abs(np.diff(data))
                        threshold = 3 * data.std()
                        if threshold > 0:
                            large_changes = np.sum(diffs > threshold)
                            if large_changes > 0:
                                print(f"   维度{i:2d}: 发现{large_changes}处突变")
                                total_gaps += large_changes
                
                if total_gaps == 0:
                    print("   ✅ 数据连续性良好")
                else:
                    print(f"   ⚠️  共发现{total_gaps}处突变")
                
                # ✅ 新增功能：检测连续重复帧
                print(f"\n🔁 检测连续重复帧（与上一帧完全一致）:")
                repeated_frames = []
                total_frames = len(action_14d)
                
                for i in range(1, total_frames):
                    if np.array_equal(action_14d[i], action_14d[i-1]):
                        repeated_frames.append(i)
                
                if len(repeated_frames) == 0:
                    print("   ✅ 未发现任何连续重复帧 —— 数据无冻结！")
                else:
                    print(f"   ⚠️  发现 {len(repeated_frames)} 个连续重复帧！")
                    print(f"   重复帧编号（从1开始）: {repeated_frames[:10]}{'...' if len(repeated_frames) > 10 else ''}")
                    
                    # 显示前5个重复帧的详细内容
                    print(f"\n   前5个重复帧详情（当前帧 & 上一帧）:")
                    for idx in repeated_frames[:5]:
                        prev = action_14d[idx - 1]
                        curr = action_14d[idx]
                        # 格式化为一行字符串
                        prev_str = ", ".join(f"{x:.3f}" for x in prev)
                        curr_str = ", ".join(f"{x:.3f}" for x in curr)
                        print(f"     帧{idx-1:4d}: [{prev_str}]")
                        print(f"     帧{idx:4d}: [{curr_str}] ← 完全相同！")
                        print()
                
                # 统计最长连续重复序列
                if repeated_frames:
                    max_run = 1
                    current_run = 1
                    for i in range(1, len(repeated_frames)):
                        if repeated_frames[i] == repeated_frames[i-1] + 1:
                            current_run += 1
                            max_run = max(max_run, current_run)
                        else:
                            current_run = 1
                    print(f"   最长连续重复帧序列长度: {max_run} 帧")

                # 检查元数据
                print(f"\n📝 元数据:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"   {attr_name}: {attr_value}")
                
            else:
                print("❌ 未找到action_14d数据集")
                
            # 检查其他常见数据集
            common_datasets = ['transforms/leftHand', 'transforms/rightHand', 'actions_48d']
            for ds in common_datasets:
                if ds in f:
                    print(f"   [✓] 存在 {ds}: {f[ds].shape}")
                else:
                    print(f"   [ ] 缺失 {ds}")
                
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        import traceback
        traceback.print_exc()

def print_keys_recursive(group, prefix=""):
    """递归打印HDF5文件结构"""
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            print(f"   {prefix}[Dataset] {key}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"   {prefix}[Group] {key}")
            print_keys_recursive(item, prefix + "  ")

def main():
    """主函数"""
    # 文件路径
    hdf5_file = "/home/yinzhenhan/Human_Pre/dataset/zhenji/put_the_cup_on_the_coaster/data/episode_1.hdf5"
    
    # 如果文件不存在，尝试备用路径
    if not os.path.exists(hdf5_file):
        backup_file = "/home/yinzhenhan/Human_Pre/H_RDT/datasets/robotwin2/single_task_clean/place_object_basket/data/episode0.hdf5"
        if os.path.exists(backup_file):
            hdf5_file = backup_file
            print(f"📁 主文件不存在，使用备用文件: {hdf5_file}")
        else:
            print(f"❌ 两个文件都不存在:")
            print(f"   主文件: {hdf5_file}")
            print(f"   备用文件: {backup_file}")
            return
    
    # 检查14D数据
    check_14d_data(hdf5_file)
    
    print(f"\n{'='*60}")
    print("✅ 14D数据检查完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()