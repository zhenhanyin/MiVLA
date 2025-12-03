#!/usr/bin/env python3
"""
查看HDF5文件中virtual_human_real_robot_62d数据集的前3帧和最后3帧数据
支持指定单个HDF5文件路径进行查看
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def print_hdf5_dataset_frames(hdf5_path: str, dataset_name: str = "virtual_human_real_robot_62d"):
    """
    读取HDF5文件，打印指定数据集的前3帧和最后3帧数据
    
    Args:
        hdf5_path: HDF5文件的完整路径
        dataset_name: 要查看的数据集名称，默认是virtual_human_real_robot_62d
    """
    # 检查文件是否存在
    hdf5_file = Path(hdf5_path)
    if not hdf5_file.exists():
        print(f"❌ 错误：文件不存在 - {hdf5_path}")
        return
    if not hdf5_file.is_file():
        print(f"❌ 错误：{hdf5_path} 不是一个有效文件")
        return

    print(f"\n--- 正在检查文件: {hdf5_path} ---")
    try:
        # 只读模式打开HDF5文件
        with h5py.File(hdf5_path, 'r') as f:
            # 1. 打印语言元数据（llm_description）
            print("\n>>> 语言元数据 (Attributes):")
            if 'llm_description' in f.attrs:
                print(f"  llm_description: {f.attrs['llm_description']}")
            else:
                print("  未找到 'llm_description' 属性。")
            print("--------------------")

            # 2. 检查指定数据集是否存在
            print("\n>>> 文件内部结构 (Datasets and Groups):")
            if dataset_name not in f:
                print(f"  ❌ 未找到指定数据集: {dataset_name}")
                # 可选：打印文件中所有可用的数据集，方便用户确认正确名称
                print("  📋 文件中可用的数据集如下:")
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        print(f"    - [Dataset] {key}: shape={f[key].shape}, dtype={f[key].dtype}")
                return

            # 3. 获取数据集基本信息
            dataset = f[dataset_name]
            print(f"  - [Dataset] {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
            total_frames = dataset.shape[0]
            data_dim = dataset.shape[1] if len(dataset.shape) > 1 else 1
            print(f"    数据集总帧数: {total_frames}, 每帧数据维度: {data_dim}")
            print("--------------------")

            # 4. 确定要打印的帧数（处理总帧数不足6帧的情况）
            start_frames = min(3, total_frames)
            end_frames = min(3, total_frames)

            # 5. 打印前N帧数据
            print(f"\n>>> 前 {start_frames} 帧数据:")
            for i in range(start_frames):
                frame_data = dataset[i]
                print(f"\n  第 {i+1} 帧:")
                # 按维度分段打印，避免一行过长（62维数据分4段，每段15-16个数值）
                segment_size = 16
                for seg_idx in range(0, data_dim, segment_size):
                    end_seg = min(seg_idx + segment_size, data_dim)
                    seg_data = frame_data[seg_idx:end_seg]
                    # 保留6位小数，确保数据精度且格式整齐
                    print(f"    维度 [{seg_idx+1}-{end_seg}]: {np.round(seg_data, 6)}")

            # 6. 打印最后N帧数据（若总帧数大于start_frames，避免重复打印）
            if total_frames > start_frames:
                print(f"\n>>> 最后 {end_frames} 帧数据:")
                for i in range(total_frames - end_frames, total_frames):
                    frame_data = dataset[i]
                    print(f"\n  第 {i+1} 帧 (倒数第 {total_frames - i} 帧):")
                    # 同样分段打印
                    for seg_idx in range(0, data_dim, segment_size):
                        end_seg = min(seg_idx + segment_size, data_dim)
                        seg_data = frame_data[seg_idx:end_seg]
                        print(f"    维度 [{seg_idx+1}-{end_seg}]: {np.round(seg_data, 6)}")

            print(f"\n✅ 数据查看完成！共查看 {start_frames + end_frames} 帧数据")

    except Exception as e:
        print(f"\n❌ 读取文件时发生错误: {str(e)}")
        return


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="查看HDF5文件中virtual_human_real_robot_62d数据集的前后帧数据")
    parser.add_argument("hdf5_path", type=str, help="HDF5文件的完整路径（例如：/home/xxx/episode0.hdf5）")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="virtual_human_real_robot_62d",
        help="可选：指定要查看的数据集名称，默认是virtual_human_real_robot_62d"
    )
    args = parser.parse_args()

    # 调用核心函数查看数据
    print_hdf5_dataset_frames(args.hdf5_path, args.dataset)


if __name__ == "__main__":
    main()