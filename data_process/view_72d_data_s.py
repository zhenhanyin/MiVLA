#!/usr/bin/env python3
"""
一个用于查看特定HDF5文件中 'virtual_human_real_robot_76d' 和 'success_rate' 数据集的脚本。
"""

import h5py
import numpy as np
import argparse
import os

def view_hdf5_data(file_path: str, num_rows: int = 5):
    """
    读取并显示HDF5文件中的 'virtual_human_real_robot_76d' 和 'success_rate' 数据。

    Args:
        file_path (str): HDF5文件的路径。
        num_rows (int): 要显示的virtual_human_real_robot_76d数据的前后行数。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    print(f"--- 正在检查文件: {file_path} ---\n")

    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 查看 success_rate
            print(">>> 成功率 (success_rate):")
            if 'success_rate' in f:
                success_rate = f['success_rate'][()]  # [()] 用于读取标量值
                print(f"  - 值: {success_rate:.4f} ({success_rate * 100:.2f}%)")
            else:
                print("  - 未找到 'success_rate' 数据集。")
            
            print("-" * 20)

            # 2. 查看 virtual_human_real_robot_76d
            print(">>> 72维动作数据 (virtual_human_real_robot_76d):")
            if 'virtual_human_real_robot_76d' in f:
                virtual_human_real_robot_76d_dset = f['virtual_human_real_robot_76d']
                
                # 打印基本信息
                print(f"  - 形状 (Shape): {virtual_human_real_robot_76d_dset.shape}")
                print(f"  - 数据类型 (Dtype): {virtual_human_real_robot_76d_dset.dtype}")

                num_frames = virtual_human_real_robot_76d_dset.shape[0]

                if num_frames > 0:
                    # 为了方便查看，设置numpy打印选项
                    np.set_printoptions(precision=4, suppress=True, linewidth=150)

                    # 显示前几行
                    print(f"\n  --- 前 {min(num_rows, num_frames)} 行数据 ---")
                    print(virtual_human_real_robot_76d_dset[:num_rows, :])

                    # 如果总行数大于要显示的两倍，则显示后几行
                    if num_frames > num_rows * 2:
                        print("\n  ...")
                        print(f"\n  --- 后 {min(num_rows, num_frames)} 行数据 ---")
                        print(virtual_human_real_robot_76d_dset[-num_rows:, :])
                else:
                    print("\n  - 数据集为空 (0帧)。")

            else:
                print("  - 未找到 'virtual_human_real_robot_76d' 数据集。")

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")

def main():
    """主函数，用于解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="查看HDF5文件中的 'virtual_human_real_robot_76d' 和 'success_rate' 数据。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'file_path', 
        type=str, 
        help='要检查的HDF5文件的路径。'
    )
    parser.add_argument(
        '--rows', 
        type=int, 
        default=5, 
        help='要预览的virtual_human_real_robot_76d数据的前后行数 (默认: 5)。'
    )

    args = parser.parse_args()
    view_hdf5_data(args.file_path, args.rows)

if __name__ == "__main__":
    main()