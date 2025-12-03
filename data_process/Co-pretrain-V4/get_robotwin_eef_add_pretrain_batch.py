#!/usr/bin/env python3
"""
从HDF5文件中提取endpose数据并生成eef_14d和virtual_human_real_robot_76d数据集

功能：
- 读取endpose/left_endpose和endpose/right_endpose数据
- 组合为14维EEF数据
- 生成virtual_human_real_robot_76d数据集（62D + 14D）
- 多进程处理
"""

import h5py
import numpy as np
import glob
import os
import multiprocessing
import argparse
from tqdm import tqdm

def process_file(paths):
    """处理单个文件"""
    input_path, output_path = paths

    try:
        with h5py.File(input_path, 'r') as f_in:
            # 检查必需的数据集是否存在
            left_key = 'endpose/left_endpose'
            right_key = 'endpose/right_endpose'

            if left_key not in f_in or right_key not in f_in:
                return f"No endpose data: {input_path}"

            # 读取endpose数据
            left_endpose = f_in[left_key][:]  # (N, 7)
            right_endpose = f_in[right_key][:]  # (N, 7)

            # 组合为14维EEF数据
            eef_14d = np.concatenate([left_endpose, right_endpose], axis=1)  # (N, 14)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存到已有文件
            with h5py.File(output_path, 'a') as f_out:  # 'a'模式追加写入
                # 添加eef_14d数据集
                if 'eef_14d' in f_out:
                    del f_out['eef_14d']  # 删除已存在的数据集
                dset_eef = f_out.create_dataset('eef_14d', data=eef_14d)
                dset_eef.attrs['source'] = 'endpose/left_endpose + endpose/right_endpose'

                # 添加virtual_human_real_robot_76d数据集
                if 'virtual_human_real_robot_62d' in f_out:
                    virtual_62d = f_out['virtual_human_real_robot_62d'][:]
                    # 检查帧数是否匹配
                    if virtual_62d.shape[0] != eef_14d.shape[0]:
                        raise ValueError(f"Frame count mismatch: virtual_human_real_robot_62d has {virtual_62d.shape[0]} frames, eef_14d has {eef_14d.shape[0]} frames in {output_path}")

                    virtual_76d = np.concatenate([virtual_62d, eef_14d], axis=1)

                    if 'virtual_human_real_robot_76d' in f_out:
                        del f_out['virtual_human_real_robot_76d']  # 删除已存在的数据集
                    dset_virtual = f_out.create_dataset('virtual_human_real_robot_76d', data=virtual_76d)
                    dset_virtual.attrs['description'] = '62D virtual_human_real_robot + 14D EEF endpose'
                else:
                    raise ValueError(f"No virtual_human_real_robot_62d found in {output_path}")

            return f"Added eef_14d and virtual_human_real_robot_76d: {output_path}"

    except Exception as e:
        return f"Failed: {input_path} - {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Robotwin_gen_48d_process", help='输入目录')
    parser.add_argument('--output_dir', type=str, default="/home/yinzhenhan/Human_Pre/dataset/Robotwin-gen-human-full-62d", help='输出目录')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return

    print(f"🚀 开始处理目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🛠️ 使用 {args.num_workers} 个工作进程")

    # 查找HDF5文件
    files = glob.glob(os.path.join(args.input_dir, '**', '*.hdf5'), recursive=True)
    if not files:
        print("❌ 未找到任何HDF5文件")
        return

    print(f"📊 找到 {len(files)} 个HDF5文件")

    # 构建路径对
    path_pairs = []
    for f in files:
        rel_path = os.path.relpath(f, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        path_pairs.append((f, out_path))

    # 多进程处理
    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, path_pairs),
                           total=len(path_pairs), desc="转换进度"))

    # 统计
    success = sum(1 for r in results if r.startswith('Added'))
    failed = [r for r in results if r.startswith('Failed') or r.startswith('No endpose data')]

    print(f"\n✅ 转换完成！")
    print(f"   - 成功转换: {success}")
    print(f"   - 失败: {len(failed)}")
    print(f"   - 总计: {len(files)}")

    if args.verbose and success > 0:
        print("\n成功处理的文件:")
        for r in results:
            if r.startswith('Added'):
                print(f"  {r}")

    if failed:
        print("\n失败详情:")
        for f in failed:
            print(f"  {f}")

if __name__ == '__main__':
    main()