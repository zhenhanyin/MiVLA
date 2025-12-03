#!/usr/bin/env python3
"""
删除HDF5文件中指定数据集的脚本

功能：
- 递归遍历指定目录下的所有HDF5文件
- 删除 [Dataset] action_76d 和 [Dataset] eef_14d 数据集
- 支持多进程并行处理
"""

import h5py
import os
import glob
import multiprocessing
import argparse
from tqdm import tqdm

def process_file(file_path):
    """处理单个HDF5文件，删除指定数据集"""
    try:
        datasets_to_remove = ['action_76d', 'eef_14d']
        removed_datasets = []

        # 检查文件中是否包含要删除的数据集
        with h5py.File(file_path, 'r') as f:
            existing_datasets = [ds for ds in datasets_to_remove if ds in f]

        if not existing_datasets:
            return f"No datasets to remove: {file_path}"

        # 以读写模式打开文件并删除数据集
        with h5py.File(file_path, 'a') as f:
            for dataset_name in existing_datasets:
                del f[dataset_name]
                removed_datasets.append(dataset_name)

        return f"Removed {removed_datasets}: {file_path}"

    except Exception as e:
        return f"Failed to process {file_path}: {e}"

def find_hdf5_files(directory, exclude_dirs=None):
    """递归查找所有HDF5文件"""
    if exclude_dirs is None:
        exclude_dirs = []

    hdf5_files = []
    print(f"🔍 搜索HDF5文件于 '{directory}'")
    if exclude_dirs:
        print(f"排除目录: {exclude_dirs}")

    for root, dirs, files in os.walk(directory):
        # 排除指定目录
        if root == directory:
            dirs_to_remove = [d for d in dirs if d in exclude_dirs]
            for d in dirs_to_remove:
                dirs.remove(d)

        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))

    return hdf5_files

def main():
    parser = argparse.ArgumentParser(description='删除HDF5文件中的指定数据集')
    parser.add_argument('--directory', type=str, required=True, help='要处理的目录路径')
    parser.add_argument('--num_workers', type=int, default=24, help='工作进程数')
    parser.add_argument('--exclude_dirs', nargs='+', default=[], help='排除的目录')
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"❌ 目录不存在: {args.directory}")
        return

    print(f"🚀 开始处理目录: {args.directory}")
    print(f"🛠️ 使用 {args.num_workers} 个工作进程")

    # 查找所有HDF5文件
    hdf5_files = find_hdf5_files(args.directory, exclude_dirs=args.exclude_dirs)
    if not hdf5_files:
        print("❌ 未找到任何HDF5文件")
        return

    print(f"📊 找到 {len(hdf5_files)} 个HDF5文件")

    # 多进程处理
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, hdf5_files),
                          total=len(hdf5_files), desc="删除数据集"))

    # 统计结果
    removed_count = sum(1 for r in results if r.startswith('Removed'))
    no_datasets_count = sum(1 for r in results if r.startswith('No datasets'))
    failed_count = sum(1 for r in results if r.startswith('Failed'))

    print(f"\n✅ 处理完成！")
    print(f"   - 成功删除数据集: {removed_count}")
    print(f"   - 无数据集可删除: {no_datasets_count}")
    print(f"   - 处理失败: {failed_count}")
    print(f"   - 总文件数: {len(hdf5_files)}")

    if args.verbose and removed_count > 0:
        print(f"\n成功处理的文件:")
        for r in results:
            if r.startswith('Removed'):
                print(f"  {r}")

    if failed_count > 0:
        print(f"\n失败的文件:")
        for r in results:
            if r.startswith('Failed'):
                print(f"  {r}")

if __name__ == '__main__':
    main()