#!/usr/bin/env python3
"""
视频数据集预处理脚本
将MP4文件转换为HDF5+JPEG格式，为机器学习训练优化

核心功能：
1. 使用decord库进行CPU视频解码
2. 使用PyTurboJPEG进行高性能JPEG编码
3. 使用h5py库存储HDF5文件
4. 使用multiprocessing进行并行处理
5. 使用tqdm显示进度条
6. 支持断点继续处理

使用方法：
python video_preprocess.py --input_dir /path/to/Egodex-full --output_dir /path/to/Egodex-full-pretrain --num_processes 8 --jpeg_quality 95
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 第三方库
import numpy as np
import h5py
from tqdm import tqdm

# 视频处理库
try:
    import decord
    USE_DECORD = True
except ImportError:
    print("错误：未找到decord库，请安装: pip install decord")
    sys.exit(1)

# JPEG编码库
try:
    from turbojpeg import TurboJPEG
    USE_TURBOJPEG = True
except ImportError:
    print("错误：未找到PyTurboJPEG库，请安装: pip install PyTurboJPEG")
    sys.exit(1)


# 配置参数
JPEG_QUALITY = 95
NEW_DATASET_NAME = 'observation/rgb_jpeg'


def scan_video_files(input_dir: str, output_dir: str) -> List[Dict[str, str]]:
    """
    扫描目录中的所有MP4文件，支持断点继续
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        
    Returns:
        包含待处理任务信息的字典列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    video_files = []
    
    print(f"正在扫描目录: {input_dir}")
    
    # 遍历目录结构（参考select_correct_embeddings.py的逻辑）
    for part_dir in input_path.iterdir():
        if part_dir.is_dir() and part_dir.name.startswith('part'):
            print(f"扫描子目录: {part_dir.name}")
            
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    # 查找MP4文件
                    mp4_files = list(task_dir.glob('*.mp4'))
                    print(f"在 {task_dir.name} 中找到 {len(mp4_files)} 个MP4文件")
                    
                    for mp4_file in mp4_files:
                        # 构建对应的输出路径
                        relative_path = mp4_file.relative_to(input_path)
                        target_hdf5_path = output_path / relative_path
                        target_hdf5_path = target_hdf5_path.with_suffix('.hdf5')
                        
                        # 检查目标文件是否已存在（断点继续）
                        if target_hdf5_path.exists():
                            print(f"跳过已存在的文件: {target_hdf5_path}")
                            continue
                        
                        video_files.append({
                            'source_mp4_path': str(mp4_file),
                            'target_hdf5_path': str(target_hdf5_path),
                            'task_name': task_dir.name,
                            'part_name': part_dir.name
                        })
    
    print(f"总共找到 {len(video_files)} 个待处理视频文件")
    return video_files


def worker_function(task_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    工作函数：每个并行进程执行的核心逻辑
    
    Args:
        task_info: 包含任务信息的字典
        
    Returns:
        处理结果字典，失败时返回None
    """
    source_mp4_path = task_info['source_mp4_path']
    target_hdf5_path = task_info['target_hdf5_path']
    
    # 初始化TurboJPEG实例（每个进程独立初始化）
    try:
        jpeg_encoder = TurboJPEG()
    except Exception as e:
        print(f"初始化TurboJPEG失败: {e}")
        return None
    
    try:
        # 3a. 视频解码
        # 使用decord.VideoReader打开视频，指定CPU上下文
        vr = decord.VideoReader(source_mp4_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            print(f"警告: {source_mp4_path} 视频帧数为0")
            return None
        
        # 批量解码所有帧为NumPy数组
        print(f"正在解码视频: {source_mp4_path} ({total_frames}帧)")
        # 使用get_batch获取所有帧
        frames_list = vr.get_batch(list(range(total_frames)))
        # 转换为numpy数组
        if hasattr(frames_list, 'asnumpy'):
            frames_array = frames_list.asnumpy()
        else:
            frames_array = np.array(frames_list)
        
        if frames_array is None or len(frames_array) == 0:
            print(f"警告: {source_mp4_path} 解码失败")
            return None
        
        # 3b. JPEG编码（使用PyTurboJPEG）
        encoded_frames = []
        max_len = 0
        
        print(f"正在编码JPEG: {source_mp4_path}")
        
        # 使用tqdm显示单个文件的处理进度
        with tqdm(total=total_frames, desc=f"编码 {Path(source_mp4_path).name}", 
                 leave=False, position=mp.current_process().pid % 10) as pbar:
            
            for frame_rgb in frames_array:
                # PyTurboJPEG默认处理BGR格式，需要转换
                frame_bgr = frame_rgb[:, :, ::-1]  # RGB转BGR
                
                # 编码为JPEG字节流
                jpeg_bytes = jpeg_encoder.encode(frame_bgr, quality=JPEG_QUALITY)
                
                encoded_frames.append(jpeg_bytes)
                max_len = max(max_len, len(jpeg_bytes))
                pbar.update(1)
        
        # 3c. 数据填充
        padded_frames = []
        for jpeg_bytes in encoded_frames:
            # 填充到相同长度
            padded_jpeg = jpeg_bytes.ljust(max_len, b'\0')
            padded_frames.append(padded_jpeg)
        
        # 3d. 返回结果
        return {
            "status": "success",
            "source_mp4_path": source_mp4_path,
            "target_hdf5_path": target_hdf5_path,
            "padded_jpeg_frames": padded_frames,
            "dtype_str": f"|S{max_len}",
            "total_frames": total_frames,
            "max_len": max_len,
            "original_size_mb": os.path.getsize(source_mp4_path) / (1024 * 1024)
        }
        
    except Exception as e:
        print(f"处理视频失败 {source_mp4_path}: {e}")
        traceback.print_exc()
        return None


def save_to_hdf5(result: Dict[str, Any]) -> bool:
    """
    将处理结果保存到HDF5文件
    
    Args:
        result: 工作函数返回的处理结果
        
    Returns:
        保存是否成功
    """
    try:
        target_hdf5_path = result['target_hdf5_path']
        
        # 创建输出目录
        output_dir = Path(target_hdf5_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入HDF5文件
        with h5py.File(target_hdf5_path, 'w') as f:
            # 写入图像数据
            jpeg_array = np.array(result['padded_jpeg_frames'], dtype=result['dtype_str'])
            f.create_dataset(NEW_DATASET_NAME, data=jpeg_array)
            
            # 写入元数据
            metadata = f.create_group('metadata')
            metadata.attrs['total_frames'] = result['total_frames']
            metadata.attrs['max_jpeg_size'] = result['max_len']
            metadata.attrs['jpeg_quality'] = JPEG_QUALITY
            metadata.attrs['dataset_name'] = NEW_DATASET_NAME
            metadata.attrs['original_mp4_path'] = result['source_mp4_path']
            metadata.attrs['processing_time'] = time.time()
            
            # 计算并存储压缩统计
            compressed_size = os.path.getsize(target_hdf5_path)
            compression_ratio = result['original_size_mb'] / (compressed_size / (1024 * 1024))
            metadata.attrs['compression_ratio'] = compression_ratio
        
        print(f"✓ 成功保存: {target_hdf5_path} (压缩比: {compression_ratio:.2f}x)")
        return True
        
    except Exception as e:
        print(f"保存HDF5文件失败 {target_hdf5_path}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='EgoDex视频数据集预处理')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入目录路径 (例如: /path/to/Egodex-full)')
    parser.add_argument('--output_dir', type=str, 
                        help='输出目录路径 (默认: input_dir/../Egodex-full-pretrain)')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='进程数 (默认: CPU核心数)')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help=f'JPEG压缩质量 (1-100, 默认: 95)')
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式：只处理前10个文件')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        input_path = Path(args.input_dir)
        output_dir = input_path.parent / 'Egodex-full-pretrain'
    else:
        output_dir = Path(args.output_dir)
    
    # 设置进程数
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    
    # 更新全局JPEG质量设置
    global JPEG_QUALITY
    JPEG_QUALITY = args.jpeg_quality
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"进程数: {args.num_processes}")
    print(f"JPEG质量: {JPEG_QUALITY}")
    print(f"数据集名称: {NEW_DATASET_NAME}")
    
    # 步骤2: 任务发现
    video_tasks = scan_video_files(args.input_dir, str(output_dir))
    
    if not video_tasks:
        print("未找到任何待处理的MP4文件！")
        return
    
    # 测试模式：只处理前10个文件
    if args.test_mode:
        video_tasks = video_tasks[:10]
        print("测试模式：只处理前10个文件")
    
    print(f"开始处理 {len(video_tasks)} 个视频文件...")
    
    # 统计变量
    successful_files = 0
    failed_files = 0
    total_original_size = 0
    total_compressed_size = 0
    
    # 步骤3: 并行处理
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(worker_function, task): task 
                         for task in video_tasks}
        
        # 使用tqdm显示总体进度
        with tqdm(total=len(video_tasks), desc="总体进度") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    # 获取处理结果
                    result = future.result()
                    
                    if result is not None:
                        # 保存到HDF5文件
                        if save_to_hdf5(result):
                            successful_files += 1
                            total_original_size += result['original_size_mb']
                            compressed_size = os.path.getsize(result['target_hdf5_path']) / (1024 * 1024)
                            total_compressed_size += compressed_size
                            print(f"✓ 完成: {Path(result['source_mp4_path']).name} "
                                  f"({result['total_frames']}帧)")
                        else:
                            failed_files += 1
                            print(f"✗ 保存失败: {Path(task['source_mp4_path']).name}")
                    else:
                        failed_files += 1
                        print(f"✗ 处理失败: {Path(task['source_mp4_path']).name}")
                    
                except Exception as e:
                    failed_files += 1
                    print(f"✗ 处理异常: {Path(task['source_mp4_path']).name} - {e}")
                    traceback.print_exc()
                
                pbar.update(1)
    
    # 计算总统计信息
    total_time = time.time() - start_time
    overall_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
    
    # 打印最终统计
    print("\n" + "="*60)
    print("处理完成！")
    print(f"成功处理: {successful_files} 个文件")
    print(f"处理失败: {failed_files} 个文件")
    print(f"总处理时间: {total_time:.1f} 秒")
    print(f"平均速度: {total_time/len(video_tasks):.2f} 秒/文件")
    print(f"原始大小: {total_original_size:.1f} MB")
    print(f"压缩后大小: {total_compressed_size:.1f} MB")
    print(f"总体压缩比: {overall_compression_ratio:.2f}x")
    print(f"输出目录: {output_dir}")
    print(f"数据集路径: {NEW_DATASET_NAME}")
    print("="*60)
    
    # 保存处理报告
    report_path = output_dir / 'processing_report.json'
    report = {
        'input_directory': str(args.input_dir),
        'output_directory': str(output_dir),
        'num_processes': args.num_processes,
        'jpeg_quality': JPEG_QUALITY,
        'dataset_name': NEW_DATASET_NAME,
        'total_files': len(video_tasks),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_time_seconds': total_time,
        'total_original_size_mb': total_original_size,
        'total_compressed_size_mb': total_compressed_size,
        'overall_compression_ratio': overall_compression_ratio,
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"处理报告已保存到: {report_path}")


if __name__ == "__main__":
    main()