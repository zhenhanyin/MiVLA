#!/usr/bin/env python3
"""
真机视频数据预处理脚本
将MP4视频文件转换为HDF5+JPEG格式，添加到现有HDF5文件中

核心功能：
1. 使用decord库进行CPU视频解码
2. 使用PyTurboJPEG进行高性能JPEG编码
3. 将图像数据添加到现有HDF5文件中的observation/rgb_jpeg
4. 使用multiprocessing进行并行处理
5. 使用tqdm显示进度条
6. 支持断点继续处理

使用方法：
python simple_video_preprocess.py --input_dir /path/to/zhenji --camera_name left_wrist --jpeg_quality 95
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
import cv2

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
DATASET_NAME = 'observation/head_camera/rgb'


def scan_video_files(input_dir: str, camera_name: str) -> List[Dict[str, str]]:
    """
    扫描目录中的所有HDF5和对应的MP4文件
    
    Args:
        input_dir: 输入目录路径
        camera_name: 相机名称（如left_wrist）
        
    Returns:
        包含待处理任务信息的字典列表
    """
    input_path = Path(input_dir)
    video_files = []
    
    print(f"正在扫描目录: {input_dir}")
    
    # 检查当前目录是否有文件（扁平结构）
    hdf5_files = list(input_path.glob('*.hdf5'))
    
    if hdf5_files:
        print(f"扫描当前目录: {input_path.name}")
        print(f"在当前目录中找到 {len(hdf5_files)} 个HDF5文件")
        
        # 处理扁平结构
        for hdf5_file in hdf5_files:
            # 查找对应的MP4文件
            episode_name = hdf5_file.stem
            mp4_file = input_path / f"{episode_name}_{camera_name}.mp4"
            
            if mp4_file.exists():
                # 检查HDF5文件中是否已有图像数据
                if check_image_data_exists(hdf5_file):
                    print(f"跳过已处理的文件: {hdf5_file}")
                    continue
                
                video_files.append({
                    'hdf5_path': str(hdf5_file),
                    'mp4_path': str(mp4_file),
                    'task_name': input_path.name,
                    'episode_name': episode_name,
                    'camera_name': camera_name
                })
            else:
                print(f"警告: 未找到对应的视频文件 {mp4_file}")
    else:
        # 如果没有找到文件，尝试原有的子目录结构
        for task_dir in input_path.iterdir():
            if task_dir.is_dir():
                print(f"扫描任务目录: {task_dir.name}")
                
                # 查找HDF5文件
                hdf5_files = list(task_dir.glob('*.hdf5'))
                
                for hdf5_file in hdf5_files:
                    # 查找对应的MP4文件
                    episode_name = hdf5_file.stem
                    mp4_file = task_dir / f"{episode_name}_{camera_name}.mp4"
                    
                    if mp4_file.exists():
                        # 检查HDF5文件中是否已有图像数据
                        if check_image_data_exists(hdf5_file):
                            print(f"跳过已处理的文件: {hdf5_file}")
                            continue
                        
                        video_files.append({
                            'hdf5_path': str(hdf5_file),
                            'mp4_path': str(mp4_file),
                            'task_name': task_dir.name,
                            'episode_name': episode_name,
                            'camera_name': camera_name
                        })
                    else:
                        print(f"警告: 未找到对应的视频文件 {mp4_file}")
    
    print(f"总共找到 {len(video_files)} 个待处理的视频文件")
    return video_files


def check_image_data_exists(hdf5_path: str) -> bool:
    """
    检查HDF5文件中是否已存在图像数据
    
    Args:
        hdf5_path: HDF5文件路径
        
    Returns:
        是否已存在图像数据
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if DATASET_NAME in f:
                print(f"HDF5文件中已存在图像数据: {DATASET_NAME}")
                return True
        return False
    except Exception as e:
        print(f"检查HDF5文件失败 {hdf5_path}: {e}")
        return False


def worker_function(task_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    工作函数：每个并行进程执行的核心逻辑
    
    Args:
        task_info: 包含任务信息的字典
        
    Returns:
        处理结果字典，失败时返回None
    """
    mp4_path = task_info['mp4_path']
    hdf5_path = task_info['hdf5_path']
    
    # 初始化TurboJPEG实例（每个进程独立初始化）
    try:
        jpeg_encoder = TurboJPEG()
    except Exception as e:
        print(f"初始化TurboJPEG失败: {e}")
        return None
    
    try:
        # 1. 获取HDF5文件中的数据长度
        with h5py.File(hdf5_path, 'r') as f:
            # 从动作数据获取序列长度
            action_data = f['joint_action/left_arm'][:]
            target_length = len(action_data)
            print(f"目标序列长度: {target_length}")
        
        # 2. 视频解码
        vr = decord.VideoReader(mp4_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            print(f"警告: {mp4_path} 视频帧数为0")
            return None
        
        print(f"视频信息: {mp4_path} ({total_frames}帧)")
        
        # 3. 帧率调整和采样
        if total_frames != target_length:
            print(f"帧数不匹配: 视频{total_frames}帧，需要{target_length}帧")
            
            # 计算采样间隔
            sample_indices = np.linspace(0, total_frames-1, target_length, dtype=int)
            
            # 批量解码需要的帧
            frames_list = vr.get_batch(sample_indices)
        else:
            # 帧数匹配，直接解码所有帧
            frames_list = vr.get_batch(list(range(total_frames)))
        
        # 转换为numpy数组
        if hasattr(frames_list, 'asnumpy'):
            frames_array = frames_list.asnumpy()
        else:
            frames_array = np.array(frames_list)
        
        if frames_array is None or len(frames_array) == 0:
            print(f"警告: {mp4_path} 解码失败")
            return None
        
        # 4. 图像预处理（调整到期望的尺寸）
        processed_frames = []
        print(f"正在处理图像: {mp4_path}")
        
        with tqdm(total=len(frames_array), desc=f"处理 {Path(mp4_path).name}", 
                 leave=False, position=mp.current_process().pid % 10) as pbar:
            
            for frame_rgb in frames_array:
                # 保留原始分辨率，让视觉编码器自动处理
                # 如果需要强制调整尺寸，取消下面的注释
                # if frame_rgb.shape[1] != 320 or frame_rgb.shape[0] != 240:
                #     frame_rgb = cv2.resize(frame_rgb, (320, 240))
                
                processed_frames.append(frame_rgb)
                pbar.update(1)
        
        # 5. JPEG编码
        encoded_frames = []
        max_len = 0
        
        print(f"正在编码JPEG: {mp4_path}")
        
        with tqdm(total=len(processed_frames), desc=f"编码 {Path(mp4_path).name}", 
                 leave=False, position=mp.current_process().pid % 10) as pbar:
            
            for frame_rgb in processed_frames:
                # PyTurboJPEG默认处理BGR格式，需要转换
                frame_bgr = frame_rgb[:, :, ::-1]  # RGB转BGR
                
                # 编码为JPEG字节流
                jpeg_bytes = jpeg_encoder.encode(frame_bgr, quality=JPEG_QUALITY)
                
                encoded_frames.append(jpeg_bytes)
                max_len = max(max_len, len(jpeg_bytes))
                pbar.update(1)
        
        # 6. 数据填充到固定长度
        padded_frames = []
        for jpeg_bytes in encoded_frames:
            # 填充到相同长度
            padded_jpeg = jpeg_bytes.ljust(max_len, b'\0')
            padded_frames.append(padded_jpeg)
        
        # 7. 返回结果
        return {
            "status": "success",
            "mp4_path": mp4_path,
            "hdf5_path": hdf5_path,
            "padded_jpeg_frames": padded_frames,
            "dtype_str": f"|S{max_len}",
            "total_frames": len(processed_frames),
            "max_len": max_len,
            "original_size_mb": os.path.getsize(mp4_path) / (1024 * 1024)
        }
        
    except Exception as e:
        print(f"处理视频失败 {mp4_path}: {e}")
        traceback.print_exc()
        return None


def add_to_hdf5(result: Dict[str, Any]) -> bool:
    """
    将处理结果添加到现有HDF5文件
    
    Args:
        result: 工作函数返回的处理结果
        
    Returns:
        添加是否成功
    """
    try:
        hdf5_path = result['hdf5_path']
        
        # 在现有HDF5文件中添加数据
        with h5py.File(hdf5_path, 'a') as f:
            # 创建observation/head_camera组（如果不存在）
            if 'observation' not in f:
                obs_group = f.create_group('observation')
            else:
                obs_group = f['observation']
            
            if 'head_camera' not in obs_group:
                camera_group = obs_group.create_group('head_camera')
            else:
                camera_group = obs_group['head_camera']
            
            # 添加图像数据
            jpeg_array = np.array(result['padded_jpeg_frames'], dtype=result['dtype_str'])
            camera_group.create_dataset('rgb', data=jpeg_array)
        
        print(f"✓ 成功添加图像数据到: {hdf5_path}")
        return True
        
    except Exception as e:
        print(f"添加图像数据到HDF5文件失败 {hdf5_path}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='真机视频数据预处理（简化版）')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='输入目录路径 (例如: /path/to/zhenji)')
    parser.add_argument('--camera_name', type=str, default='left_wrist',
                        help='相机名称 (默认: left_wrist)')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='进程数 (默认: CPU核心数)')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help=f'JPEG压缩质量 (1-100, 默认: 95)')
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式：只处理前5个文件')
    
    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    
    # 更新全局JPEG质量设置
    global JPEG_QUALITY
    JPEG_QUALITY = args.jpeg_quality
    
    print(f"输入目录: {args.input_dir}")
    print(f"相机名称: {args.camera_name}")
    print(f"进程数: {args.num_processes}")
    print(f"JPEG质量: {JPEG_QUALITY}")
    print(f"数据集路径: {DATASET_NAME}")
    
    # 步骤1: 任务发现
    video_tasks = scan_video_files(args.input_dir, args.camera_name)
    
    if not video_tasks:
        print("未找到任何待处理的视频文件！")
        return
    
    # 测试模式：只处理前5个文件
    if args.test_mode:
        video_tasks = video_tasks[:5]
        print("测试模式：只处理前5个文件")
    
    print(f"开始处理 {len(video_tasks)} 个视频文件...")
    
    # 统计变量
    successful_files = 0
    failed_files = 0
    total_original_size = 0
    
    # 步骤2: 并行处理
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
                        # 添加到HDF5文件
                        if add_to_hdf5(result):
                            successful_files += 1
                            total_original_size += result['original_size_mb']
                            print(f"✓ 完成: {Path(result['mp4_path']).name} "
                                  f"({result['total_frames']}帧)")
                        else:
                            failed_files += 1
                            print(f"✗ 添加失败: {Path(task['mp4_path']).name}")
                    else:
                        failed_files += 1
                        print(f"✗ 处理失败: {Path(task['mp4_path']).name}")
                    
                except Exception as e:
                    failed_files += 1
                    print(f"✗ 处理异常: {Path(task['mp4_path']).name} - {e}")
                    traceback.print_exc()
                
                pbar.update(1)
    
    # 计算总统计信息
    total_time = time.time() - start_time
    
    # 打印最终统计
    print("\n" + "="*60)
    print("处理完成！")
    print(f"成功处理: {successful_files} 个文件")
    print(f"处理失败: {failed_files} 个文件")
    print(f"总处理时间: {total_time:.1f} 秒")
    print(f"平均速度: {total_time/len(video_tasks):.2f} 秒/文件")
    print(f"原始视频大小: {total_original_size:.1f} MB")
    print(f"输入目录: {args.input_dir}")
    print(f"相机名称: {args.camera_name}")
    print(f"数据集路径: {DATASET_NAME}")
    print("="*60)
    
    # 保存处理报告
    output_dir = Path(args.input_dir)
    report_path = output_dir / 'simple_video_processing_report.json'
    report = {
        'input_directory': str(args.input_dir),
        'camera_name': args.camera_name,
        'num_processes': args.num_processes,
        'jpeg_quality': JPEG_QUALITY,
        'dataset_name': DATASET_NAME,
        'total_files': len(video_tasks),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_time_seconds': total_time,
        'total_original_size_mb': total_original_size,
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"处理报告已保存到: {report_path}")


if __name__ == "__main__":
    main()