"""
EgoDex-Robotwin协同预训练数据集 V2 - 76维版本

支持两种不同格式的数据：
1. EgoDex格式：HDF5(76D动作) + MP4视频 + 每轨迹独立语言嵌入
2. Robotwin格式：HDF5(76D动作+RGB图像) + 每任务共享语言嵌入

统一输出76维动作数据进行协同训练
"""

import json
import os
import random
import numpy as np
import h5py
import torch
from PIL import Image, ImageOps
import cv2
import decord
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from collections import defaultdict
import logging
import pickle
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")


class EgoDexRobotwinCopretrainDatasetV2:
    """EgoDex-Robotwin协同预训练数据集类 V2 - 76维版本"""

    DATASET_NAME = "egodex_robotwin_copretrain_V2"

    # 类级别的缓存，避免重复加载
    _cache_lock = threading.Lock()
    _dataset_cache = {}

    def __init__(self,
                 dataset_dir: str,
                 chunk_size: int = 16,
                 img_history_size: int = 1,
                 upsample_rate: int = 3,
                 skip_frames: int = 1,
                 balance_datasets: bool = False,
                 egodex_weight: float = 0.8,
                 stats_file: str = None,
                 **kwargs):
        """
        初始化协同预训练数据集 V2

        Args:
            dataset_dir: 包含两种数据集的根目录
            chunk_size: 动作序列块大小
            img_history_size: 图像历史帧数
            upsample_rate: 时间上采样率
            skip_frames: 跳帧数
            balance_datasets: 是否平衡两个数据集的采样
            egodex_weight: EgoDex数据采样权重
            stats_file: 76维统计信息文件路径
        """
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_history_size = img_history_size
        self.upsample_rate = upsample_rate
        self.skip_frames = skip_frames
        self.balance_datasets = balance_datasets
        self.egodex_weight = egodex_weight

        # 数据路径配置
        base_dir = os.path.join(dataset_dir, "dataset") if not dataset_dir.endswith("dataset") else dataset_dir
        self.egodex_data_dir = os.path.join(base_dir, "Egodex-full-pretrain")
        self.egodex_video_dir = os.path.join(base_dir, "Egodex-full")
        self.robotwin_data_dir = os.path.join(base_dir, "Robotwin-gen-human-full-62d")
        self.robotwin_lang_dir = os.path.join(base_dir, "Robotwin-gen-human-full-62d_lang_embeddings")

        # 缓存文件路径
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"egodex_robotwin_v2_{hashlib.md5(base_dir.encode()).hexdigest()}.pkl")

        # 加载76维统计信息
        self.stats = self._load_stats(stats_file)

        # 初始化数据列表
        self.egodex_episodes = []
        self.robotwin_episodes = []

        # 预加载语言嵌入
        self.egodex_embeddings = {}
        self.robotwin_embeddings = {}

        # 加载数据（使用缓存）
        self._load_datasets_cached()

        # 创建平衡采样列表
        self._create_balanced_sampling_list()

        logger.info(f"Dataset V2 initialized: {len(self.egodex_episodes)} EgoDex episodes, "
                   f"{len(self.robotwin_episodes)} Robotwin episodes, "
                   f"total {len(self)} samples")

    def _load_stats(self, stats_file: str) -> Dict:
        """加载76维统计信息"""
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        else:
            # 如果找不到指定的统计文件就报错
            raise FileNotFoundError(f"统计文件不存在: {stats_file}\n" +
                                  f"请确保76维统计文件存在，并提供正确的路径。")

    def _load_datasets_cached(self):
        """使用缓存加载两种数据集"""
        # 线程安全检查缓存
        with self._cache_lock:
            if self.cache_file in self._dataset_cache:
                cache_data = self._dataset_cache[self.cache_file]
                logger.info("✅ Loading dataset from memory cache")
                self.egodex_episodes = cache_data['egodex_episodes']
                self.robotwin_episodes = cache_data['robotwin_episodes']
                return

            # 检查磁盘缓存
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'rb') as f:
                        cache_data = pickle.load(f)

                    cache_time = os.path.getmtime(self.cache_file)
                    current_time = time.time()

                    # 检查缓存是否过期（24小时）
                    if current_time - cache_time < 24 * 3600:
                        logger.info("✅ Loading dataset from disk cache")
                        self.egodex_episodes = cache_data['egodex_episodes']
                        self.robotwin_episodes = cache_data['robotwin_episodes']
                        self._dataset_cache[self.cache_file] = cache_data
                        return
                    else:
                        logger.info("⚠️  Cache expired, rebuilding...")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}")

        # 缓存未命中，重新加载数据
        logger.info("🔄 Building dataset cache...")
        self._load_datasets()

        # 保存到缓存
        cache_data = {
            'egodex_episodes': self.egodex_episodes,
            'robotwin_episodes': self.robotwin_episodes,
            'timestamp': time.time()
        }

        try:
            # 内存缓存
            self._dataset_cache[self.cache_file] = cache_data

            # 磁盘缓存
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"✅ Dataset cached to {self.cache_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save cache: {e}")

    def _load_datasets(self):
        """加载两种数据集"""
        # 加载EgoDex数据
        self._load_egodex_data()

        # 加载Robotwin数据
        self._load_robotwin_data()

    def _load_egodex_data(self):
        """加载EgoDex格式数据（优化版本）"""
        if not os.path.exists(self.egodex_data_dir):
            logger.warning(f"EgoDex data directory not found: {self.egodex_data_dir}")
            return

        if not os.path.exists(self.egodex_video_dir):
            logger.warning(f"EgoDex video directory not found: {self.egodex_video_dir}")
            return

        logger.info("⚡ Loading EgoDex dataset V2 (76D) with optimizations...")

        # 快速收集所有HDF5文件路径
        hdf5_candidates = []

        # 遍历所有part目录
        for part_name in os.listdir(self.egodex_data_dir):
            part_data_path = os.path.join(self.egodex_data_dir, part_name)
            part_video_path = os.path.join(self.egodex_video_dir, part_name)

            if not (os.path.isdir(part_data_path) and os.path.isdir(part_video_path)):
                continue

            for task_name in os.listdir(part_data_path):
                task_data_path = os.path.join(part_data_path, task_name)
                task_video_path = os.path.join(part_video_path, task_name)

                if not os.path.isdir(task_data_path):
                    continue

                # 快速收集HDF5文件
                for hdf5_file in os.listdir(task_data_path):
                    if hdf5_file.endswith('.hdf5'):
                        episode_id = hdf5_file.replace('.hdf5', '')

                        hdf5_path = os.path.join(task_data_path, hdf5_file)
                        pt_path = os.path.join(task_data_path, f"{episode_id}.pt")
                        mp4_path = os.path.join(task_video_path, f"{episode_id}.mp4")

                        # 快速检查必要文件是否存在（避免后续重复检查）
                        if os.path.exists(hdf5_path) and os.path.exists(pt_path):
                            hdf5_candidates.append({
                                'hdf5_path': hdf5_path,
                                'pt_path': pt_path,
                                'mp4_path': mp4_path,
                                'task': task_name,
                                'episode_id': episode_id,
                                'part': part_name
                            })

        logger.info(f"📁 Found {len(hdf5_candidates)} HDF5 candidate files, validating...")

        # 使用并行处理验证HDF5文件（减少IO瓶颈）
        valid_episodes = self._validate_hdf5_files_parallel(hdf5_candidates)

        # 添加到episode列表
        for episode_info in valid_episodes:
            self.egodex_episodes.append({
                'dataset_type': 'egodex',
                **episode_info,
                'embedding': None,  # 延迟加载，按需获取
            })

        logger.info(f"✅ Loaded {len(self.egodex_episodes)} valid EgoDex episodes (76D)")

    def _validate_hdf5_files_parallel(self, candidates: List[Dict], max_workers: int = 8) -> List[Dict]:
        """并行验证HDF5文件是否包含76维数据"""
        valid_episodes = []

        def validate_single_file(candidate):
            hdf5_path = candidate['hdf5_path']
            try:
                # 快速检查是否包含76维数据
                with h5py.File(hdf5_path, 'r') as f:
                    if 'actions_76d' not in f:
                        return None

                    # 检查视频文件是否存在（可选，不影响episode创建）
                    has_video = os.path.exists(candidate['mp4_path'])

                    return {
                        **candidate,
                        'has_video': has_video
                    }
            except Exception as e:
                logger.debug(f"Failed to validate {hdf5_path}: {e}")
                return None

        # 使用线程池并行验证
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_candidate = {
                executor.submit(validate_single_file, candidate): candidate
                for candidate in candidates
            }

            # 收集结果（带进度条）
            valid_count = 0
            for future in tqdm(as_completed(future_to_candidate),
                             total=len(candidates),
                             desc="Validating HDF5 files"):
                result = future.result()
                if result is not None:
                    valid_episodes.append(result)
                    valid_count += 1

        logger.info(f"🎯 Validation complete: {valid_count}/{len(candidates)} files valid")
        return valid_episodes

    def _load_robotwin_data(self):
        """加载Robotwin格式数据"""
        if not os.path.exists(self.robotwin_data_dir):
            logger.warning(f"Robotwin data directory not found: {self.robotwin_data_dir}")
            return

        logger.info("Loading Robotwin dataset V2 (76D)...")

        # 预加载所有任务的语言嵌入
        if os.path.exists(self.robotwin_lang_dir):
            for pt_file in os.listdir(self.robotwin_lang_dir):
                if pt_file.endswith('.pt'):
                    task_name = pt_file.replace('.pt', '')
                    pt_path = os.path.join(self.robotwin_lang_dir, pt_file)
                    try:
                        embed_data = torch.load(pt_path, map_location='cpu')
                        if 'embeddings' in embed_data:
                            embeddings = embed_data['embeddings']
                            if embeddings.dim() == 3:
                                embeddings = embeddings.squeeze(0)  # 移除批次维度
                            self.robotwin_embeddings[task_name] = embeddings
                    except Exception as e:
                        logger.warning(f"Failed to load embedding for {task_name}: {e}")

        # 遍历所有任务目录
        for task_name in os.listdir(self.robotwin_data_dir):
            task_path = os.path.join(self.robotwin_data_dir, task_name)
            if not os.path.isdir(task_path):
                continue

            # 遍历子目录查找HDF5文件
            for root, dirs, files in os.walk(task_path):
                for hdf5_file in files:
                    if hdf5_file.endswith('.hdf5'):
                        hdf5_path = os.path.join(root, hdf5_file)

                        # 检查是否包含76维数据
                        try:
                            with h5py.File(hdf5_path, 'r') as f:
                                if 'virtual_human_real_robot_76d' in f:
                                    # 检查是否有RGB数据
                                    has_rgb = 'observation/head_camera/rgb' in f

                                    # 验证RGB数据是否有效
                                    rgb_valid = True
                                    if has_rgb:
                                        rgb_data = f['observation/head_camera/rgb']
                                        # 检查RGB数据是否包含有效内容
                                        if len(rgb_data) > 0:
                                            # 检查第一个数据点是否为None
                                            first_frame = rgb_data[0]
                                            if first_frame is None:
                                                rgb_valid = False
                                                logger.warning(f"RGB data contains None values: {hdf5_path}")

                                    # 只有数据有效时才添加episode
                                    if rgb_valid:  # 只有RGB数据有效才添加
                                        self.robotwin_episodes.append({
                                            'dataset_type': 'robotwin',
                                            'hdf5_path': hdf5_path,
                                            'task': task_name,
                                            'episode_id': hdf5_file.replace('.hdf5', ''),
                                            'has_rgb': True,
                                            'embedding': self.robotwin_embeddings.get(task_name, None)
                                        })
                                    else:
                                        # 没有RGB数据，记录但不添加
                                        logger.debug(f"No RGB data in {hdf5_path}, skipping episode")
                        except Exception as e:
                            logger.warning(f"Failed to load {hdf5_path}: {e}")

        logger.info(f"Loaded {len(self.robotwin_episodes)} Robotwin episodes (76D)")

    def _create_balanced_sampling_list(self):
        """创建平衡采样列表"""
        if not self.balance_datasets:
            # 简单拼接
            self.sampling_list = (
                [(ep, 'egodex') for ep in self.egodex_episodes] +
                [(ep, 'robotwin') for ep in self.robotwin_episodes]
            )
        else:
            # 平衡采样
            egodex_count = len(self.egodex_episodes)
            robotwin_count = len(self.robotwin_episodes)

            if egodex_count == 0 or robotwin_count == 0:
                # 如果其中一个数据集为空，使用另一个
                if egodex_count > 0:
                    self.sampling_list = [(ep, 'egodex') for ep in self.egodex_episodes]
                else:
                    self.sampling_list = [(ep, 'robotwin') for ep in self.robotwin_episodes]
            else:
                # 计算采样权重
                egodex_samples = max(1, int(egodex_count * self.egodex_weight))
                robotwin_samples = max(1, int(robotwin_count * (1 - self.egodex_weight)))

                # 创建平衡采样列表
                self.sampling_list = []

                # 添加EgoDex样本
                egodex_samples = min(egodex_samples, egodex_count)
                selected_egodex = random.sample(self.egodex_episodes, egodex_samples)
                self.sampling_list.extend([(ep, 'egodex') for ep in selected_egodex])

                # 添加Robotwin样本
                robotwin_samples = min(robotwin_samples, robotwin_count)
                selected_robotwin = random.sample(self.robotwin_episodes, robotwin_samples)
                self.sampling_list.extend([(ep, 'robotwin') for ep in selected_robotwin])

        logger.info(f"Created sampling list with {len(self.sampling_list)} samples")

    def __len__(self):
        """返回数据集大小"""
        return len(self.sampling_list)

    def _load_egodex_sample(self, episode_info: Dict, index: int) -> Dict:
        """加载EgoDex格式样本 (76D)"""
        hdf5_path = episode_info['hdf5_path']
        mp4_path = episode_info['mp4_path']

        # 加载76维动作数据
        try:
            with h5py.File(hdf5_path, 'r') as f:
                actions = f['actions_76d'][:]
        except Exception as e:
            logger.error(f"Failed to load actions from {hdf5_path}: {e}")
            return None

        # 计算有效索引
        total_frames = len(actions)
        max_start = max(0, total_frames - self.chunk_size * self.upsample_rate)
        if index >= max_start:
            index = random.randint(0, max_start)

        # 加载当前动作
        current_action = actions[index:index+1]

        # 加载动作序列
        action_indices = [
            min(index + i * self.upsample_rate, total_frames - 1)
            for i in range(self.chunk_size)
        ]
        action_chunk = actions[action_indices]

        # 归一化76维动作 - 使用原始项目的归一化方式（归一化到[-1, 1]）
        if self.stats is not None and 'egodex_76d' in self.stats:
            action_min = np.array(self.stats['egodex_76d']['min'])
            action_max = np.array(self.stats['egodex_76d']['max'])

            # 计算范围，避免除零错误
            action_range = action_max - action_min
            action_range = np.where(action_range < 1e-8, 1.0, action_range)  # 对范围过小的维度，设置范围为1

            # 按照原始项目的方式归一化到[-1, 1]范围
            action_chunk = (action_chunk - action_min) / action_range * 2 - 1
            current_action = (current_action - action_min) / action_range * 2 - 1

            # 裁剪到有效范围
            action_chunk = np.clip(action_chunk, -1, 1)
            current_action = np.clip(current_action, -1, 1)

            # 检查是否包含NaN或Inf
            if np.any(np.isnan(action_chunk)) or np.any(np.isnan(current_action)):
                print(f"⚠️ [EgoDex 76D] NaN detected after normalization!")
                print(f"   - action_range min/max: {action_range.min():.10f} / {action_range.max():.10f}")
                print(f"   - Zero-range dimensions: {np.where(action_range == 1.0)[0]}")
                raise ValueError("NaN detected in EgoDex 76D action normalization")

            # 打印归一化应用信息（仅在第一次调用时打印）
            if not hasattr(self, '_printed_egodex_norm_info'):
                zero_range_dims = np.where(action_range == 1.0)[0]
                print(f"✅ [EgoDex 76D] Action normalization APPLIED:")
                print(f"   - Dataset: egodex_76d")
                print(f"   - Original range: min=[{action_min.min():.6f}, {action_min.max():.6f}], max=[{action_max.min():.6f}, {action_max.max():.6f}]")
                print(f"   - Normalized to: [-1.0, 1.0] range")
                print(f"   - Formula: ((action - min) / (max - min)) * 2 - 1")
                print(f"   - Zero-range dimensions fixed: {zero_range_dims.tolist() if len(zero_range_dims) > 0 else 'None'}")
                print(f"   - Clipping: np.clip(value, -1, 1)")
                self._printed_egodex_norm_info = True

        # 加载图像数据
        if episode_info['has_video']:
            image_frames = self._load_egodex_images(mp4_path, index)
            image_mask = np.ones(self.img_history_size, dtype=bool)
        else:
            # 如果没有视频，使用零填充
            image_frames = np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)
            image_mask = np.zeros(self.img_history_size, dtype=bool)

        
        # 🚀 延迟加载语言嵌入
        if episode_info['pt_path'] and os.path.exists(episode_info['pt_path']):
            try:
                embed_data = torch.load(episode_info['pt_path'], map_location='cpu')
                if 'embeddings' in embed_data:
                    instruction = embed_data['embeddings']
                else:
                    instruction = episode_info['task']
            except Exception as e:
                logger.warning(f"Failed to load embedding from {episode_info['pt_path']}: {e}")
                instruction = episode_info['task']
        else:
            instruction = episode_info['task']

        return {
            'states': current_action,
            'actions': action_chunk,
            'action_norm': np.ones_like(action_chunk),
            'current_images': [image_frames],
            'current_images_mask': [image_mask],
            'instruction': instruction,  # 可以是张量或字符串
            'dataset_name': self.DATASET_NAME,
            'task': episode_info['task'],
            'dataset_type': 'egodex',
            'file_info': episode_info
        }

    def _load_robotwin_sample(self, episode_info: Dict, index: int) -> Dict:
        """加载Robotwin格式样本 (76D)"""
        hdf5_path = episode_info['hdf5_path']

        # 加载76维动作和图像数据
        try:
            with h5py.File(hdf5_path, 'r') as f:
                actions = f['virtual_human_real_robot_76d'][:]

                # 计算有效索引
                total_frames = len(actions)
                max_start = max(0, total_frames - self.chunk_size * self.upsample_rate)
                if index >= max_start:
                    index = random.randint(0, max_start)

                # 加载当前动作
                current_action = actions[index:index+1]

                # 加载动作序列
                action_indices = [
                    min(index + i * self.upsample_rate, total_frames - 1)
                    for i in range(self.chunk_size)
                ]
                action_chunk = actions[action_indices]

                # 归一化76维动作 - 使用原始项目的归一化方式（归一化到[-1, 1]）
                if self.stats is not None and 'robotwin_76d' in self.stats:
                    action_min = np.array(self.stats['robotwin_76d']['min'])
                    action_max = np.array(self.stats['robotwin_76d']['max'])

                    # 计算范围，避免除零错误
                    action_range = action_max - action_min
                    action_range = np.where(action_range < 1e-8, 1.0, action_range)  # 对范围过小的维度，设置范围为1

                    # 按照原始项目的方式归一化到[-1, 1]范围
                    action_chunk = (action_chunk - action_min) / action_range * 2 - 1
                    current_action = (current_action - action_min) / action_range * 2 - 1

                    # 裁剪到有效范围
                    action_chunk = np.clip(action_chunk, -1, 1)
                    current_action = np.clip(current_action, -1, 1)

                    # 检查是否包含NaN或Inf
                    if np.any(np.isnan(action_chunk)) or np.any(np.isnan(current_action)):
                        print(f"⚠️ [Robotwin 76D] NaN detected after normalization!")
                        print(f"   - action_range min/max: {action_range.min():.10f} / {action_range.max():.10f}")
                        print(f"   - Zero-range dimensions: {np.where(action_range == 1.0)[0]}")
                        raise ValueError("NaN detected in Robotwin 76D action normalization")

                    # 打印归一化应用信息（仅在第一次调用时打印）
                    if not hasattr(self, '_printed_robotwin_norm_info'):
                        zero_range_dims = np.where(action_range == 1.0)[0]
                        print(f"✅ [Robotwin 76D] Action normalization APPLIED:")
                        print(f"   - Dataset: robotwin_76d")
                        print(f"   - Original range: min=[{action_min.min():.6f}, {action_min.max():.6f}], max=[{action_max.min():.6f}, {action_max.max():.6f}]")
                        print(f"   - Normalized to: [-1.0, 1.0] range")
                        print(f"   - Formula: ((action - min) / (max - min)) * 2 - 1")
                        print(f"   - Zero-range dimensions fixed: {zero_range_dims.tolist() if len(zero_range_dims) > 0 else 'None'}")
                        print(f"   - Clipping: np.clip(value, -1, 1)")
                        self._printed_robotwin_norm_info = True

                # 加载图像数据
                if episode_info['has_rgb']:
                    image_frames = self._load_robotwin_images(f, index)
                    image_mask = np.ones(self.img_history_size, dtype=bool)
                else:
                    # 如果没有RGB数据，使用零填充
                    image_frames = np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)
                    image_mask = np.zeros(self.img_history_size, dtype=bool)

                
        except Exception as e:
            logger.error(f"Failed to load Robotwin sample from {hdf5_path}: {e}")
            # 返回默认样本而不是None，避免训练中断
            return self._get_default_robotwin_sample()

        # 处理语言嵌入
        if episode_info['embedding'] is not None:
            instruction = episode_info['embedding']
        else:
            instruction = episode_info['task']

        return {
            'states': current_action,
            'actions': action_chunk,
            'action_norm': np.ones_like(action_chunk),
            'current_images': [image_frames],
            'current_images_mask': [image_mask],
            'instruction': instruction,  # 可以是张量或字符串
            'dataset_name': self.DATASET_NAME,
            'task': episode_info['task'],
            'dataset_type': 'robotwin',
            'file_info': episode_info
        }

    def _load_egodex_images(self, mp4_path: str, index: int) -> np.ndarray:
        """加载EgoDex图像帧"""
        try:
            vr = decord.VideoReader(mp4_path)
            total_frames = len(vr)

            # 计算历史帧索引
            frame_indices = []
            for i in range(self.img_history_size):
                frame_idx = max(0, index - (self.img_history_size - 1 - i) * self.upsample_rate)
                frame_idx = min(frame_idx, total_frames - 1)
                frame_indices.append(frame_idx)

            # 获取帧
            frames = vr.get_batch(frame_indices).numpy()  # (T, H, W, C)

            # 确保形状正确
            if frames.shape[-1] != 3:
                frames = np.transpose(frames, (0, 2, 3, 1))

            # 统一尺寸为320x240
            resized_frames = []
            for frame in frames:
                resized_frame = cv2.resize(frame, (320, 240))
                resized_frames.append(resized_frame)

            return np.stack(resized_frames, axis=0).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Failed to load images from {mp4_path}: {e}")
            # 返回零填充
            return np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)

    def _load_robotwin_images(self, hdf5_file: h5py.File, index: int) -> np.ndarray:
        """加载Robotwin图像帧"""
        try:
            rgb_data = hdf5_file['observation/head_camera/rgb']
            total_frames = len(rgb_data)

            # 计算历史帧索引
            frame_indices = []
            for i in range(self.img_history_size):
                frame_idx = max(0, index - (self.img_history_size - 1 - i) * self.upsample_rate)
                frame_idx = min(frame_idx, total_frames - 1)
                frame_indices.append(frame_idx)

            # 获取帧并解码
            frames = []
            for frame_idx in frame_indices:
                frame_data = rgb_data[frame_idx]

                if isinstance(frame_data, np.ndarray) and frame_data.ndim == 3:
                    # 已经是图像数组
                    frame = frame_data
                else:
                    # 直接用np.frombuffer处理字符串数据
                    try:
                        # 直接用np.frombuffer处理字符串数据
                        nparr = np.frombuffer(frame_data, np.uint8)
                        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if bgr_img is not None:
                            # OpenCV使用BGR格式，转换为RGB
                            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                            # 确保尺寸为320x240
                            if rgb_img.shape[:2] != (240, 320):
                                rgb_img = cv2.resize(rgb_img, (320, 240))
                            frame = rgb_img
                        else:
                            # 解码失败，使用零填充
                            frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    except Exception as e:
                        logger.warning(f"Robotwin image decode failed: {e}")
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frames.append(frame)

            image_frames = np.stack(frames, axis=0)

            # 统一尺寸为320x240
            resized_frames = []
            for frame in image_frames:
                resized_frame = cv2.resize(frame, (320, 240))
                resized_frames.append(resized_frame)

            return np.stack(resized_frames, axis=0).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Failed to load Robotwin images: {e}")
            return np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)

    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本"""
        episode_info, dataset_type = self.sampling_list[idx]

        # 生成随机索引
        index = random.randint(0, 1000)  # 实际加载时会调整到有效范围

        # 根据数据类型加载样本
        if dataset_type == 'egodex':
            sample = self._load_egodex_sample(episode_info, index)
        else:  # robotwin
            sample = self._load_robotwin_sample(episode_info, index)

        # 如果加载失败，返回默认样本
        if sample is None:
            return self._get_default_sample()

        return sample

    def get_item(self, idx: int = None) -> Dict:
        """获取数据样本 - 为了兼容VLAConsumerDataset的调用"""
        if idx is None:
            # 如果没有提供索引，随机选择一个
            idx = random.randint(0, len(self) - 1)
        return self.__getitem__(idx)

    def _get_default_sample(self) -> Dict:
        """返回默认样本（用于错误处理）"""
        return {
            'states': np.zeros((1, 76), dtype=np.float32),
            'actions': np.zeros((self.chunk_size, 76), dtype=np.float32),
            'action_norm': np.ones((self.chunk_size, 76), dtype=np.float32),
            'current_images': [np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)],
            'current_images_mask': [np.zeros(self.img_history_size, dtype=bool)],
            'instruction': 'default task',
            'dataset_name': self.DATASET_NAME,
            'task': 'default',
            'dataset_type': 'default',
            'file_info': {}
        }

    def _get_default_robotwin_sample(self) -> Dict:
        """返回Robotwin默认样本（用于错误处理）"""
        return {
            'states': np.zeros((1, 76), dtype=np.float32),
            'actions': np.zeros((self.chunk_size, 76), dtype=np.float32),
            'action_norm': np.ones((self.chunk_size, 76), dtype=np.float32),
            'current_images': [np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)],
            'current_images_mask': [np.zeros(self.img_history_size, dtype=bool)],
            'instruction': 'default robotwin task',
            'dataset_name': self.DATASET_NAME,
            'task': 'default_robotwin',
            'dataset_type': 'robotwin',
            'file_info': {}
        }