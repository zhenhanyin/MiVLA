"""
EgoDex-Robotwin协同预训练数据集

支持两种不同格式的数据：
1. EgoDex格式：HDF5(62D动作) + MP4视频 + 每轨迹独立语言嵌入
2. Robotwin格式：HDF5(62D动作+RGB图像) + 每任务共享语言嵌入

统一输出62维动作数据进行协同训练
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")


class EgoDexRobotwinCopretrainDataset:
    """EgoDex-Robotwin协同预训练数据集类"""

    DATASET_NAME = "egodex_robotwin_copretrain"

    def __init__(self,
                 dataset_dir: str,
                 chunk_size: int = 16,
                 img_history_size: int = 1,
                 upsample_rate: int = 3,
                 skip_frames: int = 1,
                 balance_datasets: bool = False,
                 egodex_weight: float = 0.8,
                 stats_file: str = None,
                 image_aug: bool = False,
                 image_aug_threshold: float = 0.1,
                 **kwargs):
        """
        初始化协同预训练数据集

        Args:
            dataset_dir: 包含两种数据集的根目录
            chunk_size: 动作序列块大小
            img_history_size: 图像历史帧数
            upsample_rate: 时间上采样率
            skip_frames: 跳帧数
            balance_datasets: 是否平衡两个数据集的采样
            egodex_weight: EgoDex数据采样权重
            stats_file: 统计信息文件路径
            image_aug: 是否使用图像增强
            image_aug_threshold: 图像损坏阈值
        """
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_history_size = img_history_size
        self.upsample_rate = upsample_rate
        self.skip_frames = skip_frames
        self.balance_datasets = balance_datasets
        self.egodex_weight = egodex_weight
        self.image_aug = image_aug
        self.image_aug_threshold = image_aug_threshold

        # 数据路径配置
        base_dir = os.path.join(dataset_dir, "dataset") if not dataset_dir.endswith("dataset") else dataset_dir
        self.egodex_data_dir = os.path.join(base_dir, "Egodex-full-pretrain")
        self.egodex_video_dir = os.path.join(base_dir, "Egodex-full")
        self.robotwin_data_dir = os.path.join(base_dir, "Robotwin-gen-human-full-62d")
        self.robotwin_lang_dir = os.path.join(base_dir, "Robotwin-gen-human-full-62d_lang_embeddings")

        # 加载统计信息
        self.stats = self._load_stats(stats_file)

        # 初始化数据列表
        self.egodex_episodes = []
        self.robotwin_episodes = []

        # 预加载语言嵌入
        self.egodex_embeddings = {}
        self.robotwin_embeddings = {}

        # 加载数据
        self._load_datasets()

        # 创建平衡采样列表
        self._create_balanced_sampling_list()

        logger.info(f"Dataset initialized: {len(self.egodex_episodes)} EgoDex episodes, "
                   f"{len(self.robotwin_episodes)} Robotwin episodes, "
                   f"total {len(self)} samples")

    def _load_stats(self, stats_file: str) -> Dict:
        """加载统计信息"""
        if stats_file and os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        else:
            # 如果找不到指定的统计文件就报错
            raise FileNotFoundError(f"统计文件不存在: {stats_file}\n" +
                                  f"请确保统计文件存在，并提供正确的路径。")

    def _load_datasets(self):
        """加载两种数据集"""
        # 加载EgoDex数据
        self._load_egodex_data()

        # 加载Robotwin数据
        self._load_robotwin_data()

    def _load_egodex_data(self):
        """加载EgoDex格式数据"""
        if not os.path.exists(self.egodex_data_dir):
            logger.warning(f"EgoDex data directory not found: {self.egodex_data_dir}")
            return

        if not os.path.exists(self.egodex_video_dir):
            logger.warning(f"EgoDex video directory not found: {self.egodex_video_dir}")
            return

        logger.info("Loading EgoDex dataset...")

        # 遍历所有part目录
        for part_name in os.listdir(self.egodex_data_dir):
            part_data_path = os.path.join(self.egodex_data_dir, part_name)
            part_video_path = os.path.join(self.egodex_video_dir, part_name)

            if not os.path.isdir(part_data_path) or not os.path.isdir(part_video_path):
                continue

            for task_name in os.listdir(part_data_path):
                task_data_path = os.path.join(part_data_path, task_name)
                task_video_path = os.path.join(part_video_path, task_name)

                if not os.path.isdir(task_data_path):
                    continue

                # 查找HDF5和对应的PT、MP4文件
                hdf5_files = [f for f in os.listdir(task_data_path) if f.endswith('.hdf5')]

                for hdf5_file in hdf5_files:
                    episode_id = hdf5_file.replace('.hdf5', '')

                    # 构建文件路径
                    hdf5_path = os.path.join(task_data_path, hdf5_file)
                    pt_path = os.path.join(task_data_path, f"{episode_id}.pt")
                    mp4_path = os.path.join(task_video_path, f"{episode_id}.mp4")

                    # 检查所有必要文件是否存在
                    if os.path.exists(hdf5_path) and os.path.exists(pt_path):
                        # 🚀 延迟加载策略：不预加载语言嵌入，只存储路径
                        self.egodex_episodes.append({
                            'dataset_type': 'egodex',
                            'hdf5_path': hdf5_path,
                            'mp4_path': mp4_path,
                            'pt_path': pt_path,
                            'task': task_name,
                            'episode_id': episode_id,
                            'part': part_name,
                            'embedding': None,  # 延迟加载，按需获取
                            'has_video': os.path.exists(mp4_path)
                        })

        logger.info(f"Loaded {len(self.egodex_episodes)} EgoDex episodes")

    def _load_robotwin_data(self):
        """加载Robotwin格式数据"""
        if not os.path.exists(self.robotwin_data_dir):
            logger.warning(f"Robotwin data directory not found: {self.robotwin_data_dir}")
            return

        logger.info("Loading Robotwin dataset...")

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

                        # 检查是否包含62维数据
                        try:
                            with h5py.File(hdf5_path, 'r') as f:
                                if 'virtual_human_real_robot_62d' in f:
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

        logger.info(f"Loaded {len(self.robotwin_episodes)} Robotwin episodes")

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
        """加载EgoDex格式样本"""
        hdf5_path = episode_info['hdf5_path']
        mp4_path = episode_info['mp4_path']

        # 加载动作数据
        try:
            with h5py.File(hdf5_path, 'r') as f:
                actions = f['actions_62d'][:]
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

        # 归一化动作
        if self.stats is not None and 'egodex_62d' in self.stats:
            action_min = np.array(self.stats['egodex_62d']['min'])
            action_max = np.array(self.stats['egodex_62d']['max'])
            action_chunk = (action_chunk - action_min) / (action_max - action_min + 1e-8)
            current_action = (current_action - action_min) / (action_max - action_min + 1e-8)

        # 加载图像数据
        if episode_info['has_video']:
            image_frames = self._load_egodex_images(mp4_path, index)
            image_mask = np.ones(self.img_history_size, dtype=bool)
        else:
            # 如果没有视频，使用零填充
            image_frames = np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)
            image_mask = np.zeros(self.img_history_size, dtype=bool)

        # 图像增强
        if self.image_aug and random.random() < self.image_aug_threshold:
            image_frames = self._apply_image_augmentation(image_frames)

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
        """加载Robotwin格式样本"""
        hdf5_path = episode_info['hdf5_path']

        # 加载动作和图像数据
        try:
            with h5py.File(hdf5_path, 'r') as f:
                actions = f['virtual_human_real_robot_62d'][:]

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

                # 归一化动作
                if self.stats is not None and 'robotwin_62d' in self.stats:
                    action_min = np.array(self.stats['robotwin_62d']['min'])
                    action_max = np.array(self.stats['robotwin_62d']['max'])
                    action_chunk = (action_chunk - action_min) / (action_max - action_min + 1e-8)
                    current_action = (current_action - action_min) / (action_max - action_min + 1e-8)

                # 加载图像数据
                if episode_info['has_rgb']:
                    image_frames = self._load_robotwin_images(f, index)
                    image_mask = np.ones(self.img_history_size, dtype=bool)
                else:
                    # 如果没有RGB数据，使用零填充
                    image_frames = np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)
                    image_mask = np.zeros(self.img_history_size, dtype=bool)

                # 图像增强
                if self.image_aug and random.random() < self.image_aug_threshold:
                    image_frames = self._apply_image_augmentation(image_frames)

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
                    # 直接使用 robotwin_agilex_dataset.py 的解码方式
                    # 参考 decode_image_with_opencv 函数 (第269行)
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

    def _decode_image(self, image_data) -> np.ndarray:
        """解码图像数据（兼容多种数据类型）"""
        try:
            # 参考 robotwin_agilex_dataset.py 的解码方式
            if isinstance(image_data, bytes):
                # 直接字节数据
                nparr = np.frombuffer(image_data, np.uint8)
            elif isinstance(image_data, (str, bytes, np.str_)):
                # 字符串类型（包括numpy字符串）
                # 尝试转换为字节数据
                try:
                    image_data = image_data.encode('latin1') if isinstance(image_data, str) else bytes(image_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                except:
                    # 如果编码失败，使用ASCII编码
                    if isinstance(image_data, str):
                        image_data = image_data.encode('ascii', errors='ignore')
                        nparr = np.frombuffer(image_data, np.uint8)
                    else:
                        nparr = np.frombuffer(image_data.astype(bytes), np.uint8)
            else:
                # numpy数组类型或其他
                nparr = image_data.astype(np.uint8) if hasattr(image_data, 'astype') else np.array(image_data, dtype=np.uint8)

            # 使用OpenCV解码
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_img is not None:
                # OpenCV成功解码
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                return rgb_img
            else:
                # 如果OpenCV失败，尝试PIL
                from io import BytesIO
                image = Image.open(BytesIO(nparr.tobytes()))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)

        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8)
        """解码二进制图像数据"""
        try:
            # 使用OpenCV解码
            frame_arr = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            else:
                # 如果OpenCV失败，尝试PIL
                from io import BytesIO
                image = Image.open(BytesIO(image_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8)

    def _decode_image_like_robotwin(self, img_data) -> np.ndarray:
        """直接参考 robotwin_agilex_dataset.py decode_image_with_opencv 的解码方式"""
        try:
            # 直接使用 np.frombuffer 处理（包括字符串类型）
            nparr = np.frombuffer(img_data, np.uint8)

            # 使用OpenCV解码
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_img is None:
                logger.warning("OpenCV decoding failed")
                return np.zeros((240, 320, 3), dtype=np.uint8)

            # 转换BGR到RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # 确保尺寸为320x240
            if rgb_img.shape[:2] != (240, 320):
                rgb_img = cv2.resize(rgb_img, (320, 240))

            return rgb_img

        except Exception as e:
            logger.warning(f"Failed to decode image (robotwin style): {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8)

    def _apply_image_augmentation(self, image_frames: np.ndarray) -> np.ndarray:
        """应用图像增强（随机损坏）"""
        try:
            # 随机选择损坏类型
            damage_type = random.choice(['noise', 'blur', 'brightness'])

            if damage_type == 'noise':
                # 添加高斯噪声
                noise = np.random.normal(0, 10, image_frames.shape).astype(np.int16)
                image_frames = np.clip(image_frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            elif damage_type == 'blur':
                # 运动模糊
                for i in range(len(image_frames)):
                    if random.random() < 0.3:  # 30%概率模糊
                        kernel_size = random.choice([3, 5])
                        kernel = np.ones((kernel_size, 1), np.float32) / kernel_size
                        image_frames[i] = cv2.filter2D(image_frames[i], -1, kernel)

            elif damage_type == 'brightness':
                # 亮度调整
                brightness_factor = random.uniform(0.7, 1.3)
                image_frames = np.clip(image_frames.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Image augmentation failed: {e}")

        return image_frames

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
            'states': np.zeros((1, 62), dtype=np.float32),
            'actions': np.zeros((self.chunk_size, 62), dtype=np.float32),
            'action_norm': np.ones((self.chunk_size, 62), dtype=np.float32),
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
            'states': np.zeros((1, 62), dtype=np.float32),
            'actions': np.zeros((self.chunk_size, 62), dtype=np.float32),
            'action_norm': np.ones((self.chunk_size, 62), dtype=np.float32),
            'current_images': [np.zeros((self.img_history_size, 240, 320, 3), dtype=np.uint8)],
            'current_images_mask': [np.zeros(self.img_history_size, dtype=bool)],
            'instruction': 'default robotwin task',
            'dataset_name': self.DATASET_NAME,
            'task': 'default_robotwin',
            'dataset_type': 'robotwin',
            'file_info': {}
        }