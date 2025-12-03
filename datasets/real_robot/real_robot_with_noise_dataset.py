import os
import random
import h5py
import numpy as np
import cv2
import json
import time
import glob
from typing import List, Dict, Optional
import sys
import warnings
import traceback
import torch

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*multichannel.*"
)

if not hasattr(np, 'bool'):
    np.bool = bool


class RealRobotWithNoiseDataset:
    """
    真机数据集的62D微调版本：48D噪声 + 14D真实数据
    用于真机环境的62D微调训练，保留预训练中学到的人类先验知识
    """
    def __init__(
        self,
        root_dir: str,
        task_name: str,
        hdf5_folder: str = "data",
        config: dict = None,
        normalize_actions: bool = True,
        stat_path: str = None,
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_corrupt_severity: int = 5,
        upsample_rate: int = 3,
        noise_strategy: str = "gaussian",  # "gaussian", "uniform", "zero"
        noise_scale: float = 1.0,  # 噪声缩放因子
        human_data_stats: dict = None  # 人类数据统计信息，用于生成更真实的噪声
    ):
        """
        初始化带噪声的真机数据集

        Args:
            root_dir: 真机数据根目录
            task_name: 任务名称
            hdf5_folder: HDF5文件夹名称，默认为"data"
            config: 配置字典
            normalize_actions: 是否对动作进行归一化
            stat_path: 统计文件路径
            max_episodes: 最大episode数量
            val: 是否为验证集
            image_corrupt_severity: 图像损坏严重程度
            upsample_rate: 上采样率
            noise_strategy: 噪声生成策略
                - "gaussian": 高斯噪声
                - "uniform": 均匀噪声
                - "zero": 零噪声
            noise_scale: 噪声缩放因子
            human_data_stats: 人类数据统计信息字典，包含mean和std
        """
        self.root_dir = root_dir
        self.task_name = task_name
        self.hdf5_folder = hdf5_folder
        self.config = config
        self.normalize_actions = normalize_actions
        self.max_episodes = max_episodes
        self.val = val
        self.image_corrupt_severity = image_corrupt_severity
        self.upsample_rate = upsample_rate

        # 设置基本参数 - 注意这里是62D！
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = 62  # 62D状态 (48D噪声 + 14D真实)
        self.action_dim = 62  # 62D动作 (48D噪声 + 14D真实)
        self.img_history_size = config['common']['img_history_size']

        # 设置相机参数
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 2:
            self.cameras = ["head_camera", "left_camera"]
        elif self.num_cameras == 3:
            # 根据HDF5文件结构调整摄像头名称
            self.cameras = ["head_camera", "wrist_left_camera", "wrist_right_camera"]
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 2 or 3 cameras supported.")

        # 加载统计信息用于归一化
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'stats.json')

        try:
            with open(stat_path, 'r') as file:
                stat = json.load(file)
            if task_name in stat:
                self.action_min = np.array(stat[task_name]['min'])
                self.action_max = np.array(stat[task_name]['max'])
            elif 'real_robot' in stat:
                self.action_min = np.array(stat['real_robot']['min'])
                self.action_max = np.array(stat['real_robot']['max'])
            else:
                raise ValueError(f"No statistics found for task {task_name}")
        except Exception as e:
            print(f"Warning: Could not load action statistics from {stat_path}: {e}")
            # 使用默认的归一化范围
            self.action_min = np.array([-1.0] * 14)
            self.action_max = np.array([1.0] * 14)

        # 噪声相关参数
        self.noise_strategy = noise_strategy
        self.noise_scale = noise_scale
        self.human_data_stats = human_data_stats or {}

        # 初始化数据结构
        self.episode_files = []
        self.total_episodes = 0

        # 初始化数据集
        self._initialize_dataset()

        # 初始化噪声生成器
        self.noise_generator = self._init_noise_generator()

    def _init_noise_generator(self):
        """初始化噪声生成器"""
        if self.noise_strategy == "gaussian":
            if self.human_data_stats and 'mean' in self.human_data_stats:
                # 使用人类数据统计信息生成更真实的噪声
                human_mean = self.human_data_stats['mean'][:48]  # 前48D
                human_std = self.human_data_stats['std'][:48]

                def generator(batch_size):
                    return torch.randn(batch_size, 48) * human_std + human_mean
            else:
                # 标准高斯噪声
                def generator(batch_size):
                    return torch.randn(batch_size, 48) * self.noise_scale

        elif self.noise_strategy == "uniform":
            def generator(batch_size):
                return (torch.rand(batch_size, 48) - 0.5) * 2 * self.noise_scale

        elif self.noise_strategy == "zero":
            def generator(batch_size):
                return torch.zeros(batch_size, 48)

        else:
            raise ValueError(f"Unknown noise strategy: {self.noise_strategy}")

        return generator

    def _initialize_dataset(self):
        """初始化数据集，扫描HDF5文件"""
        print("Initializing real robot dataset...")

        task_dir = os.path.join(self.root_dir, self.task_name, self.hdf5_folder)
        if not os.path.exists(task_dir):
            print(f"Warning: Task folder {task_dir} does not exist")
            return

        # 查找所有HDF5文件
        hdf5_files = []
        for f in os.listdir(task_dir):
            if f.endswith(".hdf5"):
                hdf5_path = os.path.join(task_dir, f)
                hdf5_files.append(hdf5_path)

        # 排序以确保一致性
        hdf5_files.sort()

        # 限制episode数量
        if self.max_episodes is not None:
            hdf5_files = hdf5_files[:self.max_episodes]

        self.episode_files = hdf5_files
        self.total_episodes = len(hdf5_files)

        print(f"Real robot dataset initialized. Total {self.total_episodes} episodes")

        if self.total_episodes == 0:
            raise ValueError("Error: No HDF5 files found, please check data path")

    def generate_human_noise(self, batch_size=1):
        """
        生成48D人类动作噪声

        Args:
            batch_size: 批次大小

        Returns:
            torch.Tensor: [batch_size, 48] 噪声数据
        """
        return self.noise_generator(batch_size)

    def parse_img_data(self, dataset, idx):
        """
        处理单个摄像头的图像数据

        Args:
            dataset: 包含单个摄像头图像的HDF5数据集
            idx: 当前帧索引

        Returns:
            处理后的图像序列，形状为 [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # 使用标准分辨率
        frames = np.zeros((num_frames, 240, 320, 3), dtype=np.uint8)

        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < len(dataset):
                    img_data = dataset[frame_idx]

                    decoded_img = self.decode_image_with_opencv(img_data)
                    if decoded_img is not None:
                        frames[i] = decoded_img

        except Exception as e:
            print(f"[DEBUG] decode_image_with_opencv error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames])

        return frames

    def decode_image_with_opencv(self, img_data):
        """
        使用OpenCV解码图像数据，保持RGB格式

        Args:
            img_data (bytes): 二进制图像数据

        Returns:
            np.ndarray: 解码后的图像数组，shape=(240, 320, 3)，RGB格式
        """
        try:
            # 使用OpenCV解码
            nparr = np.frombuffer(img_data, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_img is None:
                return None

            # OpenCV默认使用BGR格式，转换为RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # 如果需要，调整大小到标准尺寸 (320x240)
            if rgb_img.shape[:2] != (240, 320):
                rgb_img = cv2.resize(rgb_img, (320, 240))

            return rgb_img

        except Exception as e:
            return None

    def __len__(self):
        """返回数据集的近似长度"""
        return self.total_episodes * 200  # 假设每个HDF5文件有200个样本

    def get_dataset_name(self):
        """返回数据集名称"""
        return "real_robot_with_noise"

    def normalize_action(self, action):
        """
        归一化动作到[-1, 1]范围

        Args:
            action: 原始动作值，可以是单个向量或批量数据

        Returns:
            归一化后的动作值（如果normalize_actions为False，则返回原始值）
        """
        if not self.normalize_actions:  # 检查归一化开关
            return action  # 如果关闭归一化，直接返回原始数据

        action = np.array(action)
        action_range = self.action_max - self.action_min

        # 🚨 关键修复：处理批量数据和防止除零错误
        zero_range_mask = (action_range == 0)
        normalized_action = np.zeros_like(action)

        # 只对非零范围进行归一化
        non_zero_indices = np.where(~zero_range_mask)[0]  # 获取非零维度的索引

        if len(non_zero_indices) > 0:
            # 对于每个非零维度进行归一化
            for idx in non_zero_indices:
                if action.ndim == 1:
                    # 单个向量
                    normalized_action[idx] = (
                        2.0 * (action[idx] - self.action_min[idx]) / action_range[idx] - 1.0
                    )
                else:
                    # 批量数据 (shape: [batch_size, action_dim])
                    normalized_action[:, idx] = (
                        2.0 * (action[:, idx] - self.action_min[idx]) / action_range[idx] - 1.0
                    )

        return normalized_action

    def get_item(self, index=None):
        """
        获取数据项，如果index为None则随机选择

        Args:
            index: 可选，指定的索引。如果为None，则随机选择

        Returns:
            处理后的数据字典，如果失败则返回None
        """
        if not self.episode_files:
            print("Error: No available episodes")
            return None

        # 随机选择一个HDF5文件
        episode_file = random.choice(self.episode_files)

        # 尝试提取样本数据
        for _ in range(3):  # 最多尝试3次
            item = self.extract_episode_item_with_noise(episode_file)
            if item is not None:
                return item
            # 如果当前样本提取失败，随机选择另一个
            episode_file = random.choice(self.episode_files)

        print(f"Warning: Failed to extract sample from {episode_file}")
        return None

    def extract_episode_item_with_noise(self, hdf5_file_path):
        """
        从HDF5文件中提取带有62D噪声构造的单个样本

        Args:
            hdf5_file_path: HDF5文件路径

        Returns:
            包含提取数据的字典，如果提取失败则返回None
        """
        try:
            with h5py.File(hdf5_file_path, 'r', swmr=True, libver='latest') as f:
                # 加载14D机器人动作数据
                try:
                    left_arm = f["joint_action/left_arm"][:]
                    left_gripper = f["joint_action/left_gripper"][:]
                    right_arm = f["joint_action/right_arm"][:]
                    right_gripper = f["joint_action/right_gripper"][:]

                    # 处理维度不匹配
                    if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                        left_gripper = left_gripper.reshape(-1, 1)
                    if len(right_arm.shape) == 2 and len(right_gripper.shape) == 1:
                        right_gripper = right_gripper.reshape(-1, 1)

                    # 构造14D机器人数据 [left_arm(6D) + left_gripper(1D) + right_arm(6D) + right_gripper(1D)]
                    robot_action_14d = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)

                except Exception as e:
                    print(f"Error loading robot action data: {e}")
                    return None

                # 随机选择时间点
                max_index = len(robot_action_14d) - 2
                if max_index <= 0:
                    return None

                index = random.randint(0, max_index)

                # 当前状态 (14D机器人数据)
                robot_current = robot_action_14d[index]

                # 未来动作序列 (14D)
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                robot_chunk = robot_action_14d[index+1:action_end+1:self.upsample_rate]

                # 填充如果不足
                if robot_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(robot_chunk[-1:], self.chunk_size - robot_chunk.shape[0], axis=0)
                    robot_chunk = np.concatenate([robot_chunk, last_part], axis=0)

                # ===== 关键步骤：构造62D数据 =====

                # 1. 生成48D人类噪声
                human_noise_48d = self.generate_human_noise(self.chunk_size).numpy()  # [chunk_size, 48]

                # 2. 对14D机器人数据进行归一化 (关键修复！)
                robot_current_normalized = self.normalize_action(robot_current)  # [14]
                robot_chunk_normalized = self.normalize_action(robot_chunk)  # [chunk_size, 14]

                # 3. 构造62D状态数据：48D噪声 + 14D归一化真实数据
                state_62d = np.concatenate([
                    human_noise_48d[0:1],                    # 当前状态：48D噪声 [1, 48]
                    robot_current_normalized[np.newaxis, :]  # 当前状态：14D归一化真实数据 [1, 14]
                ], axis=1).squeeze(0)  # [62]

                # 4. 构造62D动作数据：48D噪声 + 14D归一化真实数据
                action_62d = np.concatenate([
                    human_noise_48d,         # 动作序列：48D噪声 [chunk_size, 48]
                    robot_chunk_normalized   # 动作序列：14D归一化真实数据 [chunk_size, 14]
                ], axis=1)  # [chunk_size, 62]

                # 加载多视角相机数据
                try:
                    current_images = []

                    # 定义相机路径
                    camera_paths = {
                        "head_camera": "observation/head_camera/rgb",
                        "left_camera": "observation/left_camera/rgb",
                        "right_camera": "observation/right_camera/rgb",
                        "wrist_left_camera": "observation/wrist_left_camera/rgb",
                        "wrist_right_camera": "observation/wrist_right_camera/rgb"
                    }

                    # 从每个配置的摄像头加载图像
                    for cam_idx, cam_name in enumerate(self.cameras):
                        cam_path = camera_paths.get(cam_name)
                        if cam_path and cam_path in f:
                            camera_data = f[cam_path]
                            img_frames = self.parse_img_data(camera_data, index)
                            current_images.append(img_frames)
                        else:
                            print(f"Warning: Camera {cam_name} not found in {hdf5_file_path}")
                            return None

                    # 确保摄像头数量正确
                    if len(current_images) != self.num_cameras:
                        print(f"Error: Expected {self.num_cameras} cameras, but got {len(current_images)}")
                        return None

                    # 转换为numpy数组，形状为 [num_cameras, history_size, H, W, 3]
                    img_frames_np = np.array(current_images)

                    # 为每个摄像头创建图像掩码
                    mask_length = self.img_history_size
                    current_images_mask = [
                        np.array([True]*mask_length, dtype=bool) for _ in range(self.num_cameras)
                    ]

                except Exception as e:
                    print(f"Error accessing camera data in {hdf5_file_path}: {e}")
                    traceback.print_exc()
                    return None

                # 加载预编码的语言指令
                language_embedding = self.load_language_embedding(hdf5_file_path)
                if language_embedding is None:
                    # 如果没有语言嵌入，创建一个默认的
                    language_embedding = torch.zeros(1024, 4096)  # 默认大小

                state_indicator = np.ones_like(state_62d)
                action_norm = np.ones_like(action_62d)

                # 创建结果字典
                result = {
                    "current_images": img_frames_np,  # 当前帧图像
                    "current_images_mask": current_images_mask,  # 图像掩码
                    "actions": action_62d,  # 62D动作序列
                    "states": np.expand_dims(state_62d, axis=0),  # 62D状态
                    "state_indicator": state_indicator,
                    "action_norm": action_norm,
                    "instruction": language_embedding,  # 预编码的语言指令
                    "dataset_name": self.get_dataset_name(),  # 数据集名称
                }

                return result

        except Exception as e:
            print(f"Error processing {hdf5_file_path}: {e}")
            traceback.print_exc()
            return None

    def load_language_embedding(self, hdf5_file_path):
        """
        从集中的lang_embeddings文件夹加载预编码的语言指令嵌入

        Args:
            hdf5_file_path (str): HDF5 episode文件路径

        Returns:
            torch.Tensor: 任务指令嵌入，如果未找到则返回None
        """
        try:
            # 构建嵌入文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_path = os.path.join(current_dir, 'lang_embeddings', f"{self.task_name}.pt")

            if not os.path.exists(embedding_path):
                # 如果没有特定任务的嵌入，尝试使用通用的
                embedding_path = os.path.join(current_dir, 'lang_embeddings', "default.pt")

            if not os.path.exists(embedding_path):
                print(f"Warning: No language embedding found for {self.task_name}")
                return None

            # 加载嵌入数据
            embedding_data = torch.load(embedding_path, map_location='cpu')

            # 提取嵌入张量
            embeddings = embedding_data.get('embeddings', None)

            if embeddings is None:
                print(f"Warning: No embeddings found in {embedding_path}")
                return None

            # 如果存在批次维度则移除（从3D转换为2D）
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)

            return embeddings

        except Exception as e:
            print(f"Error loading language embedding for {hdf5_file_path}: {e}")
            return None

    def get_noise_statistics(self, num_samples=1000):
        """获取噪声统计信息"""
        noise_samples = []
        for _ in range(num_samples):
            noise = self.generate_human_noise(1).numpy()[0]
            noise_samples.append(noise)

        noise_samples = np.array(noise_samples)
        return {
            'mean': np.mean(noise_samples, axis=0),
            'std': np.std(noise_samples, axis=0),
            'min': np.min(noise_samples, axis=0),
            'max': np.max(noise_samples, axis=0)
        }

    def __getitem__(self, idx):
        """保持向后兼容性的getitem方法"""
        return self.get_item(idx)