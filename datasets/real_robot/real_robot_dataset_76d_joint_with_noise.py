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


class RealRobotDataset76dJointWithNoise:
    """
    真机数据集的76D微调版本：48D噪声 + 14D真实 + 14D噪声
    用于真机环境的76D微调训练，保留预训练中学到的人类先验知识

    76D动作空间分配：
    - [0-47]:   48D人类动作（噪声填充）
    - [48-61]:  14D双臂关节动作（真实机器人数据）
    - [62-75]:  14D末端位姿（噪声填充）
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
        human_data_stats: dict = None,  # 人类数据统计信息，用于生成更真实的噪声
        end_effector_noise_scale: float = 1.0  # 末端执行器噪声缩放因子
    ):
        """
        初始化76D带噪声的真机数据集

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
            noise_scale: 人类动作噪声缩放因子
            human_data_stats: 人类数据统计信息字典，包含mean和std
            end_effector_noise_scale: 末端执行器噪声缩放因子
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

        # 设置基本参数 - 注意这里是76D！
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = 76  # 76D状态 (48D噪声 + 14D真实 + 14D噪声)
        self.action_dim = 76  # 76D动作 (48D噪声 + 14D真实 + 14D噪声)
        self.img_history_size = config['common']['img_history_size']

        # 设置相机参数
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 1:
            self.cameras = ["head_camera"]
        elif self.num_cameras == 2:
            self.cameras = ["head_camera", "left_camera"]
        elif self.num_cameras == 3:
            # 根据HDF5文件结构调整摄像头名称
            self.cameras = ["head_camera", "wrist_left_camera", "wrist_right_camera"]
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 1, 2, or 3 cameras supported.")

        # 加载统计信息用于归一化
        # 只有在normalize_actions=True时才加载统计文件
        if self.normalize_actions:
            if stat_path is None:
                raise ValueError(
                    "normalize_actions is True but action_stats_file is not specified. "
                    "Please provide --action_stats_file parameter when using --normalize_actions."
                )

            if not os.path.exists(stat_path):
                raise FileNotFoundError(
                    f"Statistics file not found at specified path: {stat_path}. "
                    "Please provide a valid statistics file path when using --normalize_actions."
                )

            try:
                with open(stat_path, 'r') as file:
                    stat = json.load(file)
                if task_name in stat:
                    self.action_min = np.array(stat[task_name]['min'])
                    self.action_max = np.array(stat[task_name]['max'])
                elif 'real_robot_76d' in stat:
                    self.action_min = np.array(stat['real_robot_76d']['min'])
                    self.action_max = np.array(stat['real_robot_76d']['max'])
                elif 'real_robot' in stat:
                    # 尝试使用现有的统计信息并扩展到76D
                    existing_min = np.array(stat['real_robot']['min'])  # 应该是14D
                    existing_max = np.array(stat['real_robot']['max'])  # 应该是14D

                    # 扩展到76D：[0-47]人类动作噪声，[48-61]真实机器人动作，[62-75]末端位姿噪声
                    human_min = np.array([-1.0] * 48)  # 人类动作噪声范围
                    human_max = np.array([1.0] * 48)
                    eef_min = np.array([-1.0] * 14)   # 末端位姿噪声范围
                    eef_max = np.array([1.0] * 14)

                    self.action_min = np.concatenate([human_min, existing_min, eef_min])
                    self.action_max = np.concatenate([human_max, existing_max, eef_max])
                    print(f"Expanded 14D stats to 76D: human [0-47], robot [48-61], eef [62-75]")
                else:
                    raise ValueError(f"No statistics found for task {task_name} in file: {stat_path}")

                print(f"Loaded statistics from specified file: {stat_path}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in statistics file {stat_path}: {e}")
            except KeyError as e:
                raise ValueError(f"Missing required key {e} in statistics file {stat_path}")
            except Exception as e:
                raise ValueError(f"Error loading statistics from {stat_path}: {e}")
        else:
            # 不进行归一化时，设置虚拟的min/max值（不会被使用）
            self.action_min = np.zeros(self.action_dim)
            self.action_max = np.ones(self.action_dim)

        # 噪声相关参数
        self.noise_strategy = noise_strategy
        self.noise_scale = noise_scale
        self.end_effector_noise_scale = end_effector_noise_scale
        self.human_data_stats = human_data_stats or {}

        # 初始化数据结构
        self.episode_files = []
        self.total_episodes = 0

        # 初始化数据集
        self._initialize_dataset()

        # 初始化噪声生成器
        self.noise_generator_human = self._init_noise_generator(dim=48, scale=self.noise_scale)
        self.noise_generator_eef = self._init_noise_generator(dim=14, scale=self.end_effector_noise_scale)

        print(f"76D Real Robot Dataset initialized:")
        print(f"  - Dimensions: {self.action_dim}D")
        print(f"  - Human actions [0-47]: Noise padding (strategy: {self.noise_strategy})")
        print(f"  - Robot joints [48-61]: Real dual-arm joint actions")
        print(f"  - End-effector [62-75]: Noise padding (strategy: {self.noise_strategy})")
        print(f"  - Total episodes: {self.total_episodes}")

    def _init_noise_generator(self, dim: int, scale: float = 1.0):
        """初始化噪声生成器"""
        if self.noise_strategy == "gaussian":
            if self.human_data_stats and 'mean' in self.human_data_stats:
                # 使用人类数据统计信息生成更真实的噪声
                if dim == 48:
                    human_mean = self.human_data_stats['mean'][:dim]  # 前48D
                    human_std = self.human_data_stats['std'][:dim]
                else:  # dim == 14
                    # 对于末端位姿，使用缩放后的统计信息
                    human_mean = np.zeros(dim)  # 末端位姿通常中心化
                    human_std = np.ones(dim) * scale

                def generator(batch_size):
                    noise = torch.randn(batch_size, dim) * human_std + human_mean
                    # 截断到 [-scale, scale] 范围
                    return torch.clamp(noise, -scale, scale)
            else:
                # 截断高斯噪声 - 确保噪声在[-scale, scale]范围内
                def generator(batch_size):
                    noise = torch.randn(batch_size, dim) * scale
                    # 截断到 [-scale, scale] 范围
                    return torch.clamp(noise, -scale, scale)

        elif self.noise_strategy == "uniform":
            def generator(batch_size):
                return (torch.rand(batch_size, dim) - 0.5) * 2 * scale

        elif self.noise_strategy == "zero":
            def generator(batch_size):
                return torch.zeros(batch_size, dim)

        else:
            raise ValueError(f"Unknown noise strategy: {self.noise_strategy}")

        return generator

    def _initialize_dataset(self):
        """初始化数据集，扫描HDF5文件"""
        print("Initializing 76D real robot dataset...")

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
        return self.noise_generator_human(batch_size)

    def generate_eef_noise(self, batch_size=1):
        """
        生成14D末端位姿噪声

        Args:
            batch_size: 批次大小

        Returns:
            torch.Tensor: [batch_size, 14] 噪声数据
        """
        return self.noise_generator_eef(batch_size)

    def load_language_embedding(self, hdf5_file_path):
        """
        Load pre-encoded language instruction embedding from lang_embeddings folder

        Args:
            hdf5_file_path (str): Path to HDF5 episode file

        Returns:
            torch.Tensor: Task instruction embedding, or None if not found
        """
        try:
            # For real robot, use the configured task_name
            task_name = self.task_name

            if task_name is None:
                print(f"Warning: No task name configured")
                return None

            # Build embedding file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_path = os.path.join(current_dir, 'lang_embeddings', f"{task_name}.pt")

            if not os.path.exists(embedding_path):
                print(f"Warning: Task embedding file not found: {embedding_path}")
                return None

            # Load embedding data
            embedding_data = torch.load(embedding_path, map_location='cpu')

            # Extract embeddings tensor
            embeddings = embedding_data.get('embeddings', None)

            if embeddings is None:
                print(f"Warning: No embeddings found in {embedding_path}")
                return None

            # Remove batch dimension if present (convert from 3D to 2D)
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)

            return embeddings

        except Exception as e:
            print(f"Error loading language embedding from {hdf5_file_path}: {e}")
            return None

    def parse_img_data(self, dataset, idx):
        """
        处理单个相机的图像数据

        Args:
            dataset: HDF5数据集，包含单个相机的图像
            idx: 当前帧索引

        Returns:
            处理后的图像序列，形状为 [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # Use standard resolution for camera images
        frames = np.zeros((num_frames, 360, 640, 3), dtype=np.uint8)  # Real robot size: 640x360

        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < len(dataset):
                    img_data = dataset[frame_idx]

                    decoded_img = self.decode_image_with_opencv(img_data)
                    if decoded_img is None:
                        raise Exception(f"[DEBUG] decode error")

                    if decoded_img is not None:
                        frames[i] = decoded_img

        except Exception as e:
            print(f"Error processing image data: {e}")
            # 使用黑色图像填充
            for i in range(num_frames):
                frames[i] = np.zeros((480, 640, 3), dtype=np.uint8)

        # 如果历史帧数不足，用第一帧填充
        if num_frames < self.img_history_size:
            padding = np.zeros((self.img_history_size - num_frames, 480, 640, 3), dtype=np.uint8)
            frames = np.concatenate([padding, frames], axis=0)

        return frames
    def decode_image_with_opencv(self, img_data):
        """使用OpenCV解码图像数据"""
        try:
            # 假设img_data是压缩的JPEG字节数据
            if isinstance(img_data, np.ndarray):
                # 如果是numpy数组，直接使用cv2.imdecode
                decoded_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if decoded_img is None:
                    # 如果解码失败，可能是原始数据
                    return img_data
                return decoded_img
            else:
                # 如果是字节数据，先转换为numpy数组
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return decoded_img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    def __len__(self):
        """返回数据集大小"""
        return self.total_episodes * 200  # 假设每个HDF5文件有200帧

    def get_dataset_name(self):
        """返回数据集名称"""
        return "real_robot_76d_joint_with_noise"

    def get_item(self, index=None):
        """
        获取数据项，如果index为None则随机选择

        Args:
            index: 可选，指定索引。如果为None，则随机选择

        Returns:
            处理后的数据字典，如果失败则返回None
        """
        # 随机选择一个HDF5文件
        if not self.episode_files:
            print("Error: No available episodes")
            return None

        episode_file = random.choice(self.episode_files)

        # 尝试提取样本数据
        for _ in range(3):  # 最多尝试3次
            try:
                result = self._extract_sample_from_hdf5(episode_file)
                if result is not None:
                    return result
            except Exception as e:
                print(f"Error processing {episode_file}: {e}")
                continue

        print(f"Failed to extract sample from {episode_file} after 3 attempts")
        return None

    def __getitem__(self, idx):
        """保持向后兼容性的getitem方法"""
        return self.get_item(idx)

    def _extract_sample_from_hdf5(self, hdf5_file_path):
        """
        从HDF5文件中提取样本并构造76D数据

        Args:
            hdf5_file_path: HDF5文件路径

        Returns:
            包含提取数据的字典，如果提取失败则返回None
        """
        try:
            with h5py.File(hdf5_file_path, 'r', swmr=True, libver='latest') as f:
                # 加载14D机器人关节动作数据 (dimensions 48-61)
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
                    robot_joint_14d = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)

                except Exception as e:
                    print(f"Error loading robot joint action data: {e}")
                    return None

                # 随机选择时间点
                total_frames = len(left_arm)
                if total_frames <= self.chunk_size:
                    print(f"Warning: Episode too short ({total_frames} frames), skipping")
                    return None

                index = random.randint(self.chunk_size, total_frames - 1)

                # 提取14D机器人动作块
                start_idx = index - self.chunk_size + 1
                robot_joint_chunk = robot_joint_14d[start_idx:index + 1]  # [chunk_size, 14]
                robot_joint_current = robot_joint_14d[index]  # [14]

                # ===== 关键步骤：构造76D数据 =====

                # 1. 生成48D人类噪声 (dimensions 0-47)
                human_noise_48d = self.generate_human_noise(self.chunk_size).numpy()  # [chunk_size, 48]

                # 2. 生成14D末端位姿噪声 (dimensions 62-75)
                eef_noise_14d = self.generate_eef_noise(self.chunk_size).numpy()  # [chunk_size, 14]

                # 3. 构造76D状态数据: 48D噪声 + 14D真实 + 14D噪声
                state_76d = np.concatenate([
                    human_noise_48d[0:1],              # 当前状态: 48D噪声 [1, 48]
                    robot_joint_current[np.newaxis, :], # 当前状态: 14D真实 [1, 14]
                    eef_noise_14d[0:1]                  # 当前状态: 14D噪声 [1, 14]
                ], axis=1).squeeze(0)  # [76]

                # 4. 构造76D动作数据: 48D噪声 + 14D真实 + 14D噪声
                action_76d = np.concatenate([
                    human_noise_48d,         # 动作序列: 48D噪声 [chunk_size, 48]
                    robot_joint_chunk,       # 动作序列: 14D真实 [chunk_size, 14]
                    eef_noise_14d            # 动作序列: 14D噪声 [chunk_size, 14]
                ], axis=1)  # [chunk_size, 76]

                # 加载多视角相机数据
                try:
                    current_images = []

                    # Define camera paths based on camera configuration
                    if self.num_cameras == 1:
                        camera_paths = {
                            "head_camera": "observation/head_camera/rgb"
                        }
                    elif self.num_cameras == 2:
                        # 双摄像头配置：head_camera + left_camera
                        camera_paths = {
                            "head_camera": "observation/head_camera/rgb",
                            "left_camera": "observation/left_camera/rgb"
                        }
                    elif self.num_cameras == 3:
                        # 三摄像头配置：head_camera + wrist_left_camera + wrist_right_camera
                        camera_paths = {
                            "head_camera": "observation/head_camera/rgb",
                            "wrist_left_camera": "observation/wrist_left_camera/rgb",
                            "wrist_right_camera": "observation/wrist_right_camera/rgb"
                        }

                    # Load images from each configured camera
                    for cam_idx, cam_name in enumerate(self.cameras):
                        cam_path = camera_paths.get(cam_name)
                        if cam_path and cam_path in f:
                            camera_data = f[cam_path]
                            img_frames = self.parse_img_data(camera_data, index)
                            current_images.append(img_frames)
                        else:
                            print(f"Warning: Camera {cam_name} not found in {hdf5_file_path}")
                            return None

                    # 确保我们有正确数量的相机
                    if len(current_images) != self.num_cameras:
                        print(f"Error: Expected {self.num_cameras} cameras, got {len(current_images)}")
                        return None

                    # 堆叠图像 [num_cameras, history_size, H, W, 3]
                    current_images = np.stack(current_images, axis=0)

                except Exception as e:
                    print(f"Error loading camera data: {e}")
                    return None

                # 加载预编码的语言指令嵌入
                language_embedding = self.load_language_embedding(hdf5_file_path)
                if language_embedding is None:
                    print(f"Warning: Failed to load language embedding for {hdf5_file_path}")
                    return None

                # 状态指示器：指示哪些维度是真实数据
                state_indicator = np.zeros(self.state_dim, dtype=np.float32)
                state_indicator[48:62] = 1.0  # 只有机器人关节动作是真实数据

                # 归一化动作数据 - 对所有维度进行归一化以保持与预训练一致
                if self.normalize_actions:
                    # 归一化机器人关节部分 [48-61] - 使用统计文件，归一化到[-1,1]以匹配预训练
                    robot_joint_min = self.action_min[48:62]  # 14D机器人关节的最小值
                    robot_joint_max = self.action_max[48:62]  # 14D机器人关节的最大值
                    # 归一化到[-1,1]范围：2 * (value - min) / (max - min) - 1
                    robot_joint_norm = 2 * (action_76d[:, 48:62] - robot_joint_min) / (robot_joint_max - robot_joint_min + 1e-8) - 1
                    robot_joint_norm = np.clip(robot_joint_norm, -1, 1)

                    # 人类动作噪声 [0-47] - 已经在[-noise_scale, noise_scale]范围内，与预训练一致
                    # 如果noise_scale=1.0，则噪声已在[-1,1]范围内，无需归一化
                    if self.noise_scale == 1.0:
                        human_noise_norm = action_76d[:, 0:48]  # 直接使用，已在[-1,1]
                    else:
                        # 如果scale不是1.0，需要缩放到[-1,1]
                        human_noise_norm = action_76d[:, 0:48] / self.noise_scale
                        human_noise_norm = np.clip(human_noise_norm, -1, 1)

                    # 末端位姿噪声 [62-75] - 已经在[-end_effector_noise_scale, end_effector_noise_scale]范围内
                    # 如果end_effector_noise_scale=1.0，则噪声已在[-1,1]范围内，无需归一化
                    if self.end_effector_noise_scale == 1.0:
                        eef_noise_norm = action_76d[:, 62:76]  # 直接使用，已在[-1,1]
                    else:
                        # 如果scale不是1.0，需要缩放到[-1,1]
                        eef_noise_norm = action_76d[:, 62:76] / self.end_effector_noise_scale
                        eef_noise_norm = np.clip(eef_noise_norm, -1, 1)

                    # 组合所有归一化后的数据
                    action_norm = np.concatenate([
                        human_noise_norm,     # [0-47] 归一化的人类噪声
                        robot_joint_norm,    # [48-61] 归一化的真实机器人数据
                        eef_noise_norm       # [62-75] 归一化的末端位姿噪声
                    ], axis=1)

                    # print(f"Normalized all 76D dimensions to [-1,1] range for pretraining compatibility")
                    # print(f"  - Human noise [0-47]: already in [-{self.noise_scale}, {self.noise_scale}] range, matching pretraining")
                    # print(f"  - Robot joints [48-61]: statistical normalization from real data to [-1,1]")
                    # print(f"  - End-effector noise [62-75]: already in [-{self.end_effector_noise_scale}, {self.end_effector_noise_scale}] range, matching pretraining")
                else:
                    action_norm = action_76d

                # 为每个摄像头创建图像掩码
                mask_length = self.img_history_size
                current_images_mask = [
                    np.array([True]*mask_length, dtype=bool) for _ in range(self.num_cameras)
                ]

                # 构造最终结果 - 修复：使用归一化数据作为actions字段
                result = {
                    "current_images": current_images,  # 当前帧图像 [num_cameras, history_size, H, W, 3]
                    "current_images_mask": current_images_mask,  # 图像掩码
                    "actions": action_norm,  # ✅ 修复：使用归一化数据
                    "states": np.expand_dims(state_76d, axis=0),  # 76D状态
                    "state_indicator": state_indicator,
                    "action_norm": action_norm,  # 保持兼容性
                    "instruction": language_embedding,  # 预编码的语言指令
                    "dataset_name": self.get_dataset_name(),  # 数据集名称
                }

                return result

        except Exception as e:
            print(f"Error processing {hdf5_file_path}: {e}")
            traceback.print_exc()
            return None

    def get_noise_statistics(self, num_samples=1000):
        """获取噪声统计信息"""
        human_noise_samples = []
        eef_noise_samples = []

        for _ in range(num_samples):
            human_noise = self.generate_human_noise(1).numpy()[0]
            eef_noise = self.generate_eef_noise(1).numpy()[0]
            human_noise_samples.append(human_noise)
            eef_noise_samples.append(eef_noise)

        human_noise_samples = np.array(human_noise_samples)
        eef_noise_samples = np.array(eef_noise_samples)

        return {
            'human_noise': {
                'mean': np.mean(human_noise_samples, axis=0),
                'std': np.std(human_noise_samples, axis=0),
                'min': np.min(human_noise_samples, axis=0),
                'max': np.max(human_noise_samples, axis=0),
            },
            'eef_noise': {
                'mean': np.mean(eef_noise_samples, axis=0),
                'std': np.std(eef_noise_samples, axis=0),
                'min': np.min(eef_noise_samples, axis=0),
                'max': np.max(eef_noise_samples, axis=0),
            }
        }


def test_dataset():
    """测试数据集"""
    config = {
        'common': {
            'action_chunk_size': 16,
            'img_history_size': 2,
            'num_cameras': 3,
        }
    }

    dataset = RealRobotDataset76dJointWithNoise(
        root_dir="/home/yinzhenhan/Human_Pre/H_RDT/datasets/real_robot",
        task_name="test_task",
        config=config,
        noise_strategy="gaussian",
        max_episodes=1
    )

    # 获取一个样本
    sample = dataset.get_item()
    if sample:
        print("Sample keys:", sample.keys())
        print("Action shape:", sample['actions'].shape)
        print("State shape:", sample['states'].shape)
        print("Images shape:", sample['observations']['images'].shape)
        print("Dataset name:", sample['dataset_name'])
    else:
        print("Failed to get sample")


if __name__ == "__main__":
    test_dataset()