import os
import random
import h5py
import numpy as np
import cv2
import json
import pandas as pd
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

# Note: imgaug not used in original dataset class

if not hasattr(np, 'bool'):
    np.bool = bool

class RobotwinAgilexWithNoiseDataset:
    """
    RobotWin数据集的62D微调版本：48D噪声 + 14D真实数据
    用于保留预训练中学到的人类先验知识

    相比于robotwin_agilex_with_noise_dataset_nonorm.py

    添加了后14d动作归一化和损失缩放   前48维保持标准高斯 损失缩放为1


    """
    def __init__(
        self, 
        mode="single_task",  # "single_task" or "multi_task"
        single_task_root_dir=None,  # For single task mode
        multi_task_root_dir=None,  # For multi task mode
        task_name=None,  # Required for single_task mode
        hdf5_folder=None,  # Required for single_task mode, e.g., "demo_clean/data"
        max_episodes=None,  # Maximum number of episodes to load for single task
        config=None,
        stat_path=None,
        upsample_rate=3,
        val=False,
        image_corrupt_severity=5,
        noise_strategy="gaussian",  # "gaussian", "uniform", "zero"
        noise_scale=1.0,  # 噪声缩放因子
        human_data_stats=None  # 人类数据统计信息，用于生成更真实的噪声
    ):
        """
        初始化带噪声的RobotWin数据集
        
        Args:
            noise_strategy: 噪声生成策略
                - "gaussian": 高斯噪声
                - "uniform": 均匀噪声  
                - "zero": 零噪声
            noise_scale: 噪声缩放因子
            human_data_stats: 人类数据统计信息字典，包含mean和std
        """
        self.DATASET_NAME = "robotwin_agilex_with_noise"
        self.mode = mode
        self.single_task_root_dir = single_task_root_dir
        self.multi_task_root_dir = multi_task_root_dir
        self.task_name = task_name
        self.hdf5_folder = hdf5_folder
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_corrupt_severity = image_corrupt_severity
        
        # Set default stat_path if not provided (relative to this file)
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'stats.json')
        
        # Load statistics for normalization
        with open(stat_path, 'r') as file:
            stat = json.load(file)
        self.action_min = np.array(stat['robotwin_agilex']['min'])
        self.action_max = np.array(stat['robotwin_agilex']['max'])
        
        # 计算用于action_norm的归一化范围
        self.range_14d = self.action_max - self.action_min
        # 防止除以零：将零范围替换为很小的正数
        self.range_14d[self.range_14d == 0] = 1e-6
        
        # 预计算62维action_norm模板
        # 前48维: 1.0 (噪声部分，中性值)
        # 后14维: 真实动作范围 (用于损失缩放)
        self.action_norm_template_62d = np.ones(62, dtype=np.float32)
        self.action_norm_template_62d[48:] = self.range_14d
        
        # 噪声相关参数
        self.noise_strategy = noise_strategy
        self.noise_scale = noise_scale
        self.human_data_stats = human_data_stats or {}
        
        # 验证模式特定参数
        if mode == "single_task":
            if not task_name or not hdf5_folder:
                raise ValueError("single_task mode requires task_name and hdf5_folder parameters")
        elif mode == "multi_task":
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single_task' or 'multi_task'")
        
        # 设置基本参数 - 注意这里是62D！
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = 62  # 62D状态 (48D噪声 + 14D真实)
        self.action_dim = 62  # 62D动作 (48D噪声 + 14D真实)
        self.img_history_size = config['common']['img_history_size']
        
        # 设置相机参数
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 3:
            self.cameras = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 3 cameras supported.")

        self.camera_mapping = {
            "cam_high": "head_camera", 
            "cam_left_wrist": "left_camera",
            "cam_right_wrist": "right_camera"
        }

        # Initialize data structures
        if mode == "single_task":
            self.episode_files = []  # List of episode files for single task
        else:
            self.task_to_episodes = {}  # Task name -> episode files mapping for multi task
            self.task_weights = {}      # Task sampling weights
        
        self.total_episodes = 0     # Total number of episodes
        
        # Initialize dataset
        self._initialize_dataset()
        
        # 打印归一化配置信息
        print(f"=== 62D微调归一化配置 ===")
        print(f"机器人动作范围 (14D):")
        print(f"action_norm模板 (62D):")
        print(f"  前48维: 1.0 (噪声部分)")
        print(f"  后14维: {self.range_14d} (真实动作范围)")
        print(f"========================")
            
        # 初始化噪声生成器（新增功能）
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
    
    def _scan_single_task_folder(self):
        """Scan single task folder to get all HDF5 files"""
        task_dir = os.path.join(self.single_task_root_dir, self.task_name, self.hdf5_folder)
        if not os.path.exists(task_dir):
            print(f"Warning: Task folder {task_dir} does not exist")
            return []
        
        # Find all HDF5 files in the folder
        hdf5_files = []
        for f in os.listdir(task_dir):
            if f.endswith(".hdf5"):
                hdf5_path = os.path.join(task_dir, f)
                hdf5_files.append(hdf5_path)
        
        # Sort files to ensure consistent ordering
        hdf5_files.sort()
        
        # Limit number of episodes if specified
        if self.max_episodes is not None:
            hdf5_files = hdf5_files[:self.max_episodes]
        
        print(f"Single task {self.task_name}: Found {len(hdf5_files)} HDF5 files")
        return hdf5_files
    
    def _scan_multi_task_folders(self):
        """Scan multi-task folders to get all HDF5 files"""
        if not os.path.exists(self.multi_task_root_dir):
            print(f"Warning: Multi-task root directory {self.multi_task_root_dir} does not exist")
            return {}
        
        # Get all task folders
        task_folders = [d for d in os.listdir(self.multi_task_root_dir) 
                       if os.path.isdir(os.path.join(self.multi_task_root_dir, d))]
        
        task_to_episodes = {}
        
        for task_folder in task_folders:
            task_dir = os.path.join(self.multi_task_root_dir, task_folder)
            hdf5_files = []
            
            # Recursively find all HDF5 files in task folder
            for root, dirs, files in os.walk(task_dir):
                for file in files:
                    if file.endswith(".hdf5"):
                        hdf5_path = os.path.join(root, file)
                        hdf5_files.append(hdf5_path)
            
            # Shuffle files for randomness
            random.shuffle(hdf5_files)
            task_to_episodes[task_folder] = hdf5_files
            print(f"Multi-task {task_folder}: Found {len(hdf5_files)} HDF5 files")
        
        return task_to_episodes
    
    def _initialize_dataset(self):
        """Initialize dataset by rescanning folders and updating sampling weights"""
        print("Initializing dataset...")
        
        if self.mode == "single_task":
            # Single task mode: scan single task folder
            self.episode_files = self._scan_single_task_folder()
            self.total_episodes = len(self.episode_files)
            
            if self.total_episodes == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")
            
            print(f"Single task dataset initialized. Total {self.total_episodes} episodes")
        
        else:
            # Multi task mode: scan all task folders
            self.task_to_episodes = self._scan_multi_task_folders()
            
            # Calculate total episodes
            all_task_count = sum(len(episodes) for episodes in self.task_to_episodes.values())
            
            if all_task_count == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")
            
            # Calculate sampling weights - equal weight for all tasks (1:1:1:1:1:1 sampling)
            task_count = len(self.task_to_episodes)
            for task_name in self.task_to_episodes.keys():
                self.task_weights[task_name] = 1.0 / task_count
            
            self.total_episodes = all_task_count
            
            print(f"Multi-task dataset initialized. Total {all_task_count} episodes across {task_count} tasks")
            print(f"Task weights: {self.task_weights}")
    
    def __len__(self):
        """Return approximate dataset length"""
        return self.total_episodes * 200  # Assume 200 samples per HDF5 file
    
    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME
    
    def parse_img_data(self, dataset, idx):
        """
        Process single camera image data from individual camera paths
        
        Args:
            dataset: HDF5 dataset containing single camera images
            idx: Current frame index
            
        Returns:
            Processed image sequence with shape [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # Use standard resolution for individual camera images
        frames = np.zeros((num_frames, 240, 320, 3), dtype=np.uint8)  # Standard size: 320x240
        
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
            print(f"[DEBUG] decode_image_with_opencv error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames])
        
        return frames

    def decode_image_with_opencv(self, img_data):
        """
        Decode image data using OpenCV, maintaining RGB format
        
        Args:
            img_data (bytes): Binary image data
        
        Returns:
            np.ndarray: Decoded image array, shape=(240, 320, 3), RGB format
        """
        try:
            # Decode using OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure decoding was successful
            if bgr_img is None:
                raise Exception("OpenCV decoding failed")
                
            # OpenCV uses BGR format by default, convert to RGB
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            
            # Resize to expected standard size if needed (320x240)
            if rgb_img.shape[:2] != (240, 320):
                rgb_img = cv2.resize(rgb_img, (320, 240))
                
            return rgb_img
            
        except Exception as e:
            print(f"[DEBUG] Image decoding failed: {e}")
            return None

    def load_language_embedding(self, hdf5_file_path):
        """
        Load pre-encoded language instruction embedding from centralized lang_embeddings folder
        
        Args:
            hdf5_file_path (str): Path to HDF5 episode file
            
        Returns:
            torch.Tensor: Task instruction embedding, or None if not found
        """
        try:
            # Extract task name from HDF5 file path
            task_name = None
            
            if self.mode == "single_task":
                # For single task mode, use the provided task_name
                task_name = self.task_name
            else:
                # For multi-task mode, extract task name from file path
                if self.multi_task_root_dir and self.multi_task_root_dir in hdf5_file_path:
                    # Get relative path from multi_task_root_dir
                    relative_path = os.path.relpath(hdf5_file_path, self.multi_task_root_dir)
                    # Task name is the first directory in relative path
                    task_name = relative_path.split(os.sep)[0]
            
            if task_name is None:
                print(f"Warning: Could not extract task name from path {hdf5_file_path}")
                return None
            
            # Build embedding file path using relative path to current script location
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
    
    
    def generate_human_noise(self, batch_size=1):
        """
        生成48D人类动作噪声
        
        Args:
            batch_size: 批次大小
            
        Returns:
            torch.Tensor: [batch_size, 48] 噪声数据
        """
        return self.noise_generator(batch_size)
    
    def get_item(self, index=None):
        """
        Get data item, randomly select one if index is None
        
        Args:
            index: Optional, specified index. If None, randomly select
            
        Returns:
            Processed data dictionary, or None if failed
        """
        if self.mode == "single_task":
            # Single task mode: randomly select from episode files
            if not self.episode_files:
                self._initialize_dataset()
            
            if not self.episode_files:
                print("Error: No available episodes")
                return None
            
            # Randomly select an HDF5 file
            episode_file = random.choice(self.episode_files)
            
        else:
            # Multi task mode: balanced task sampling
            if not self.task_to_episodes:
                self._initialize_dataset()
            
            # Randomly select a task based on task weights
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1
            )[0]
            
            # Randomly select a sample from the selected task
            task_episodes = self.task_to_episodes.get(task_name, [])
            if not task_episodes:
                print(f"Warning: Task {task_name} has no available samples")
                # Select from other tasks
                alternative_tasks = [t for t in self.task_to_episodes.keys() if t != task_name and self.task_to_episodes.get(t, [])]
                if not alternative_tasks:
                    print("Error: No available samples")
                    return None
                task_name = random.choice(alternative_tasks)
                task_episodes = self.task_to_episodes.get(task_name, [])
            
            # Randomly select an HDF5 file
            episode_file = random.choice(task_episodes)
        
        # Try to extract sample data
        for _ in range(3):  # Maximum 3 attempts
            item = self.extract_episode_item_with_noise(episode_file)
            if item is not None:
                return item
            # If current sample extraction fails, randomly select another
            if self.mode == "single_task":
                episode_file = random.choice(self.episode_files)
            else:
                task_episodes = self.task_to_episodes.get(task_name, [])
                if task_episodes:
                    episode_file = random.choice(task_episodes)
        
        print(f"Warning: Failed to extract sample, returning None")
        return None
    
    def extract_episode_item_with_noise(self, hdf5_file_path):
        """
        Extract a single sample from HDF5 file with 62D noise construction
        
        Args:
            hdf5_file_path: HDF5 file path
            
        Returns:
            Dictionary containing extracted data, or None if extraction fails
        """
        try:
            with h5py.File(hdf5_file_path, 'r', swmr=True, libver='latest') as f:
                # Load 14D robot action data
                try:
                    left_arm = f["joint_action/left_arm"][:]
                    left_gripper = f["joint_action/left_gripper"][:]
                    right_arm = f["joint_action/right_arm"][:]
                    right_gripper = f["joint_action/right_gripper"][:]
                    
                    # Handle dimension mismatch
                    if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                        left_gripper = left_gripper.reshape(-1, 1)
                    if len(right_arm.shape) == 2 and len(right_gripper.shape) == 1:
                        right_gripper = right_gripper.reshape(-1, 1)
                    
                    # Construct 14D robot data [left_arm(6D) + left_gripper(1D) + right_arm(6D) + right_gripper(1D)]
                    robot_action_14d = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                    
                except Exception as e:
                    print(f"Error loading robot action data: {e}")
                    return None
                
                # Randomly select time point
                max_index = len(robot_action_14d) - 2
                index = random.randint(0, max_index)
                
                # Current state (14D robot data)
                robot_current = robot_action_14d[index]
                
                # Future action sequence (14D)
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                robot_chunk = robot_action_14d[index+1:action_end+1:self.upsample_rate]
                
                # Padding if insufficient
                if robot_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(robot_chunk[-1:], self.chunk_size - robot_chunk.shape[0], axis=0)
                    robot_chunk = np.concatenate([robot_chunk, last_part], axis=0)
                
                # ===== Key step: Construct 62D data with proper normalization =====
                
                # 1. Generate 48D human noise (继续保持标准高斯噪声)
                human_noise_48d = self.generate_human_noise(self.chunk_size).numpy()  # [chunk_size, 48]
                
                # 2. 归一化后14维机器人动作数据到[-1, 1]范围
                # 使用min-max归一化: (x - min) / (max - min) * 2 - 1
                robot_current_normalized = (robot_current - self.action_min) / self.range_14d * 2 - 1
                robot_chunk_normalized = (robot_chunk - self.action_min) / self.range_14d * 2 - 1
                
                # 3. Construct 62D state data: 48D noise + 14D normalized real
                state_62d = np.concatenate([
                    human_noise_48d[0:1],                    # Current state: 48D noise [1, 48]
                    robot_current_normalized[np.newaxis, :] # Current state: 14D normalized [1, 14]
                ], axis=1).squeeze(0)  # [62]
                
                # 4. Construct 62D action data: 48D noise + 14D normalized real
                action_62d = np.concatenate([
                    human_noise_48d,           # Action sequence: 48D noise [chunk_size, 48]
                    robot_chunk_normalized     # Action sequence: 14D normalized [chunk_size, 14]
                ], axis=1)  # [chunk_size, 62]
                
                # 5. 构造正确的action_norm：前48维为1.0，后14维为真实动作范围
                action_norm_final = np.broadcast_to(self.action_norm_template_62d, action_62d.shape)
                
                # Load multi-view camera data
                try:
                    current_images = []
                    
                    # Define camera paths for 3-view setup
                    camera_paths = {
                        "cam_high": "observation/head_camera/rgb",
                        "cam_left_wrist": "observation/left_camera/rgb",
                        "cam_right_wrist": "observation/right_camera/rgb"
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
                    
                    # Ensure we have the correct number of cameras
                    if len(current_images) != self.num_cameras:
                        print(f"Error: Expected {self.num_cameras} cameras, but got {len(current_images)}")
                        return None
                    
                    # Convert to numpy array with shape [num_cameras, history_size, H, W, 3]
                    img_frames_np = np.array(current_images)
                    
                    # Create image masks for each camera
                    mask_length = self.img_history_size
                    current_images_mask = [
                        np.array([True]*mask_length, dtype=bool) for _ in range(self.num_cameras)
                    ]
                    
                except Exception as e:
                    print(f"Error accessing camera data in {hdf5_file_path}: {e}")
                    traceback.print_exc()
                    return None
                
                # Load pre-encoded language instruction
                language_embedding = self.load_language_embedding(hdf5_file_path)
                if language_embedding is None:
                    print(f"Warning: Failed to load language embedding for {hdf5_file_path}")
                    return None
                
                state_indicator = np.ones_like(state_62d)
                # action_norm已经正确构造为action_norm_final
                
                # Create result dictionary
                result = {
                    "current_images": img_frames_np,  # Current frame images
                    "current_images_mask": current_images_mask,  # Image masks
                    "actions": action_62d,  # 62D action sequence (后14维已归一化到[-1,1])
                    "states": np.expand_dims(state_62d, axis=0),  # 62D states (后14维已归一化到[-1,1])
                    "state_indicator": state_indicator,
                    "action_norm": action_norm_final,  # 正确的归一化因子: 前48维=1.0, 后14维=真实范围
                    "instruction": language_embedding,  # Pre-encoded language instruction
                    "dataset_name": self.DATASET_NAME,  # Dataset name
                }
                
                return result
                
        except Exception as e:
            print(f"Error processing {hdf5_file_path}: {e}")
            traceback.print_exc()
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