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

class RealRobotDataset7D:
    """
    Dataset for loading real robot single-arm data with 7 dimensions.
    Supports single-task mode for real robot fine-tuning with 7D actions (6D arm + 1D gripper).
    """
    def __init__(
        self,
        root_dir=None,  # Root directory for real robot data
        task_name=None,  # Task name (e.g., "pick_trash_exp")
        hdf5_folder=None,  # HDF5 folder path (e.g., "data")
        max_episodes=None,  # Maximum number of episodes to load
        config=None,
        stat_path=None,
        upsample_rate=3,
        val=False,
        image_corrupt_severity=5,
        normalize_actions=False,  # Whether to normalize actions to [-1, 1]
        dataset_name="real_robot_7d",  # Allow overriding dataset name
    ):
        """
        Initialize real robot dataset for 7D single-arm data.

        Args:
            root_dir: Root directory for real robot data
            task_name: Task name (e.g., "pick_trash_exp")
            hdf5_folder: HDF5 folder path (e.g., "data")
            max_episodes: Maximum episodes to load (None = all)
            config: Configuration dictionary
            stat_path: Path to normalization statistics file
            upsample_rate: Temporal data upsampling rate
            val: Whether this is validation set
            image_corrupt_severity: Image corruption severity level
            normalize_actions: Whether to normalize actions to [-1, 1] range
        """
        self.DATASET_NAME = dataset_name
        self.root_dir = root_dir
        self.task_name = task_name
        self.hdf5_folder = hdf5_folder
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_corrupt_severity = image_corrupt_severity
        self.normalize_actions = normalize_actions

        # Validate required parameters
        if not root_dir or not task_name or not hdf5_folder:
            raise ValueError("root_dir, task_name, and hdf5_folder parameters are required")
        
        # Set basic parameters
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = config['common']['state_dim']  # Should be 7
        self.action_dim = config['common']['action_dim']  # Should be 7
        self.img_history_size = config['common']['img_history_size']
        
        # Set camera parameters for camera setup
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 1:
            self.cameras = ["head_camera"]  # Single camera setup
        elif self.num_cameras == 2:
            self.cameras = ["head_camera", "left_camera"]  # Dual camera setup
        elif self.num_cameras == 3:
            self.cameras = ["head_camera", "wrist_left_camera", "wrist_right_camera"]  # Triple camera setup
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, only 1, 2, or 3 cameras supported for real robot.")
        
        # Set default stat_path if not provided
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Try to find task-specific stats file first, then fall back to generic stats.json
            task_specific_stats = os.path.join(current_dir, f'stats_{task_name}.json')
            generic_stats = os.path.join(current_dir, 'stats_7d.json')  # 7D specific stats

            if os.path.exists(task_specific_stats):
                stat_path = task_specific_stats
                print(f"Using task-specific stats file: {stat_path}")
            elif os.path.exists(generic_stats):
                stat_path = generic_stats
                print(f"Using generic 7D stats file: {stat_path}")
            else:
                stat_path = generic_stats  # Keep the default path for creating new stats
                print(f"No stats file found, will create default at: {stat_path}")

        # Load statistics for normalization (create default if not exists)
        if os.path.exists(stat_path):
            with open(stat_path, 'r') as file:
                stat = json.load(file)
            if 'real_robot_7d' in stat:
                self.action_min = np.array(stat['real_robot_7d']['min'])
                self.action_max = np.array(stat['real_robot_7d']['max'])
            elif 'real_robot' in stat:
                # Fallback to generic real_robot stats
                self.action_min = np.array(stat['real_robot']['min'][:7])  # Take first 7 dimensions
                self.action_max = np.array(stat['real_robot']['max'][:7])
                print(f"Warning: Using generic real_robot stats, taking first 7 dimensions")
            else:
                # Create default normalization stats for 7D
                self.action_min = np.zeros(self.action_dim)
                self.action_max = np.ones(self.action_dim)
        else:
            # Create default normalization stats for 7D
            self.action_min = np.zeros(self.action_dim)
            self.action_max = np.ones(self.action_dim)

        # Initialize data structures
        self.episode_files = []  # List of episode files
        self.total_episodes = 0

        # Initialize dataset
        self._initialize_dataset()

        # Print normalization info
        if self.normalize_actions:
            print(f"✓ Action normalization enabled for {self.DATASET_NAME}")
            print(f"  - Action min: {self.action_min}")
            print(f"  - Action max: {self.action_max}")
        else:
            print(f"✓ Action normalization disabled for {self.DATASET_NAME}")

    def _scan_episode_folder(self):
        """Scan episode folder to get all HDF5 files"""
        # Try two possible directory structures:
        # 1. root_dir/task_name/hdf5_folder/
        # 2. root_dir/task_name/ (if hdf5 files are directly in task folder)

        # Option 1: With hdf5_folder subdirectory
        task_dir = os.path.join(self.root_dir, self.task_name, self.hdf5_folder)
        if not os.path.exists(task_dir):
            # Option 2: Directly in task folder
            task_dir = os.path.join(self.root_dir, self.task_name)
            if not os.path.exists(task_dir):
                print(f"Warning: Task folder {task_dir} does not exist")
                return []

        print(f"Scanning for HDF5 files in: {task_dir}")

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

        print(f"Real robot 7D task {self.task_name}: Found {len(hdf5_files)} HDF5 files")
        return hdf5_files

    def _initialize_dataset(self):
        """Initialize dataset by scanning folder"""
        print("Initializing real robot 7D dataset...")

        # Scan task folder
        self.episode_files = self._scan_episode_folder()
        self.total_episodes = len(self.episode_files)

        if self.total_episodes == 0:
            raise ValueError("Error: No HDF5 files found, please check data path")

        print(f"Real robot 7D dataset initialized. Total {self.total_episodes} episodes")

    def __len__(self):
        """Return approximate dataset length"""
        return self.total_episodes * 200  # Assume 200 samples per HDF5 file
    
    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME

    def normalize_action(self, action):
        """
        Normalize action values from original range to [-1, 1]

        Args:
            action: Action array to normalize (single vector or batch data)

        Returns:
            Normalized action array with zero-range dimensions preserved as 0
        """
        if not self.normalize_actions:
            return action

        # Ensure action_min and action_max are numpy arrays
        if not isinstance(self.action_min, np.ndarray):
            self.action_min = np.array(self.action_min)
        if not isinstance(self.action_max, np.ndarray):
            self.action_max = np.array(self.action_max)

        # Handle zero-range dimensions
        action = np.array(action)
        action_range = self.action_max - self.action_min

        # For dimensions with zero range, return 0
        zero_range_mask = (action_range == 0)
        normalized_action = np.zeros_like(action)

        # Only normalize non-zero range dimensions
        non_zero_indices = np.where(~zero_range_mask)[0]

        if len(non_zero_indices) > 0:
            for idx in non_zero_indices:
                if action.ndim == 1:
                    # Single vector
                    normalized_action[idx] = (
                        2.0 * (action[idx] - self.action_min[idx]) / action_range[idx] - 1.0
                    )
                else:
                    # Batch data (shape: [batch_size, action_dim])
                    normalized_action[:, idx] = (
                        2.0 * (action[:, idx] - self.action_min[idx]) / action_range[idx] - 1.0
                    )

        return normalized_action

    def denormalize_action(self, normalized_action):
        """
        Denormalize action values from [-1, 1] back to original range

        Args:
            normalized_action: Normalized action array

        Returns:
            Denormalized action array
        """
        if not self.normalize_actions:
            return normalized_action

        # Ensure action_min and action_max are numpy arrays
        if not isinstance(self.action_min, np.ndarray):
            self.action_min = np.array(self.action_min)
        if not isinstance(self.action_max, np.ndarray):
            self.action_max = np.array(self.action_max)

        # Avoid division by zero
        action_range = self.action_max - self.action_min
        action_range = np.where(action_range == 0, 1.0, action_range)

        # Denormalize from [-1, 1]
        action = (normalized_action + 1.0) * 0.5 * action_range + self.action_min

        return action

    def parse_img_data(self, dataset, idx):
        """
        Process multi-camera image data

        Args:
            dataset: HDF5 dataset containing camera images
            idx: Current frame index

        Returns:
            Processed image sequence with shape [history_size, H, W, 3]
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
            print(f"[DEBUG] decode_image_with_opencv error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames], axis=0)

        return frames

    def decode_image_with_opencv(self, img_data):
        """
        Decode image data using OpenCV, maintaining RGB format

        Args:
            img_data (bytes): Binary image data

        Returns:
            np.ndarray: Decoded image array, shape=(360, 640, 3), RGB format
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

            # Resize to expected size if needed (640x360)
            if rgb_img.shape[:2] != (360, 640):
                rgb_img = cv2.resize(rgb_img, (640, 360))

            return rgb_img

        except Exception as e:
            print(f"[DEBUG] Image decoding failed: {e}")
            return None

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

    def extract_episode_item(self, hdf5_file):
        """
        Extract a single sample from HDF5 file for 7D single-arm data

        Args:
            hdf5_file: HDF5 file path

        Returns:
            Dictionary containing extracted data, or None if extraction fails
        """
        try:
            with h5py.File(hdf5_file, 'r', swmr=True, libver='latest') as f:
                # Load 7D single-arm action data
                try:
                    # Try different possible action data paths for 7D single-arm
                    if 'action' in f:
                        action_data = f["action"][:]
                    elif 'actions' in f:
                        action_data = f["actions"][:]
                    elif 'joint_action' in f:
                        # For single-arm data, extract only left_arm and left_gripper (7D total)
                        if 'left_arm' in f["joint_action"] and 'left_gripper' in f["joint_action"]:
                            left_arm = f["joint_action/left_arm"][:]  # 6D
                            left_gripper = f["joint_action/left_gripper"][:]  # 1D

                            # Handle dimension mismatch
                            if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                                left_gripper = left_gripper.reshape(-1, 1)
                            elif len(left_arm.shape) == 1 and len(left_gripper.shape) == 1:
                                left_arm = left_arm.reshape(-1, 6)
                                left_gripper = left_gripper.reshape(-1, 1)

                            # Concatenate to get 7D action [6D arm + 1D gripper]
                            action_data = np.concatenate([left_arm, left_gripper], axis=1)
                            #print(f"[DEBUG] Loaded 7D single-arm data: left_arm={left_arm.shape}, left_gripper={left_gripper.shape}, action_data={action_data.shape}")
                        else:
                            raise ValueError("Expected left_arm and left_gripper in joint_action for 7D single-arm data")
                    else:
                        raise ValueError(f"No valid action data found in {hdf5_file}")

                    # Ensure action data has correct dimensions (7D)
                    if len(action_data.shape) == 1:
                        action_data = action_data.reshape(-1, self.action_dim)
                    elif action_data.shape[1] != self.action_dim:
                        print(f"Warning: Action dimension mismatch. Expected {self.action_dim}, got {action_data.shape[1]}")
                        # If we have more than 7 dimensions, take only first 7
                        if action_data.shape[1] > self.action_dim:
                            action_data = action_data[:, :self.action_dim]
                            print(f"Truncated action data to first {self.action_dim} dimensions")
                        else:
                            # Adapt to the actual dimension
                            self.action_dim = action_data.shape[1]

                except Exception as e:
                    print(f"Error loading 7D action data: {e}")
                    return None

                # Adjust data indexing
                max_index = len(action_data) - 2
                index = random.randint(0, max_index)
                
                # Current state (using action data)
                action_current = action_data[index]

                # Future action sequence
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_chunk = action_data[index+1:action_end+1:self.upsample_rate]

                # If action sequence is insufficient, repeat last frame for padding
                if action_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(action_chunk[-1:], self.chunk_size - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, last_part], axis=0)

                # Apply action normalization if enabled
                if self.normalize_actions:
                    action_current = self.normalize_action(action_current)
                    action_chunk = self.normalize_action(action_chunk)

                # Load multi-camera data (same as 14D version)
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
                            print(f"Warning: Camera {cam_name} not found in {hdf5_file}")
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
                    print(f"Error accessing camera data in {hdf5_file}: {e}")
                    traceback.print_exc()
                    return None

                # Load pre-encoded language instruction
                language_embedding = self.load_language_embedding(hdf5_file)
                if language_embedding is None:
                    print(f"Warning: Failed to load language embedding for {hdf5_file}")
                    return None

                state_indicator = np.ones_like(action_current)
                action_norm = np.ones_like(action_chunk)

                # Create result dictionary
                result = {
                    "current_images": img_frames_np,  # Current frame images
                    "current_images_mask": current_images_mask,  # Image masks
                    "actions": action_chunk,  # 7D action sequence (already normalized if normalize_actions=True)
                    "states": np.expand_dims(action_current, axis=0),  # 7D states (already normalized if normalize_actions=True)
                    "state_indicator": state_indicator,
                    "action_norm": action_norm,
                    "instruction": language_embedding,  # Pre-encoded language instruction
                    "dataset_name": self.DATASET_NAME,  # Dataset name
                }

                return result

        except Exception as e:
            print(f"Error processing {hdf5_file}: {e}")
            traceback.print_exc()
            return None

    def get_item(self, index=None):
        """
        Get data item, randomly select one if index is None

        Args:
            index: Optional, specified index. If None, randomly select

        Returns:
            Processed data dictionary, or None if failed
        """
        # Randomly select an HDF5 file
        if not self.episode_files:
            self._initialize_dataset()

        if not self.episode_files:
            print("Error: No available episodes")
            return None

        episode_file = random.choice(self.episode_files)

        # Try to extract sample data
        for _ in range(3):  # Maximum 3 attempts
            item = self.extract_episode_item(episode_file)
            if item is not None:
                return item
            # If current sample extraction fails, randomly select another
            episode_file = random.choice(self.episode_files)

        print(f"Warning: Failed to extract sample, returning None")
        return None

if __name__ == "__main__":
    # Test code for 7D dataset
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Test RealRobotDataset7D')
    parser.add_argument('--config', type=str, default='/home/yinzhenhan/Human_Pre/H_RDT/configs/finetune_real_robot_7d.yaml', help='Config file path')
    parser.add_argument('--root_dir', type=str, default='/home/yinzhenhan/Human_Pre/dataset/zhenji', help='Root directory for real robot data')
    parser.add_argument('--task_name', type=str, default='single_arm_task', help='Task name')
    parser.add_argument('--hdf5_folder', type=str, default='data', help='HDF5 folder')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum episodes')
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Create dataset instance
    print("Creating real robot 7D dataset instance...")
    ds = RealRobotDataset7D(
        root_dir=args.root_dir,
        task_name=args.task_name,
        hdf5_folder=args.hdf5_folder,
        max_episodes=args.max_episodes,
        config=config,
        val=False
    )

    # Test data loading
    print("\nTesting 7D data loading...")
    success_count = 0
    test_times = 10

    for i in range(test_times):
        print(f"\nAttempt {i+1}/{test_times}...")
        item = ds.get_item()

        if item is not None:
            success_count += 1
            print("Successfully loaded 7D data")
            print(f"Instruction embedding shape: {item['instruction'].shape}")
            print(f"Camera images shape: {item['current_images'].shape}")
            print(f"Number of cameras: {len(item['current_images_mask'])}")
            print(f"Actions shape: {item['actions'].shape}")
            print(f"States shape: {item['states'].shape}")
            print(f"Action dimension: {item['actions'].shape[1]} (should be 7)")

    print(f"\nResult: Successfully loaded {success_count}/{test_times} 7D samples")