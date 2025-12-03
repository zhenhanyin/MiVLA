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


if not hasattr(np, 'bool'):
    np.bool = bool


class LiberoJointDataset:
    """
    Dataset for loading LIBERO robot data with joint actions.
    Supports both single-task and multi-task modes with balanced sampling.
    """
    def __init__(
        self,
        mode="multi_task",  # "single_task" or "multi_task"
        single_task_root_dir=None,  # For single task mode
        multi_task_root_dir=None,  # For multi task mode
        task_name=None,  # Required for single_task mode
        hdf5_folder=None,  # Required for single_task mode
        max_episodes=None,  # Maximum number of episodes to load for single task
        config=None,
        stat_path=None,
        upsample_rate=3,
        val=False,
        exclude_dirs=None  # List of directory names to exclude
    ):
        """
        Initialize LIBERO dataset with single/multi-task support.

        Args:
            mode: "single_task" or "multi_task"
            single_task_root_dir: Root directory for single task mode
            multi_task_root_dir: Root directory for multi task mode
            task_name: Task name for single task mode (e.g., "pick_and_place")
            hdf5_folder: HDF5 folder path for single task mode
            max_episodes: Maximum episodes to load for single task (None = all)
            config: Configuration dictionary
            stat_path: Path to normalization statistics file
            upsample_rate: Temporal data upsampling rate
            val: Whether this is validation set
            exclude_dirs: List of directory names to exclude from loading
        """
        self.DATASET_NAME = "libero_joint"
        self.mode = mode
        self.single_task_root_dir = single_task_root_dir
        self.multi_task_root_dir = multi_task_root_dir
        self.task_name = task_name
        self.hdf5_folder = hdf5_folder
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else []

        # Validate mode-specific parameters
        if mode == "single_task":
            if not task_name or not hdf5_folder:
                raise ValueError("single_task mode requires task_name and hdf5_folder parameters")
        elif mode == "multi_task":
            pass  # No additional validation needed
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'single_task' or 'multi_task'")

        # Set basic parameters
        self.chunk_size = config['common']['action_chunk_size']
        self.state_dim = config['common']['action_dim']
        self.img_history_size = config['common']['img_history_size']

        # Set camera parameters for LIBERO (2 cameras only)
        self.num_cameras = config['common']['num_cameras']
        if self.num_cameras == 2:
            self.cameras = ["agentview", "eye_in_hand"]
        else:
            raise ValueError(f"Unsupported num_cameras={self.num_cameras}, LIBERO only supports 2 cameras.")

        self.camera_mapping = {
            "agentview": "agentview_rgb",
            "eye_in_hand": "eye_in_hand_rgb"
        }

        # Set default stat_path if not provided (relative to this file)
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'stats.json')

        # Load statistics for normalization if file exists
        self.action_min = None
        self.action_max = None
        if os.path.exists(stat_path):
            with open(stat_path, 'r') as file:
                stat = json.load(file)
            if 'libero_joint' in stat:
                self.action_min = np.array(stat['libero_joint']['min'])
                self.action_max = np.array(stat['libero_joint']['max'])

        # Initialize data structures
        if mode == "single_task":
            self.episode_files = []  # List of HDF5 files for single task
        else:
            self.task_to_files = {}   # Task name -> HDF5 files mapping for multi task
            self.task_weights = {}     # Task sampling weights

        self.total_trajectories = 0  # Total number of trajectories

        # Initialize dataset
        self._initialize_dataset()

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

        task_to_files = {}

        for task_folder in task_folders:
            # Skip excluded directories
            if task_folder in self.exclude_dirs:
                print(f"Skipping excluded directory: {task_folder}")
                continue

            task_dir = os.path.join(self.multi_task_root_dir, task_folder)
            hdf5_files = []

            # Find all HDF5 files in task folder (not recursive for LIBERO)
            for file in os.listdir(task_dir):
                if file.endswith(".hdf5"):
                    hdf5_path = os.path.join(task_dir, file)
                    hdf5_files.append(hdf5_path)

            # Sort files for consistency
            hdf5_files.sort()
            task_to_files[task_folder] = hdf5_files
            print(f"Multi-task {task_folder}: Found {len(hdf5_files)} HDF5 files")

        return task_to_files

    def _initialize_dataset(self):
        """Initialize dataset by rescanning folders and updating sampling weights"""
        print("Initializing LIBERO dataset...")

        if self.mode == "single_task":
            # Single task mode: scan single task folder
            self.episode_files = self._scan_single_task_folder()
            self.total_trajectories = len(self.episode_files)

            if self.total_trajectories == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")

            print(f"Single task dataset initialized. Total {self.total_trajectories} files")

        else:
            # Multi task mode: scan all task folders
            self.task_to_files = self._scan_multi_task_folders()

            # Calculate total files
            all_task_count = sum(len(files) for files in self.task_to_files.values())

            if all_task_count == 0:
                raise ValueError("Error: No HDF5 files found, please check data path")

            # Calculate sampling weights - equal weight for all tasks (1:1:1:1:1:1 sampling)
            task_count = len(self.task_to_files)
            for task_name in self.task_to_files.keys():
                self.task_weights[task_name] = 1.0 / task_count

            self.total_trajectories = all_task_count

            print(f"Multi-task dataset initialized. Total {all_task_count} files across {task_count} tasks")
            print(f"Task weights: {self.task_weights}")

    def __len__(self):
        """Return approximate dataset length"""
        return self.total_trajectories * 100  # Assume 100 trajectories per HDF5 file

    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME

    def parse_img_data(self, dataset, idx):
        """
        Process LIBERO image data (already in numpy array format)

        Args:
            dataset: HDF5 dataset containing image data with shape [T, H, W, 3]
            idx: Current frame index

        Returns:
            Processed image sequence with shape [history_size, H, W, 3]
        """
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1

        # LIBERO images are already 128x128x3
        total_frames = len(dataset)
        frames = np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)

        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < total_frames:
                    img_data = dataset[frame_idx]
                    frames[i] = img_data  # No decoding needed for LIBERO

        except Exception as e:
            print(f"[DEBUG] Image loading error: {e}")

        if num_frames < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - num_frames, axis=0)
            frames = np.concatenate([pad_frames, frames])

        return frames

    def load_language_embedding(self, hdf5_file_path):
        """
        Load pre-encoded language instruction embedding from centralized lang_embeddings folder

        Args:
            hdf5_file_path (str): Path to HDF5 file

        Returns:
            torch.Tensor: Task instruction embedding, or None if not found
        """
        try:
            # Extract task name from HDF5 file path
            # For LIBERO: HDF5 file: /path/to/libero_10/task_demo.hdf5
            # Embedding file: /path/to/lang_embeddings/task_demo.pt
            task_name = None

            if self.mode == "single_task":
                # For single task mode, use the provided task_name
                task_name = self.task_name
            else:
                # For multi-task mode, extract task name from HDF5 filename
                # For LIBERO: use the HDF5 filename (without extension) as task name
                # Example: /path/to/libero_object/pick_up_the_tomato_sauce_demo.hdf5 -> pick_up_the_tomato_sauce_demo
                task_name = os.path.splitext(os.path.basename(hdf5_file_path))[0]

            if task_name is None:
                print(f"Warning: Could not extract task name from path {hdf5_file_path}")
                return None

            # Build embedding file path using relative path to current script location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_path = os.path.join(current_dir, 'lang_embeddings', f"{task_name}.pt")

            if not os.path.exists(embedding_path):
                print(f"Warning: Task embedding file not found: {embedding_path}")
                print(f"         HDF5 file: {hdf5_file_path}")
                print(f"         Extracted task name: {task_name}")
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

    def extract_trajectory_item(self, hdf5_file):
        """
        Extract a single sample from LIBERO HDF5 file

        Args:
            hdf5_file: HDF5 file path

        Returns:
            Dictionary containing extracted data, or None if extraction fails
        """
        try:
            with h5py.File(hdf5_file, 'r', swmr=True, libver='latest') as f:
                # Get all demonstrations in the file
                if 'data' not in f:
                    print(f"Warning: No 'data' group found in {hdf5_file}")
                    return None

                data_group = f['data']
                demo_names = sorted(list(data_group.keys()))

                if not demo_names:
                    print(f"Warning: No demonstrations found in {hdf5_file}")
                    return None

                # Randomly select a demonstration
                demo_name = random.choice(demo_names)
                demo_group = data_group[demo_name]

                # Load joint states and gripper states for LIBERO
                try:
                    joint_states = demo_group["obs/joint_states"][:]      # (T, 7)
                    gripper_states = demo_group["obs/gripper_states"][:]  # (T, 2)

                    # Use both dimensions of gripper states for 9D total
                    gripper_control = gripper_states[:, 0:2]  # (T, 2)

                    # Concatenate for 9D action vector
                    action_data = np.concatenate([joint_states, gripper_control], axis=1)  # (T, 9)

                except Exception as e:
                    print(f"Error loading action data from {hdf5_file}: {e}")
                    return None

                # Adjust data indexing
                max_index = len(action_data) - 2
                index = random.randint(0, max_index)

                # Current state
                action_current = action_data[index]

                # Future action sequence
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_chunk = action_data[index+1:action_end+1:self.upsample_rate]

                # If action sequence is insufficient, repeat last frame for padding
                if action_chunk.shape[0] < self.chunk_size:
                    last_part = np.repeat(action_chunk[-1:], self.chunk_size - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, last_part], axis=0)

                # Get LIBERO camera data (2 cameras only)
                try:
                    current_images = []

                    # Define camera paths for LIBERO 2-view setup
                    camera_paths = {
                        "agentview": "obs/agentview_rgb",
                        "eye_in_hand": "obs/eye_in_hand_rgb"
                    }

                    # Load images from each configured camera (agentview first, then eye_in_hand)
                    for cam_idx, cam_name in enumerate(self.cameras):
                        cam_path = camera_paths.get(cam_name)
                        if cam_path and cam_path in demo_group:
                            camera_data = demo_group[cam_path]
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
                    "current_images": img_frames_np,  # Current frame images (possibly augmented)
                    "current_images_mask": current_images_mask,  # Image masks
                    "actions": action_chunk,  # Action sequence
                    "states": np.expand_dims(action_current, axis=0),  # States
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
        if self.mode == "single_task":
            # Single task mode: randomly select from HDF5 files
            if not self.episode_files:
                self._initialize_dataset()

            if not self.episode_files:
                print("Error: No available files")
                return None

            # Randomly select an HDF5 file
            hdf5_file = random.choice(self.episode_files)

        else:
            # Multi task mode: balanced task sampling
            if not self.task_to_files:
                self._initialize_dataset()

            # Randomly select a task based on task weights
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1
            )[0]

            # Randomly select a sample from the selected task
            task_files = self.task_to_files.get(task_name, [])
            if not task_files:
                print(f"Warning: Task {task_name} has no available samples")
                # Select from other tasks
                alternative_tasks = [t for t in self.task_to_files.keys() if t != task_name and self.task_to_files.get(t, [])]
                if not alternative_tasks:
                    print("Error: No available samples")
                    return None
                task_name = random.choice(alternative_tasks)
                task_files = self.task_to_files.get(task_name, [])

            # Randomly select an HDF5 file
            hdf5_file = random.choice(task_files)

        # Try to extract sample data
        for _ in range(3):  # Maximum 3 attempts
            item = self.extract_trajectory_item(hdf5_file)
            if item is not None:
                return item
            # If current sample extraction fails, randomly select another
            if self.mode == "single_task":
                hdf5_file = random.choice(self.episode_files)
            else:
                task_files = self.task_to_files.get(task_name, [])
                if task_files:
                    hdf5_file = random.choice(task_files)

        print(f"Warning: Failed to extract sample, returning None")
        return None


if __name__ == "__main__":
    # Test code
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Test LiberoJointDataset')
    parser.add_argument('--config', type=str, default='/share/hongzhe/VLA/round2/dino_siglip/configs/hrdt.yaml', help='Config file path')
    parser.add_argument('--mode', type=str, default='multi_task', choices=['single_task', 'multi_task'], help='Dataset mode')
    parser.add_argument('--task_name', type=str, default='pick_and_place', help='Task name for single task mode')
    parser.add_argument('--hdf5_folder', type=str, default='', help='HDF5 folder for single task mode')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum episodes for single task mode')
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Create dataset instance
    print("Creating LIBERO dataset instance...")
    if args.mode == "single_task":
        ds = LiberoJointDataset(
            mode="single_task",
            task_name=args.task_name,
            hdf5_folder=args.hdf5_folder,
            max_episodes=args.max_episodes,
            config=config,
            val=False
        )
    else:
        ds = LiberoJointDataset(
            mode="multi_task",
            multi_task_root_dir="/home/yinzhenhan/LIBERO/datasets",
            config=config,
            val=False
        )

    # Test data loading
    print("\nTesting LIBERO data loading...")
    success_count = 0
    test_times = 10

    for i in range(test_times):
        print(f"\nAttempt {i+1}/{test_times}...")
        item = ds.get_item()

        if item is not None:
            success_count += 1
            print("Successfully loaded data")
            print(f"Instruction embedding shape: {item['instruction'].shape}")
            print(f"Multi-view images shape: {item['current_images'].shape}")
            print(f"Number of cameras: {len(item['current_images_mask'])}")
            print(f"Actions shape: {item['actions'].shape}")
            print(f"States shape: {item['states'].shape}")

    print(f"\nResult: Successfully loaded {success_count}/{test_times} samples")