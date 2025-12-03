import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from utils.image_corrupt import image_corrupt
from datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
from datasets.robotwin2.robotwin_agilex_62D_with_noise_dataset import RobotwinAgilex62DWithNoiseDataset
from datasets.robotwin2.robotwin_agilex_76D_with_noise_dataset import RobotwinAgilex76DWithNoiseDataset
from datasets.pretrain.egodex_dataset import EgoDexDataset
from datasets.real_robot.real_robot_dataset import RealRobotDataset
from datasets.real_robot.real_robot_dataset_7d import RealRobotDataset7D
from datasets.real_robot.real_robot_dataset_eef_16d import RealRobotDatasetEEF16D
from datasets.real_robot.real_robot_dataset_76d_joint_with_noise import RealRobotDataset76dJointWithNoise
from datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset
from datasets.co_pretrain.egodex_robotwin_copretrain_dataset import EgoDexRobotwinCopretrainDataset
from datasets.co_pretrain_v2.egodex_robotwin_copretrain_dataset_V2 import EgoDexRobotwinCopretrainDatasetV2
from datasets.libero.libero_joint_dataset import LiberoJointDataset
from datasets.libero.libero_osc_pose_7d_dataset import LiberoOSCPose7DDataset
import h5py
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(
        self,
        config,
        image_transform,
        num_cameras,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        image_corrupt_severity=None,
        dataset_type=None,
        state_noise_snr=None,
        use_precomp_lang_embed=True,
        upsample_rate=None,
        val=False,
        task_name=None,
        dataset_name="robotwin_agilex",  # Add dataset_name parameter
        normalize_actions=False,  # action normalization parameter
        action_stats_file=None,   # custom stats file parameter
        egodex_weight=0.85,     # EgoDex sampling weight for co-pretrain
        balance_datasets=False,    # Enable balanced sampling for co-pretrain
        noise_strategy="gaussian", # Noise strategy for 62D finetuning
        args=None,  # Command line arguments for additional parameters
    ):
        super(VLAConsumerDataset, self).__init__()
        self.dataset_name = dataset_name
        DATASET_NAMES = {self.dataset_name}
        
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = config["common"]["img_history_size"]
        self.image_transform = image_transform   

        # Initialize dataset based on dataset_name
        if self.dataset_name == "egodex":
            self.hdf5_dataset = EgoDexDataset(
                config=config,
                upsample_rate=upsample_rate,
                val=val,
                use_precomp_lang_embed=use_precomp_lang_embed,
                # Note: 使用新的预训练数据目录
                data_root="/home/yinzhenhan/Human_Pre/dataset/Egodex-full-pretrain",
                stat_path="/home/yinzhenhan/Human_Pre/H_RDT/datasets/pretrain/egodex_stat_62d.json",
            )
        elif self.dataset_name == "robotwin_agilex":
            
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="multi_task",
                config=config,
                # Note: override default paths
                multi_task_root_dir="/home/yinzhenhan/RoboTwin/data/dataset",
            )
        
            # self.hdf5_dataset = RobotwinAgilexDataset(
            #     mode="single_task",
            #     task_name=task_name,
            #     hdf5_folder="data",       #"Aloha-AgileX/data"    
            #     max_episodes=50,
            #     config=config,
            #     # Note: override default paths   单个任务根目录
            #     single_task_root_dir="/home/yinzhenhan/Human_Pre/H_RDT/datasets/robotwin2/single_task_clean",
            # )
            
        elif self.dataset_name == "robotwin_agilex_62D_with_noise":
            # 62D微调版本：48D噪声 + 14D真实数据
            # 用于保留预训练中学到的人类先验知识
            self.hdf5_dataset = RobotwinAgilex62DWithNoiseDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder="data",
                max_episodes=50,
                config=config,
                # Note: override default paths   单个任务根目录
                single_task_root_dir="/home/yinzhenhan/Human_Pre/H_RDT/datasets/robotwin2/single_task_clean",
                # 噪声参数
                noise_strategy=noise_strategy,  # 从命令行参数传递
                noise_scale=1.0,
                # 人类数据统计信息（如果有的话）
                human_data_stats=None,  # 可以加载预训练的统计信息
            )

        elif self.dataset_name == "robotwin_agilex_76D_with_noise":
            # 76D微调版本：48D噪声 + 14D双臂关节 + 14D末端位姿（噪声/真实）
            # 用于76D模型的机器人动作微调
            self.hdf5_dataset = RobotwinAgilex76DWithNoiseDataset(
                            mode="multi_task",
                            #data_type="randomized",
                            # task_name=task_name,
                            hdf5_folder=None,
                            max_episodes=50,
                            config=config,
                            # Note: override default paths   多任务根目录
                            multi_task_root_dir = "/home/yinzhenhan/RoboTwin/data/dataset",
                            # 噪声参数
                            noise_strategy=noise_strategy,  # 从命令行参数传递
                            noise_scale=1.0,
                            # 人类数据统计信息（如果有的话）
                            human_data_stats=None,  # 可以加载预训练的统计信息
                            # 机器人选择参数
                            robot_choice=getattr(args, 'robot_choice', None),  # 从命令行参数传递
                        )



            
        elif self.dataset_name == "real_robot_14d":
            # 真机14D数据集
            self.hdf5_dataset = RealRobotDataset(
                root_dir="/home/yinzhenhan/Human_Pre/dataset/zhenji",  # 真机数据根目录
                task_name=task_name,  # 任务名称，如 "pick_trash_exp"     将临时变量传给
                hdf5_folder="data",  # 指定在 root_dir 根目录下的 HDF5 数据文件夹名称
                max_episodes=None,  # 使用所有可用数据
                config=config,
                val=val,
                stat_path=action_stats_file,  # 传递自定义统计文件路径
                normalize_actions=normalize_actions,  # 传递归一化标志
                dataset_name=self.dataset_name  # 传递数据集名称
            )
        elif self.dataset_name == "real_robot_7d":
            # 真机7D单臂数据集
            self.hdf5_dataset = RealRobotDataset7D(
                root_dir="/home/yinzhenhan/Human_Pre/dataset/zhenji",  # 真机数据根目录
                task_name=task_name,  # 任务名称，如 "single_arm_task"
                hdf5_folder="data",  # 指定在 root_dir 根目录下的 HDF5 数据文件夹名称
                max_episodes=None,  # 使用所有可用数据
                config=config,
                val=val,
                stat_path=action_stats_file,  # 传递自定义统计文件路径
                normalize_actions=normalize_actions,  # 传递归一化标志
                dataset_name=self.dataset_name  # 传递数据集名称
            )
        elif self.dataset_name == "real_robot_eef_16d":
            # 真机16维末端执行器数据集
            # 16D结构: [左臂末端位姿7维 + 左臂夹爪1维 + 右臂末端位姿7维 + 右臂夹爪1维]
            self.hdf5_dataset = RealRobotDatasetEEF16D(
                root_dir="/home/yinzhenhan/Human_Pre/dataset/zhenji",  # 真机数据根目录
                task_name=task_name,  # 任务名称，如 "pick_trash_exp"
                hdf5_folder="data",  # 指定在 root_dir 根目录下的 HDF5 数据文件夹名称
                max_episodes=None,  # 使用所有可用数据
                config=config,
                val=val,
                stat_path=action_stats_file,  # 传递自定义统计文件路径
                normalize_actions=normalize_actions,  # 传递归一化标志
                dataset_name=self.dataset_name  # 传递数据集名称
            )

        elif self.dataset_name == "egodex_robotwin_copretrain":
            # 协同预训练数据集
            self.hdf5_dataset = EgoDexRobotwinCopretrainDataset(
                dataset_dir="/home/yinzhenhan/Human_Pre",
                chunk_size=config["common"]["action_chunk_size"],
                img_history_size=config["common"]["img_history_size"],
                upsample_rate=upsample_rate,
                egodex_weight=egodex_weight,
                balance_datasets=balance_datasets,
                stats_file=action_stats_file if action_stats_file else "/home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain/co_stat_62d.json",
                image_aug=image_aug,
                image_aug_threshold=0.1,
            )
        elif self.dataset_name == "egodex_robotwin_copretrain_V2":
            # 76D协同预训练数据集V2
            final_stats_file = action_stats_file if action_stats_file else "/home/yinzhenhan/Human_Pre/H_RDT/datasets/co_pretrain_v2/co_stat_76d.json"

            # 打印action_stats_file加载信息
            print(f"🎯 [76D V2 Dataset] Action Stats File Configuration:")
            print(f"   - Command line action_stats_file: {action_stats_file}")
            print(f"   - Final stats_file path: {final_stats_file}")


            self.hdf5_dataset = EgoDexRobotwinCopretrainDatasetV2(
                dataset_dir="/home/yinzhenhan/Human_Pre",
                chunk_size=config["common"]["action_chunk_size"],
                img_history_size=config["common"]["img_history_size"],
                upsample_rate=upsample_rate,
                egodex_weight=egodex_weight,
                balance_datasets=balance_datasets,
                stats_file=final_stats_file,
            )

        elif self.dataset_name == "real_robot_with_noise":
            # 真机62D微调数据集：48D噪声 + 14D真实数据
            from datasets.real_robot.real_robot_with_noise_dataset import RealRobotWithNoiseDataset
            self.hdf5_dataset = RealRobotWithNoiseDataset(
                root_dir="/home/yinzhenhan/Human_Pre/dataset/zhenji",
                task_name=task_name,
                hdf5_folder="data",
                config=config,
                normalize_actions=normalize_actions,
                stat_path=action_stats_file,
                upsample_rate=upsample_rate,
                noise_strategy=noise_strategy,  # 从命令行参数传递
                noise_scale=1.0,          # 噪声缩放
                human_data_stats=None     # 人类数据统计信息
            )

        elif self.dataset_name == "real_robot_76d_joint_with_noise":
            # 真机76D微调数据集：48D人类噪声 + 14D真实关节 + 14D末端位姿噪声
            self.hdf5_dataset = RealRobotDataset76dJointWithNoise(
                root_dir="/home/yinzhenhan/Human_Pre/dataset/zhenji",
                task_name=task_name,
                hdf5_folder="",  # HDF5文件直接在任务目录下
                config=config,
                normalize_actions=normalize_actions,
                stat_path=action_stats_file,
                upsample_rate=upsample_rate,
                val=val,
                image_corrupt_severity=image_corrupt_severity,
                noise_strategy=noise_strategy,  # 从命令行参数传递
                noise_scale=1.0,              # 人类动作噪声缩放
                end_effector_noise_scale=1.0, # 末端位姿噪声缩放
                human_data_stats=None         # 人类数据统计信息
            )

        elif self.dataset_name == "libero_joint":
            # LIBERO关节动作数据集：7个关节角度 + 1个夹爪控制 (9D)
            self.hdf5_dataset = LiberoJointDataset(
                mode="multi_task",
                multi_task_root_dir="/home/yinzhenhan/LIBERO/datasets",
                config=config,
                val=val,
                upsample_rate=upsample_rate,
                exclude_dirs=getattr(args, 'exclude_libero_dirs', None) if args else None
            )

        elif self.dataset_name == "libero_osc_pose_7d":
            # LIBERO OSC位姿数据集：ee_states(6D) + gripper_states[0](1D) = 7D状态，actions(7D)动作
            self.hdf5_dataset = LiberoOSCPose7DDataset(
                mode="multi_task",
                multi_task_root_dir="/home/yinzhenhan/LIBERO/datasets",
                config=config,
                val=val,
                upsample_rate=upsample_rate,
                exclude_dirs=getattr(args, 'exclude_libero_dirs', None) if args else None
            )

        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
            
        print(f"Initialized dataset: {self.dataset_name}")

        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.dataset_type = dataset_type

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        # self.image_aug_transform = get_image_augmentation()
        self.image_aug = image_aug

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        # Get data from backend dataset
        try:
            res = self.hdf5_dataset.get_item(index)
        except Exception as e:
            print(f"Error loading episode {index}: {e}")
            return None
            
        # Add check for res being None, retry a few times if it's None
        retry_count = 0
        max_retries = 5
        while res is None and retry_count < max_retries:
            retry_count += 1
            print(f"Got None data item, retrying {retry_count} time...")
            try:
                res = self.hdf5_dataset.get_item(index)
            except Exception as e:
                print(f"Error during retry data loading: {e}")
                
        # If still None after multiple retries, return a default value to prevent training interruption
        if res is None:
            print(f"Warning: Still unable to get valid data after multiple retries, returning default value")

        data_dict = {}
        data_dict['dataset_name'] = res['dataset_name']
        data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]

        # Process state and action data
        data_dict["states"] = res['states']
        data_dict["actions"] = res['actions']
        data_dict["action_norm"] = res['action_norm']

        # Process images
        if self.dataset_name in ['egodex', 'egodex_robotwin_copretrain', 'egodex_robotwin_copretrain_V2']:
            # Single camera / stitched image processing
            image_metas = []
            images = res['current_images'][0]
            valid_mask = res.get('current_images_mask', [np.ones(self.img_history_size, dtype=bool)])[0]
            image_metas.append((images, valid_mask))

            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                images, valid_mask = image_metas[0]
                if valid_mask[hist_idx]:
                    rearranged_images.append((images[hist_idx], True))
                else:
                    rearranged_images.append((None, False))
        else:
            # Multi-view processing (original logic)
            image_metas = []
            for cam_idx in range(self.num_cameras):
                images = res['current_images'][cam_idx]
                valid_mask = res.get('current_images_mask', np.ones((self.num_cameras, self.img_history_size), dtype=bool))[cam_idx]
                image_metas.append((images, valid_mask))

            rearranged_images = []
            for hist_idx in range(self.img_history_size):
                for cam_idx in range(self.num_cameras):
                    images, valid_mask = image_metas[cam_idx]
                    if valid_mask[hist_idx]:
                        rearranged_images.append((images[hist_idx], True))
                    else:
                        rearranged_images.append((None, False))

        all_pixel_values = []
        for image, valid in rearranged_images:
            image = Image.fromarray(image) if image is not None else None

            if valid and self.auto_adjust_image_brightness:
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

            # Only apply image augmentation to 50% of the images
            if valid and self.image_aug and (random.random() > 0.5):
                aug_type = random.choice([
                    "corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)
                # image = self.image_aug_transform(image)

            pixel_values = self.image_transform(image)
            all_pixel_values.append(pixel_values)

        # Process dino-siglip format images
        pv_example = all_pixel_values[0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in all_pixel_values]
            )
            for k in pv_example
        }
        data_dict["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            # All datasets should provide lang_embeds as tensor
            if "lang_embeds" in res:
                data_dict["lang_embeds"] = res["lang_embeds"]
            elif torch.is_tensor(res["instruction"]):
                data_dict["lang_embeds"] = res["instruction"]
            else:
                # Legacy: load from file path
                data_dict["lang_embeds"] = torch.load(res["instruction"])["embeddings"].squeeze(0)

        # Convert all numpy arrays to torch tensors
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        # Verify all data is tensors
        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

        return data_dict

class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, use_precomp_lang_embed=True) -> None:
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "images": [],
            "data_indices": [],
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # Process images
            batch["images"].append(instance["images"])
            batch["data_indices"].append(instance["data_idx"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # Process dino-siglip format images
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        batch["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
        
        # 添加文件信息（用于损失尖峰侦测）
        if "file_info" in instances[0]:
            batch["file_info"] = [instance["file_info"] for instance in instances]

        return batch