import os
import sys
import subprocess
import numpy as np
import torch
import pandas as pd
from PIL import Image
import yaml

# 固定PyTorch随机种子以确保可重现性
def set_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置更高的浮点精度以减少GPU间差异
    torch.set_default_dtype(torch.float32)
    np.random.seed(seed)  # 也固定numpy种子以保持一致性
import json
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
import safetensors.torch
import cv2

current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)
parent_folder_path = os.path.dirname(os.path.dirname(current_folder_path))
sys.path.append(parent_folder_path)
sys.path.append(current_folder_path)

from models.hrdt_runner import HRDTRunner
from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone


class MyModel:
    def __init__(self, ckpt_folder, task_name, use_62d_model=False, use_12_2=False, noise_strategy="gaussian", noise_scale=1.0, inv_norm=False, use_76d_model=False, use_76d_selective_14d=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        
        # 模型参数处理
        # 修复字符串转布尔值问题
        if isinstance(use_62d_model, str):
            use_62d_model = use_62d_model.lower() in ['true', '1', 'yes', 'on']
        if isinstance(use_12_2, str):
            use_12_2 = use_12_2.lower() in ['true', '1', 'yes', 'on']
        if isinstance(inv_norm, str):
            inv_norm = inv_norm.lower() in ['true', '1', 'yes', 'on']
        if isinstance(use_76d_model, str):
            use_76d_model = use_76d_model.lower() in ['true', '1', 'yes', 'on']
        if isinstance(use_76d_selective_14d, str):
            use_76d_selective_14d = use_76d_selective_14d.lower() in ['true', '1', 'yes', 'on']

        self.use_62d_model = use_62d_model
        self.use_12_2 = use_12_2  # 动作顺序控制
        self.use_76d_model = use_76d_model  # 76D模型标志
        self.use_76d_selective_14d = use_76d_selective_14d  # 76D selective 14D模型标志
        self.noise_strategy = noise_strategy
        self.noise_scale = noise_scale
        self.inv_norm = inv_norm  # 是否执行反归一化

        # 确定模型维度
        if use_76d_model:
            self.state_dim = 76
            self.action_dim = 76
        elif use_76d_selective_14d:
            self.state_dim = 14  # 76D selective模型只使用14D输入
            self.action_dim = 14
        elif use_62d_model:
            self.state_dim = 62
            self.action_dim = 62
        else:
            self.state_dim = 14
            self.action_dim = 14
        
        # Load config based on model type
        if self.use_76d_model:
            config_file_path = os.path.join(current_folder_path, 'utils', 'hrdt_76d.yaml')
        elif self.use_76d_selective_14d:
            config_file_path = os.path.join(current_folder_path, 'utils', 'hrdt_14d.yaml')  # 76D selective模型使用14D配置
        elif self.use_62d_model:
            config_file_path = os.path.join(current_folder_path, 'utils', 'hrdt.yaml')
        else:
            config_file_path = os.path.join(current_folder_path, 'utils', 'hrdt_14d.yaml')
        
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        # 固定PyTorch随机种子以确保可重现性
        set_torch_seed(42)

        # Load action normalization stats
        # Note: In H-RDT practice, we do not perform action normalization
        stat_file_path = os.path.join(current_folder_path, 'utils', 'stats.json')
        with open(stat_file_path, 'r') as file:
            stat = json.load(file)
        self.action_min = np.array(stat['robotwin_agilex']['min'])
        self.action_max = np.array(stat['robotwin_agilex']['max'])

        ckpt_file = os.path.join(ckpt_folder, 'pytorch_model.bin')
                
        # Load vision encoder
        vision_backbone_path = os.path.join(current_folder_path, 'bak', 'dino-siglip')
        self.vision_encoder = DinoSigLIPViTBackbone(
            vision_backbone_id="dino-siglip",
            image_resize_strategy="letterbox" 
                if self.config["dataset"]["image_aspect_ratio"] == "pad" 
                else "resize-naive",
            default_image_size=384
        )
        self.vision_encoder.to(self.device, dtype=self.dtype)
        self.vision_encoder.eval()
        self.image_transform = self.vision_encoder.get_image_transform()
        
        config_state_dim = self.config["common"]["state_dim"]
        config_action_dim = self.config["common"]["action_dim"]
        pred_horizon = self.config["common"]["action_chunk_size"]
        
        # Create H-RDT model with specified training mode
        if self.use_76d_selective_14d:
            # Import the 76D selective 14D runner
            from models.hrdt_runner_76d_selective_14d_joint import HRDTRunner76Dselective14DJoint

            # For 76D selective 14D model, we need to override config to use 76D pretrained weights
            config_76d_path = os.path.join(current_folder_path, 'utils', 'hrdt_76d.yaml')
            with open(config_76d_path, 'r') as config_file:
                config_76d = yaml.safe_load(config_file)

            # Use 76D pretrained weights but 14D model
            self.policy = HRDTRunner76Dselective14DJoint(
                state_dim=14,  # 14D input
                action_dim=14,  # 14D output
                pred_horizon=pred_horizon,
                config=config_76d["model"],  # Use 76D config for pretrained weights
                act_pos_emb_config=[
                    ("state", 1),
                    ("action", pred_horizon),
                ],
                img_pos_emb_config=[
                    ("image", (config_76d["common"]["img_history_size"],
                              config_76d["common"]["num_cameras"],
                              -self.vision_encoder.num_patches)),
                ],
                lang_pos_emb_config=[
                    ("lang", -config_76d["dataset"]["tokenizer_max_length"]),
                ],
                max_img_len=config_76d["common"]["img_history_size"] * config_76d["common"]["num_cameras"] * self.vision_encoder.num_patches,
                max_lang_len=config_76d["dataset"]["tokenizer_max_length"],
                training_mode='lang',
                mode='finetune_76d_selective_14d_joint',  # Special mode for 76D selective loading
                pretrained_backbone_path=ckpt_folder,  # 76D pretrained weights
                dtype=self.dtype,
            )
        else:
            # Standard H-RDT model loading
            self.policy = HRDTRunner.from_pretrained(
                pretrained_model_name_or_path=ckpt_folder,
                state_dim=config_state_dim,
                action_dim=config_action_dim,
                pred_horizon=pred_horizon,
                config=self.config["model"],
                act_pos_emb_config=[
                    ("state", 1),
                    ("action", pred_horizon),
                ],
                img_pos_emb_config=[
                    ("image", (self.config["common"]["img_history_size"],
                              self.config["common"]["num_cameras"],
                              -self.vision_encoder.num_patches)),
                ],
                lang_pos_emb_config=[
                    ("lang", -self.config["dataset"]["tokenizer_max_length"]),
                ],
                max_img_len=self.config["common"]["img_history_size"] * self.config["common"]["num_cameras"] * self.vision_encoder.num_patches,
                max_lang_len=self.config["dataset"]["tokenizer_max_length"],
                dtype=self.dtype,
            )

        self.policy.to(self.device, dtype=self.dtype).eval()
        
        # Initialize image cache
        self.img_cache = []
        self.max_img_cache_size = self.config['common']['img_history_size']

        # Load embeddings based on training mode
        self.task_name = task_name
        self.lang_tokens = None
        self.lang_attn_mask = None
        
        # Load pre-encoded language embeddings
        lang_embed_path = os.path.join(current_folder_path, 'utils', 'lang_embeddings', f'{task_name}.pt')
        if os.path.exists(lang_embed_path):
            # Load embedding data (it's a dictionary with 'embeddings' key)
            embedding_data = torch.load(lang_embed_path, map_location=self.device)
                
            # Extract embeddings tensor from dictionary
            embeddings = embedding_data.get('embeddings', None)
            if embeddings is None:
                self.lang_tokens = None
            else:
                # Remove batch dimension if present (convert from 3D to 2D)
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(0)
                    
                self.lang_tokens = embeddings.to(dtype=self.dtype)
                # Create attention mask (all tokens are valid)
                self.lang_attn_mask = torch.ones(self.lang_tokens.shape[:1], dtype=torch.bool, device=self.device)

    def generate_human_noise(self, batch_size=1):
        """生成48D人类动作噪声"""
        if self.noise_strategy == "gaussian":
            return torch.randn(batch_size, 48) * self.noise_scale
        elif self.noise_strategy == "uniform":
            return torch.rand(batch_size, 48) * 2 - 1
        else:  # zero
            return torch.zeros(batch_size, 48)

    def generate_end_effector_noise(self, batch_size=1):
        """生成14D末端位姿噪声"""
        if self.noise_strategy == "gaussian":
            return torch.randn(batch_size, 14) * self.noise_scale
        elif self.noise_strategy == "uniform":
            return torch.rand(batch_size, 14) * 2 - 1
        else:  # zero
            return torch.zeros(batch_size, 14)

    def inverse_normalize_action(self, normalized_action):
        """反归一化动作：将[-1,1]范围的归一化动作转换回真实物理动作范围
        
        Args:
            normalized_action: 归一化动作，范围[-1,1]
            
        Returns:
            反归一化后的真实动作
        """
        # 确保输入是numpy数组
        if isinstance(normalized_action, torch.Tensor):
            normalized_action = normalized_action.cpu().numpy()
            
        # 反归一化公式: raw_action = ((normalized_action + 1) / 2) * (max_action - min_action) + min_action
        # 这里我们假设归一化到[-1,1]范围，如果实际归一化公式不同需要调整
        if self.inv_norm and self.use_62d_model:
            # 只对后14维机器人动作进行反归一化
            if len(normalized_action.shape) == 1:
                # 单个动作向量
                robot_action = normalized_action
                # 假设原始归一化是到[-1,1]范围的反归一化
                # 如果实际归一化方式不同，需要调整这个公式
                action_range = self.action_max - self.action_min
                robot_action = ((robot_action + 1) / 2) * action_range + self.action_min
                return robot_action
            else:
                # 动作序列
                robot_actions = normalized_action
                action_range = self.action_max - self.action_min
                robot_actions = ((robot_actions + 1) / 2) * action_range + self.action_min
                return robot_actions
        else:
            # 不执行反归一化，直接返回
            return normalized_action

    def update_obs(self, observation):
        self.img_cache.append(observation)
        if len(self.img_cache) > self.max_img_cache_size:
            self.img_cache.pop(0)

    @torch.no_grad()
    def get_action(self):
        if len(self.img_cache) == 0:
            return []
        
        current_obs = self.img_cache[-1]
        
        # Process state tokens - 根据模型训练格式处理关节状态
        state_tokens = None
        if 'agent_pos' in current_obs:
            original_pos = current_obs['agent_pos']  # 14维向量
            
            if self.use_12_2:

                reordered_pos = np.concatenate([
                    original_pos[0:6],     # 左臂6关节 (索引0-5)
                    original_pos[7:13],    # 右臂6关节 (索引7-12)
                    [original_pos[6]],    # 左夹爪 (索引6)
                    [original_pos[13]]    # 右夹爪 (索引13)
                ])
            else:
                # 1模型使用[左臂6, 左夹爪, 右臂6, 右夹爪]训练，无需重排输入
                reordered_pos = original_pos
            
            # 根据模型类型构造状态数据
            if self.use_76d_model:
                # 76D模型: 48D人类噪声 + 14D真实数据 + 14D末端位姿噪声
                human_noise = self.generate_human_noise(batch_size=1).numpy().squeeze(0)  # [48]
                eef_noise = self.generate_end_effector_noise(batch_size=1).numpy().squeeze(0)  # [14]

                # 构造76D状态数据：48D噪声 + 14D真实 + 14D噪声
                state_76d = np.concatenate([
                    human_noise,           # 48D人类噪声 [0-47]
                    reordered_pos,         # 14D真实数据 [48-61]
                    eef_noise             # 14D末端位姿噪声 [62-75]
                ])  # [76]

                state_tokens = torch.tensor(
                    state_76d,
                ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)
            elif self.use_76d_selective_14d:
                # 76D selective 14D模型: 直接使用14D真实数据
                state_tokens = torch.tensor(
                    reordered_pos,  # 14D真实数据
                ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)
            elif self.use_62d_model:
                # 62D模型: 48D人类噪声 + 14D真实数据
                human_noise = self.generate_human_noise(batch_size=1).numpy().squeeze(0)  # [48]

                # 构造62D状态数据：48D噪声 + 14D真实
                state_62d = np.concatenate([
                    human_noise,           # 48D噪声
                    reordered_pos          # 14D真实数据
                ])  # [62]

                state_tokens = torch.tensor(
                    state_62d,
                ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)
            else:
                # 14D模型处理
                state_tokens = torch.tensor(
                    reordered_pos,
                ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)

        # Process image tokens from separate camera views
        image_tokens = None
        if all(key in current_obs for key in ['head_cam', 'left_cam', 'right_cam']):
            camera_images = [
                current_obs['head_cam'],
                current_obs['right_cam'],
                current_obs['left_cam'],
            ]
            
            # Process images with dino-siglip encoding
            img_transform_list = []
            for img_array in camera_images:
                img = Image.fromarray(img_array)
                transformed = self.image_transform(img)
                img_transform_list.append(transformed)
            
            # Organize image data structure - process according to training code format
            # From training code, image data should be a dictionary containing dino and siglip keys
            image_inputs = {}
            pv_example = img_transform_list[0]
            for k in pv_example.keys():
                # Ensure conversion to same data type as model (bfloat16)
                image_inputs[k] = torch.stack([img[k] for img in img_transform_list], dim=0).unsqueeze(0).to(self.device, dtype=self.dtype)
            
            # Use vision encoder
            with torch.no_grad():
                # Process batch_size, sequence_length, channels, height, width
                k = next(iter(image_inputs))
                batch_size, seq_len, C, H, W = image_inputs[k].shape
                # Reshape to (batch_size * seq_len, C, H, W) to fit encoder input
                for k in image_inputs:
                    image_inputs[k] = image_inputs[k].view(-1, C, H, W)
                # Use vision encoder to get features
                image_features = self.vision_encoder(image_inputs)
                # Reshape features to correct dimensions
                image_tokens = image_features.view((batch_size, -1, self.vision_encoder.embed_dim))

        # Prepare inputs based on training mode
        lang_tokens = None
        lang_attn_mask = None
        
        lang_tokens = self.lang_tokens.unsqueeze(0)  # Add batch dimension
        lang_attn_mask = self.lang_attn_mask.unsqueeze(0) if self.lang_attn_mask is not None else None
        
        # Predict action
        action_pred = self.policy.predict_action(
            state_tokens=state_tokens,
            image_tokens=image_tokens,
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
        )
        
        normalized_actions = action_pred.float().cpu().numpy()[0]
        
        # 根据模型类型提取机器人动作
        if self.use_76d_model and len(normalized_actions.shape) > 0 and normalized_actions.shape[-1] == 76:
            # 76D模型: 提取中间14D机器人动作 [48-61]
            robot_actions = normalized_actions[:, 48:62] if len(normalized_actions.shape) > 1 else normalized_actions[48:62]

            # 动作输出重排，根据use_12_2参数选择不同顺序
            # 模型输出顺序: [左臂0-5, 右臂6-11, 左夹爪12, 右夹爪13]
            if self.use_12_2:
                # 模型使用[左臂6, 右臂6, 左夹爪, 右夹爪]训练，需要重排为环境期望顺序
                # 重排为: [左臂0-5, 左夹爪6, 右臂7-12, 右夹爪13]
                reordered_actions = np.zeros_like(robot_actions)
                if len(robot_actions.shape) == 1:  # 单个动作向量
                    reordered_actions = np.concatenate([
                        robot_actions[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                        [robot_actions[12]],   # 左夹爪 (索引12) → 位置6
                        robot_actions[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                        [robot_actions[13]]    # 右夹爪 (索引13) → 位置13
                    ])
                else:  # 动作序列 (16, 14)
                    for i, action in enumerate(robot_actions):
                        reordered_actions[i] = np.concatenate([
                            action[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                            [action[12]],   # 左夹爪 (索引12) → 位置6
                            action[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                            [action[13]]    # 右夹爪 (索引13) → 位置13
                        ])
                joint_actions = reordered_actions
            else:
                # 模型使用[左臂6, 左夹爪, 右臂6, 右夹爪]训练，与环境期望顺序一致，无需重排
                joint_actions = robot_actions
        elif self.use_62d_model and len(normalized_actions.shape) > 0 and normalized_actions.shape[-1] == 62:
            # 62D模型: 提取后14D机器人动作 [48-61]
            robot_actions = normalized_actions[:, 48:] if len(normalized_actions.shape) > 1 else normalized_actions[48:]
            
            # 动作输出重排，根据use_12_2参数选择不同顺序
            # 模型输出顺序: [左臂0-5, 右臂6-11, 左夹爪12, 右夹爪13]
            if self.use_12_2:
                # 模型使用[左臂6, 右臂6, 左夹爪, 右夹爪]训练，需要重排为环境期望顺序
                # 重排为: [左臂0-5, 左夹爪6, 右臂7-12, 右夹爪13]
                reordered_actions = np.zeros_like(robot_actions)
                if len(robot_actions.shape) == 1:  # 单个动作向量
                    reordered_actions = np.concatenate([
                        robot_actions[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                        [robot_actions[12]],   # 左夹爪 (索引12) → 位置6
                        robot_actions[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                        [robot_actions[13]]    # 右夹爪 (索引13) → 位置13
                    ])
                else:  # 动作序列 (16, 14)
                    for i, action in enumerate(robot_actions):
                        reordered_actions[i] = np.concatenate([
                            action[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                            [action[12]],   # 左夹爪 (索引12) → 位置6
                            action[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                            [action[13]]    # 右夹爪 (索引13) → 位置13
                        ])
                joint_actions = reordered_actions
            else:
                # 模型使用[左臂6, 左夹爪, 右臂6, 右夹爪]训练，与环境期望顺序一致，无需重排
                joint_actions = robot_actions
        else:
            # 14D模型的标准处理
            # 动作输出重排，根据use_12_2参数选择不同顺序
            # 模型输出顺序: [左臂0-5, 右臂6-11, 左夹爪12, 右夹爪13]
            if len(normalized_actions.shape) > 0 and normalized_actions.shape[-1] == 14:
                if self.use_12_2:
                    # 模型使用[左臂6, 右臂6, 左夹爪, 右夹爪]训练，需要重排为环境期望顺序
                    # 重排为: [左臂0-5, 左夹爪6, 右臂7-12, 右夹爪13]
                    reordered_actions = np.zeros_like(normalized_actions)
                    if len(normalized_actions.shape) == 1:  # 单个动作向量
                        reordered_actions = np.concatenate([
                            normalized_actions[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                            [normalized_actions[12]],   # 左夹爪 (索引12) → 位置6
                            normalized_actions[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                            [normalized_actions[13]]    # 右夹爪 (索引13) → 位置13
                        ])
                    else:  # 动作序列 (16, 14)
                        for i, action in enumerate(normalized_actions):
                            reordered_actions[i] = np.concatenate([
                                action[0:6],     # 左臂6关节 (索引0-5) → 位置0-5
                                [action[12]],   # 左夹爪 (索引12) → 位置6
                                action[6:12],    # 右臂6关节 (索引6-11) → 位置7-12
                                [action[13]]    # 右夹爪 (索引13) → 位置13
                            ])
                    joint_actions = reordered_actions
                else:
                    # 模型使用[左臂6, 左夹爪, 右臂6, 右夹爪]训练，与环境期望顺序一致，无需重排
                    joint_actions = normalized_actions
            else:
                joint_actions = normalized_actions
        
        # 应用反归一化（如果启用）
        if self.inv_norm:
            joint_actions = self.inverse_normalize_action(joint_actions)
        
        return joint_actions               # shape: (16, 14)


def encode_obs(observation):
    obs = {}

    # Process separate camera views
    obs['head_cam'] = observation['observation']['head_camera']['rgb']
    obs['left_cam'] = observation['observation']['left_camera']['rgb']
    obs['right_cam'] = observation['observation']['right_camera']['rgb']
    obs['agent_pos'] = observation['joint_action']['vector']

    return obs


def get_model(args):
    ckpt_folder = args.get('ckpt_setting')
    task_name = args.get('task_name')
    use_62d_model = args.get('use_62d_model', False)
    use_12_2 = args.get('use_12_2', False)
    noise_strategy = args.get('noise_strategy', 'gaussian')
    noise_scale = args.get('noise_scale', 1.0)
    inv_norm = args.get('inv_norm', False)
    use_76d_model = args.get('use_76d_model', False)
    use_76d_selective_14d = args.get('use_76d_selective_14d', False)

  
    # Convert relative path to absolute path based on project root
    if ckpt_folder.startswith('./'):
        ckpt_folder = os.path.join(parent_folder_path, ckpt_folder[2:])
    elif not os.path.isabs(ckpt_folder):
        ckpt_folder = os.path.join(parent_folder_path, ckpt_folder)

        model = MyModel(ckpt_folder, task_name, use_62d_model, use_12_2, noise_strategy, noise_scale, inv_norm, use_76d_model, use_76d_selective_14d)
    return model


def eval(TASK_ENV, model, observation):
    torch.cuda.empty_cache()

    obs = encode_obs(observation)       #  编码观测数据
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    
    actions = model.get_action()              #调用模型推理得到动作
    
    for i in range(len(actions)):         
        TASK_ENV.take_action(actions[i])     #执行动作  
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.obs_cache = []