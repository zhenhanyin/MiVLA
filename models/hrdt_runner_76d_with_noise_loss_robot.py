import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.distributions.uniform import Uniform
import math
from typing import Dict, Optional, Tuple, Any
import logging

from utils.hub_mixin import CompatiblePyTorchModelHubMixin
from models.hrdt.model import HRDT


class SigmoidTimestepSampler:
    """
    LogitNormal sampler
    Sampling: u ~ N(mean, std), t = sigmoid(u)
    """
    def __init__(self, timestep_max=1.0, mean=0.0, std=1.0):
        self.timestep_max = timestep_max
        self.mean = mean  # Normal distribution mean
        self.std = std    # Normal distribution standard deviation

    def sample(self, shape):
        """
        LogitNormal sampling, which is sigmoid(randn(m,s))

        1. u ~ N(mean, std)
        2. t = sigmoid(u)
        """
        # Generate normal distribution random numbers u ~ N(mean, std)
        u = torch.normal(mean=self.mean, std=self.std, size=shape)
        # Apply sigmoid transformation to get timesteps in (0,1) range
        t = torch.sigmoid(u)
        # Scale to [0, timestep_max]
        return t * self.timestep_max

    def visualize_distribution(self, num_samples=10000):
        """
        Visualize sampling distribution
        """
        samples = self.sample((num_samples,))
        return {
            'samples': samples,
            'mean': samples.mean().item(),
            'std': samples.std().item(),
            'min': samples.min().item(),
            'max': samples.max().item(),
            'config': f'LogitNormal(mean={self.mean}, std={self.std})'
        }


class ActionEncoder(nn.Module):
    """Action encoder that combines state and action adaptors"""

    def __init__(self, state_dim, action_dim, hidden_size, config):
        super().__init__()
        self.state_adaptor = self.build_condition_adapter(
            config['st_adaptor'],
            in_features=state_dim,
            out_features=hidden_size
        )
        self.action_adaptor = self.build_condition_adapter(
            config['act_adaptor'],
            in_features=action_dim,
            out_features=hidden_size
        )

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def encode_state(self, state_tokens):
        return self.state_adaptor(state_tokens)

    def encode_action(self, action_tokens):
        return self.action_adaptor(action_tokens)


class HRDTRunner76DWithNoiseLossRobot(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/hongzhe2002/H-RDT/"
    ):
    def __init__(self, *, state_dim, action_dim,
                 pred_horizon, config, act_pos_emb_config=None, img_pos_emb_config=None, lang_pos_emb_config=None,
                 max_img_len=None, max_lang_len=None,
                 training_mode='lang',
                 mode='pretrain',
                 pretrained_backbone_path=None,
                 joint_add_eef_loss=False,  # 是否在梯度更新中包含末端位姿损失，默认False
                 dtype=torch.bfloat16):
        super(HRDTRunner76DWithNoiseLossRobot, self).__init__()
        # Create diffusion model
        hidden_size = config['hrdt']['hidden_size']
        self.gradient_checkpointing = False
        self.hidden_size = hidden_size
        self.training_mode = training_mode
        self.mode = mode  # 'finetune_76d_with_noise_loss_robot'
        self.joint_add_eef_loss = joint_add_eef_loss  # 是否包含末端位姿损失

        # Validate mode - only support 76D specific mode
        if mode not in ['finetune_76d_with_noise_loss_robot']:
            raise ValueError(f"mode must be 'finetune_76d_with_noise_loss_robot', got {mode}")

        # Create H-RDT model
        self.model = HRDT(
            horizon=pred_horizon,
            config=config['hrdt'],
            x_pos_emb_config=act_pos_emb_config,
            img_pos_emb_config=img_pos_emb_config,
            lang_pos_emb_config=lang_pos_emb_config,
            max_img_len=max_img_len,
            max_lang_len=max_lang_len,
            training_mode=training_mode,
            dtype=dtype,
        )

        # Image features adapter - use dimensions from config
        self.img_adapter = self.build_condition_adapter(
            config.get('img_adapter', 'mlp2x_silu'),
            in_features=config.get('vision', {}).get('feature_dim', 2048),  # Default to ResNet50 dim
            out_features=hidden_size
        )

        # Action encoder (state and action adaptors)
        self.action_encoder = ActionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            config=config
        )

        # Language features adapter - use dimensions from config
        self.lang_adapter = self.build_condition_adapter(
            config.get('lang_adapter', 'mlp2x_silu'),
            in_features=config.get('text', {}).get('feature_dim', 768),  # Default to DistilBERT dim
            out_features=hidden_size
        )

        # Create noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.timestep_max = noise_scheduler_config['timestep_max']

        sampler_type = noise_scheduler_config.get('sampler_type', 'sigmoid')
        if sampler_type == 'uniform':
            self.timestep_sampler = Uniform(0, self.timestep_max)
        elif sampler_type == 'sigmoid':
            mean = noise_scheduler_config.get('sigmoid_mean', 0.0)
            std = noise_scheduler_config.get('sigmoid_std', 1.0)
            self.timestep_sampler = SigmoidTimestepSampler(self.timestep_max, mean, std)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # TimeNoise config
        self.time_noise_a = config["time_noise"]["a"]
        self.time_noise_beta_m = config["time_noise"]["beta_m"]

        self.img_pos_emb_config = img_pos_emb_config

        # Load pretrained 76D backbone weights
        if mode == 'finetune_76d_with_noise_loss_robot' and pretrained_backbone_path is not None:
            self.load_pretrained_76d_backbone(pretrained_backbone_path)

        # Print model size
        print("Model params: %e" % sum(p.numel() for p in self.parameters()))

    def load_pretrained_76d_backbone(self, pretrained_path):
        """76D微调：完整加载76D预训练权重，包含人类动作、关节动作、末端位姿"""
        logging.info(f"Loading 76D pretrained backbone from {pretrained_path}")

        # Load checkpoint
        if pretrained_path.endswith('.safetensors'):
            import safetensors.torch
            checkpoint = safetensors.torch.load_file(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 76D权重加载策略：完整加载，不做任何维度转换
        logging.info("=== 76D权重加载策略 ===")
        logging.info("策略: 完整加载76D预训练权重")
        logging.info("包含: 48D人类动作 + 14D关节动作 + 14D末端位姿")

        backbone_state_dict = {}
        skipped_keys = []

        for key, value in state_dict.items():
            # 76D微调：完整加载所有兼容权重
            if any(pattern in key for pattern in [
                'final_layer',
            ]):
                skipped_keys.append(key)
                logging.info(f"Skipping {key} - final layer")
                continue

            # 检查图像位置编码的形状兼容性
            if key == 'model.img_pos_emb':
                current_shape = self.model.img_pos_emb.shape
                if value.shape != current_shape:
                    logging.info(f"Skipping img_pos_emb due to shape mismatch: {value.shape} vs {current_shape}")
                    skipped_keys.append(key)
                    continue

            # 76D微调：保留所有权重，包括：
            # ✅ 视觉适配器：img_adapter
            # ✅ 语言适配器：lang_adapter
            # ✅ Transformer主干：model.*
            # ✅ 动作编码器：action_encoder.*, st_adaptor.*, act_adaptor.*
            # ✅ 动作解码器：action_decoder.*
            # ✅ 其他兼容的编码器和适配器

            backbone_state_dict[key] = value
            logging.debug(f"Loading {key} - shape: {value.shape}")

        # 加载权重
        missing_keys, unexpected_keys = self.load_state_dict(backbone_state_dict, strict=False)

        # 所有参数都可训练
        trainable_params = sum(p.numel() for p in self.parameters())
        frozen_params = 0

        # 统计各模块参数数量
        module_stats = {
            'img_adapter': 0, 'lang_adapter': 0, 'transformer': 0,
            'action_encoder': 0, 'action_decoder': 0, 'other': 0
        }

        for name, param in self.named_parameters():
            # 统计模块分布
            if 'img_adapter' in name:
                module_stats['img_adapter'] += param.numel()
            elif 'lang_adapter' in name:
                module_stats['lang_adapter'] += param.numel()
            elif 'model.blocks' in name:
                module_stats['transformer'] += param.numel()
            elif 'action_encoder' in name:
                module_stats['action_encoder'] += param.numel()
            elif 'action_decoder' in name:
                module_stats['action_decoder'] += param.numel()
            else:
                module_stats['other'] += param.numel()

            # 全量微调：所有参数都保持可训练
            param.requires_grad = True

        # 输出模块统计信息
        logging.info("=== 模块参数分布 ===")
        for module, count in module_stats.items():
            logging.info(f"{module}: {count:,} 参数")
        logging.info("==================")

        logging.info(f"76D完整权重加载完成:")
        logging.info(f"  - 加载权重数量: {len(backbone_state_dict)}")
        logging.info(f"  - 跳过权重数量: {len(skipped_keys)}")
        logging.info(f"  - 可训练参数: {trainable_params:,}")
        logging.info(f"  - 冻结参数: {frozen_params:,}")
        logging.info(f"  - 训练比例: 100% (全量微调)")
        logging.info(f"  - 缺失键: {missing_keys}")
        logging.info(f"  - 意外键: {unexpected_keys}")
        logging.info("76D全量微调模式: 完整76D权重 + 全参数训练")
        logging.info("策略: 保留所有先验知识，专注于14D关节动作微调优化")

        # 76D动作维度信息
        logging.info("=== 76D动作维度信息 ===")
        logging.info("  - 总维度: 76")
        logging.info("  - [0-47]: 48D人类动作")
        logging.info("  - [48-61]: 14D关节动作 (左臂6关节+左夹爪+右臂6关节+右夹爪)")
        logging.info("  - [62-75]: 14D末端位姿 (左臂7维+右臂7维)")
        logging.info("==================")

    def verify_76d_finetune_setup(self):
        """验证76D微调设置是否正确"""
        logging.info("=== 76D微调设置验证 ===")

        # 1. 验证维度设置
        assert self.action_dim == 76, f"动作维度应该是76，当前是{self.action_dim}"
        assert self.mode == 'finetune_76d_with_noise_loss_robot', f"模式应该是finetune_76d_with_noise_loss_robot，当前是{self.mode}"

        # 验证末端位姿损失设置
        logging.info(f"✅ 末端位姿损失设置: joint_add_eef_loss = {self.joint_add_eef_loss}")
        if self.joint_add_eef_loss:
            logging.info("  - 末端位姿损失将参与梯度更新")
        else:
            logging.info("  - 末端位姿损失仅用于监控")

        # 2. 验证动作编码器和动作解码器权重是否加载
        action_encoder_keys = [k for k in self.state_dict().keys() if 'action_encoder' in k]
        action_decoder_keys = [k for k in self.state_dict().keys() if 'action_decoder' in k]
        logging.info(f"动作编码器参数数量: {len(action_encoder_keys)}")
        logging.info(f"动作解码器参数数量: {len(action_decoder_keys)}")

        # 3. 验证关键模块是否存在
        required_modules = ['img_adapter', 'lang_adapter', 'action_encoder', 'model']
        for module in required_modules:
            assert hasattr(self, module), f"缺少必要模块: {module}"

        # 验证action_decoder是否存在
        assert hasattr(self.model, 'action_decoder'), "缺少动作解码器模块"

        # 4. 验证action_decoder的输出维度
        action_decoder_output_size = self.model.action_decoder.output_size
        assert action_decoder_output_size == 76, f"动作解码器输出维度应该是76，当前是{action_decoder_output_size}"
        logging.info(f"✅ 动作解码器输出维度正确: {action_decoder_output_size}")

        # 5. 统计可训练参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logging.info(f"总参数: {total_params:,}")
        logging.info(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logging.info(f"冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logging.info("✅ 全量微调模式：所有参数都可训练")

        # 6. 验证损失计算模式
        logging.info("✅ 损失计算模式正确：76D微调专用")
        logging.info("  - 梯度更新：14D关节动作损失 (维度48-61)")
        if self.joint_add_eef_loss:
            logging.info("  - 梯度更新：14D末端位姿损失 (维度62-75) - 已启用")
        else:
            logging.info("  - 监控损失：14D末端位姿损失 (维度62-75) - 仅监控")
        logging.info("  - 监控损失：48D人类动作损失 (维度0-47) - 仅监控")

        # 7. 验证权重继承策略
        logging.info("✅ 76D权重继承策略:")
        logging.info("  - 继承完整76D预训练权重")
        logging.info("  - 保留人类动作先验知识 (前48维)")
        logging.info("  保留关节动作编码解码先验 (中14维)")
        logging.info("保留末端位姿编码解码先验 (后14维)")
        logging.info("  - 专注微调：只优化14D关节动作精度")

        logging.info("=== 76D微调设置验证完成 ===")
        return True

    @classmethod
    def from_pretrained_for_finetune(cls, pretrained_path, state_dim, action_dim, pred_horizon, config, **kwargs):
        """Create model in finetune mode with pretrained 76D backbone"""
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            config=config,
            mode='finetune_76d_with_noise_loss_robot',
            pretrained_backbone_path=pretrained_path,
            **kwargs
        )

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def gradient_checkpointing_enable(self, value=True):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = value
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(value)

    def compute_loss(self, state_tokens=None, action_gt=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        """
        76D微调损失计算：
        - 输入：76D动作数据 [48D人类 + 14D关节 + 14D末端位姿]
        - 预测：76D动作
        - 梯度更新：14D关节动作损失 (维度48-61) + 可选的14D末端位姿损失 (维度62-75)
        - 监控：计算48D人类损失和末端位姿损失
        """
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        # 调试：检查输入数据尺度
        print(f"[DEBUG] action_gt range: min={action_gt.min():.4f}, max={action_gt.max():.4f}, mean={action_gt.mean():.4f}")
        print(f"[DEBUG] action_gt shape: {action_gt.shape}")

        noise = torch.randn(action_gt.shape, dtype=dtype, device=device)
        timesteps = self.timestep_sampler.sample((batch_size,)).to(device)

        broadcasted = timesteps.view(-1, 1, 1)
        noisy_action = (action_gt * broadcasted + noise * (1 - broadcasted)).to(dtype=dtype)

        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        # state/action using action encoder
        state_traj = self.action_encoder.encode_state(state_tokens)
        action_traj = self.action_encoder.encode_action(noisy_action)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        pred = self.model(state_action_traj, timesteps, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)
        target = action_gt - noise

        # 调试：检查模型预测和目标数据尺度
        print(f"[DEBUG] pred range: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
        print(f"[DEBUG] target range: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
        print(f"[DEBUG] pred_joint range: min={pred[:,:,48:62].min():.4f}, max={pred[:,:,48:62].max():.4f}")
        print(f"[DEBUG] target_joint range: min={target[:,:,48:62].min():.4f}, max={target[:,:,48:62].max():.4f}")

        # 初始化损失字典
        loss_dict = {}

        # 76D微调：关节动作损失（始终参与梯度更新）
        # [48-61]: 14D关节动作损失 (用于梯度更新)
        pred_joint = pred[:, :, 48:62]
        target_joint = target[:, :, 48:62]
        joint_loss = F.mse_loss(pred_joint, target_joint)

        # 初始化总损失为关节损失
        total_loss = joint_loss

        # 根据 joint_add_eef_loss 参数决定是否添加末端位姿损失
        if self.joint_add_eef_loss:
            # [62-75]: 14D末端位姿损失 (参与梯度更新)
            pred_eef = pred[:, :, 62:76]
            target_eef = target[:, :, 62:76]
            eef_loss = F.mse_loss(pred_eef, target_eef)
            total_loss = total_loss + eef_loss
            loss_dict["eef_loss"] = eef_loss
        else:
            # 末端位姿损失仅用于监控
            with torch.no_grad():
                pred_eef = pred[:, :, 62:76]
                target_eef = target[:, :, 62:76]
                eef_loss = F.mse_loss(pred_eef, target_eef)
                loss_dict["eef_loss_monitor"] = eef_loss

        # 设置损失字典
        loss_dict["joint_loss"] = joint_loss
        loss_dict["loss"] = total_loss
        loss_dict["diff_loss"] = total_loss  # 为了兼容性，更新diff_loss

        # 监控人类动作损失（不参与梯度更新）
        with torch.no_grad():
            # [0-47]: 48D人类动作损失（仅监控）
            pred_human = pred[:, :, :48]
            target_human = target[:, :, :48]
            human_loss = F.mse_loss(pred_human, target_human)
            loss_dict["human_loss_monitor"] = human_loss

        # 详细损失信息（用于监控）
        if self.joint_add_eef_loss:
            loss_dict["loss_breakdown"] = {
                "joint_loss": joint_loss.item(),
                "eef_loss": eef_loss.item(),
                "total_loss": total_loss.item(),
                "human_loss_monitor": human_loss.item(),
                "loss_mode": "joint+eef"
            }
        else:
            loss_dict["loss_breakdown"] = {
                "joint_loss": joint_loss.item(),
                "total_loss": total_loss.item(),
                "human_loss_monitor": human_loss.item(),
                "eef_loss_monitor": eef_loss.item(),
                "loss_mode": "joint_only"
            }

        return loss_dict

    @torch.no_grad()
    def predict_action(self, state_tokens=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        '''
        state_tokens: (batch_size, chunk_size, action_dim)
        image_tokens: (batch_size, img_len, in_feat_dim)
        lang_tokens (torch.Tensor): language features [B, L, hidden_size] (unpooled)
        lang_attn_mask: (batch_size, L), attention mask for language tokens

        return: (batch_size, chunk_size, action_dim), predicted action sequence
        '''
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        state_traj = self.action_encoder.encode_state(state_tokens)
        noisy_action = torch.randn((batch_size, self.pred_horizon, self.action_dim), dtype=dtype, device=device)
        timestep = torch.tensor([0.0], dtype=dtype, device=device)
        step_size = 1.0 / self.num_inference_timesteps

        for _ in range(self.num_inference_timesteps):
            action_traj = self.action_encoder.encode_action(noisy_action)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            pred = self.model(state_action_traj, timestep, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)
            noisy_action = pred * step_size + noisy_action
            timestep = timestep + step_size

        return noisy_action

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)