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


class ActionEncoder16DEEF(nn.Module):
    """16D末端执行器专用Action encoder"""

    def __init__(self, state_dim, action_dim, hidden_size, config):
        super().__init__()
        assert action_dim == 16, "This encoder only supports 16D end-effector actions"

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


class SelectiveWeightLoader76DEEF:
    """智能76D到16D末端执行器选择性权重加载器"""

    def __init__(self):
        self.load_stats = {
            'total_pretrained': 0,
            'loaded': 0,
            'converted': 0,
            'skipped': 0,
            'reinitialized': 0
        }

    def convert_76d_to_16d_eef_weight(self, weight_76d):
        """
        将76维权重转换为16维，按照指定顺序提取末端执行器权重

        76D结构: [48D人手动作] + [左臂6关节+左臂夹爪1维+右臂6关节+右臂夹爪1维] + [左臂末端位姿7维+右臂末端位姿7维]
        索引范围: [0-47] + [48-61] + [62-75]

        要提取的16维权重顺序：
        1. [62-68]: 左臂末端执行器位姿7维（x y z qw qx qy qz）
        2. [54]: 左臂夹爪开合1维
        3. [69-75]: 右臂末端执行器位姿7维（x y z qw qx qy qz）
        4. [61]: 右臂夹爪开合1维

        Args:
            weight_76d: torch.Tensor，形状为[..., 76]或[76, ...]

        Returns:
            torch.Tensor，转换后的16维权重（按指定顺序的末端执行器）
        """
        if weight_76d.dim() == 2:
            if weight_76d.shape[0] == 76:
                # 输出层: [76, hidden] -> [16, hidden] (例如: action_decoder.ffn.fc2)
                left_arm_eef = weight_76d[62:69, :]      # 左臂末端位姿7维 [62-68]
                left_arm_gripper = weight_76d[54:55, :]   # 左臂夹爪1维 [54]
                right_arm_eef = weight_76d[69:76, :]     # 右臂末端位姿7维 [69-75]
                right_arm_gripper = weight_76d[61:62, :]  # 右臂夹爪1维 [61]
                return torch.cat([left_arm_eef, left_arm_gripper, right_arm_eef, right_arm_gripper], dim=0)
            elif weight_76d.shape[1] == 76:
                # 输入层: [hidden, 76] -> [hidden, 16] (例如: action_encoder.*.0.weight)
                left_arm_eef = weight_76d[:, 62:69]      # 左臂末端位姿7维 [62-68]
                left_arm_gripper = weight_76d[:, 54:55]   # 左臂夹爪1维 [54]
                right_arm_eef = weight_76d[:, 69:76]     # 右臂末端位姿7维 [69-75]
                right_arm_gripper = weight_76d[:, 61:62]  # 右臂夹爪1维 [61]
                return torch.cat([left_arm_eef, left_arm_gripper, right_arm_eef, right_arm_gripper], dim=1)
        elif weight_76d.dim() == 1:
            # 偏置层: [76] -> [16] (例如: action_decoder.ffn.fc2.bias)
            left_arm_eef = weight_76d[62:69]      # 左臂末端位姿7维 [62-68]
            left_arm_gripper = weight_76d[54:55]   # 左臂夹爪1维 [54]
            right_arm_eef = weight_76d[69:76]     # 右臂末端位姿7维 [69-75]
            right_arm_gripper = weight_76d[61:62]  # 右臂夹爪1维 [61]
            return torch.cat([left_arm_eef, left_arm_gripper, right_arm_eef, right_arm_gripper], dim=0)

        # 其他维度保持不变（例如: bias [2176] 不需要转换）
        return weight_76d

    def load_selective_weights(self, model, pretrained_state_dict, config):
        """
        选择性加载预训练权重到16D末端执行器模型

        Args:
            model: 16D目标模型
            pretrained_state_dict: 76D预训练权重
            config: 模型配置

        Returns:
            load_report: 加载报告
        """
        model_state_dict = model.state_dict()
        loaded_state_dict = {}

        # 定义权重加载优先级
        high_priority_patterns = [
            'model.blocks.',           # Transformer主干
            'img_adapter.',            # 视觉适配器
            'lang_adapter.',           # 语言适配器
            'model.t_embedder.',       # 时间步嵌入
        ]

        # 只转换指定的4个层，而不是整个模块
        convert_exact_matches = [
            'model.action_decoder.ffn.fc2.weight',
            'model.action_decoder.ffn.fc2.bias',
            'action_encoder.state_adaptor.0.weight',
            'action_encoder.action_adaptor.0.weight',
        ]

        # 遍历预训练权重
        for name, param in pretrained_state_dict.items():
            self.load_stats['total_pretrained'] += 1
            loaded = False

            # 高优先级：完全保留
            for pattern in high_priority_patterns:
                if name.startswith(pattern):
                    if name in model_state_dict:
                        if model_state_dict[name].shape == param.shape:
                            loaded_state_dict[name] = param
                            self.load_stats['loaded'] += 1
                            loaded = True
                            break
                        else:
                            logging.warning(f"Shape mismatch for {name}: {param.shape} vs {model_state_dict[name].shape}")

            if loaded:
                continue

            # 中优先级：精确匹配转换指定层
            if name in convert_exact_matches:
                if name in model_state_dict:
                    try:
                        converted_param = self.convert_76d_to_16d_eef_weight(param)
                        if converted_param.shape == model_state_dict[name].shape:
                            loaded_state_dict[name] = converted_param
                            self.load_stats['converted'] += 1
                            loaded = True
                            logging.info(f"Converted {name}: {param.shape} -> {converted_param.shape}")
                        else:
                            logging.warning(f"Converted shape mismatch for {name}: {converted_param.shape} vs {model_state_dict[name].shape}")
                    except Exception as e:
                        logging.error(f"Failed to convert {name}: {e}")

            if loaded:
                continue

            # 最后：通用加载 - 尝试加载所有剩余的权重
            if not loaded and name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    loaded_state_dict[name] = param
                    self.load_stats['loaded'] += 1
                    loaded = True

        # 加载处理后的权重
        model.load_state_dict(loaded_state_dict, strict=False)

        # 生成加载报告
        missing_keys = [k for k in model_state_dict.keys() if k not in loaded_state_dict]
        load_report = {
            'stats': self.load_stats,
            'loaded_keys': list(loaded_state_dict.keys()),
            'missing_keys': missing_keys,
            'unexpected_keys': [k for k in pretrained_state_dict.keys() if k not in model_state_dict],
        }

        # 打印重新初始化的层的详细信息
        if missing_keys:
            logging.info(f"Reinitialized {len(missing_keys)} layers:")
            for key in missing_keys:
                shape = model_state_dict[key].shape
                logging.info(f"  - {key}: {shape}")

        return load_report


class HRDTRunner76Dselective16DEEF(
        nn.Module,
        CompatiblePyTorchModelHubMixin,
        repo_url="https://huggingface.co/hongzhe2002/H-RDT/"
    ):
    """76D到16D末端执行器选择性加载HRDT Runner，从76D预训练权重中选择性加载并转换为16维末端执行器动作"""

    def __init__(self, *, state_dim, action_dim,
                 pred_horizon, config, act_pos_emb_config=None, img_pos_emb_config=None, lang_pos_emb_config=None,
                 max_img_len=None, max_lang_len=None,
                 training_mode='lang',
                 mode='finetune',  # 标准模式
                 pretrained_backbone_path=None,
                 loss_mode='16d_eef_only',  # 16D末端执行器专用损失模式
                 dtype=torch.bfloat16):
        super(HRDTRunner76Dselective16DEEF, self).__init__()

        # 验证输入维度
        assert action_dim == 16, f"16D EEF model only supports action_dim=16, got {action_dim}"
        assert mode in ['pretrain', 'finetune', 'finetune_76d_selective_16d_eef'], f"Unknown mode: {mode}"
        assert loss_mode in ['standard', '16d_eef_only'], f"Unknown loss_mode: {loss_mode}"

        # Create diffusion model
        hidden_size = config['hrdt']['hidden_size']
        self.gradient_checkpointing = False
        self.hidden_size = hidden_size
        self.training_mode = training_mode
        self.mode = mode
        self.loss_mode = loss_mode

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

        # Image features adapter
        self.img_adapter = self.build_condition_adapter(
            config.get('img_adapter', 'mlp2x_silu'),
            in_features=config.get('vision', {}).get('feature_dim', 2048),
            out_features=hidden_size
        )

        # 16D末端执行器专用action encoder
        self.action_encoder = ActionEncoder16DEEF(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            config=config
        )

        # Language features adapter
        self.lang_adapter = self.build_condition_adapter(
            config.get('lang_adapter', 'mlp2x_silu'),
            in_features=config.get('text', {}).get('feature_dim', 768),
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

        # Selective loading for 76D pretrained weights
        if mode == 'finetune_76d_selective_16d_eef' and pretrained_backbone_path is not None:
            print("=" * 80)
            print("🚀 LOADING 76D PRETRAINED WEIGHTS FOR 16D END-EFFECTOR SELECTIVE FINE-TUNING")
            print(f"📂 Pretrained path: {pretrained_backbone_path}")
            print(f"🔧 Mode: {mode}")
            print("=" * 80)

            load_report = self.load_selective_pretrained_weights(pretrained_backbone_path, config)

            # 检查权重加载情况
            total_loaded = load_report['stats']['loaded'] + load_report['stats']['converted']
            conversion_rate = total_loaded / load_report['stats']['total_pretrained'] * 100

            print("=" * 80)
            print("✅ WEIGHT LOADING SUMMARY:")
            print(f"   📊 Total pretrained weights: {load_report['stats']['total_pretrained']}")
            print(f"   ✅ Directly loaded: {load_report['stats']['loaded']}")
            print(f"   🔄 Converted (76D→16D EEF): {load_report['stats']['converted']}")
            print(f"   ❌ Skipped/Reinitialized: {len(load_report['missing_keys'])}")
            print(f"   📈 Conversion rate: {conversion_rate:.1f}%")

            if conversion_rate < 50.0:
                print("\n⚠️  WARNING: Low weight loading rate detected!")
                print("   Most weights are being reinitialized from scratch.")
                print("   Consider checking the compatibility between pretrained and target models.")
            elif conversion_rate < 80.0:
                print("\n⚠️  WARNING: Moderate weight loading rate.")
                print("   Some important weights might be reinitialized.")
            else:
                print("\n🎉 SUCCESS: Good weight loading rate achieved!")

            print("=" * 80)
        elif pretrained_backbone_path is not None:
            # 有预训练权重路径但模式不匹配
            print("=" * 80)
            print("❌ WEIGHT LOADING SKIPPED!")
            print(f"🔧 Current mode: {mode}")
            print(f"📂 Pretrained path exists: {pretrained_backbone_path}")
            print("❌ Pretrained weights will NOT be loaded!")
            print("❌ Model will train from randomly initialized weights!")
            print("=" * 80)
            print("\n💡 To enable pretrained weight loading, use mode:")
            print("   --mode='finetune_76d_selective_16d_eef' (instead of current mode)")
            print("\n🛑 TRAINING STOPPED - Please fix the mode configuration!")
            print("=" * 80)
            raise ValueError(f"Weight loading disabled for mode '{mode}'. "
                           f"Use 'finetune_76d_selective_16d_eef' mode to enable pretrained weight loading.")
        else:
            # 非finetune模式，也没有预训练权重
            print("=" * 80)
            print("ℹ️  TRAINING FROM SCRATCH")
            print(f"🔧 Mode: {mode}")
            print("✅ No pretrained weights expected - training from random initialization")
            print("=" * 80)
            raise ValueError(f"check mode and pretrained weights path ")
        # Print model size
        print(f"📊 16D End-Effector Model params: {sum(p.numel() for p in self.parameters()):.2e}")
        print("=" * 80)

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

    def load_selective_pretrained_weights(self, pretrained_path, config):
        """选择性加载76D预训练权重到16D末端执行器模型"""
        logging.info(f"Loading 76D pretrained weights selectively for 16D EEF model from {pretrained_path}")

        # Load checkpoint
        if pretrained_path.endswith('.safetensors'):
            import safetensors.torch
            checkpoint = safetensors.torch.load_file(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_state_dict = checkpoint['model']
        else:
            pretrained_state_dict = checkpoint

        # 使用76D选择性权重加载器
        loader = SelectiveWeightLoader76DEEF()
        load_report = loader.load_selective_weights(self, pretrained_state_dict, config)

        # 打印加载报告
        logging.info("=== 76D Selective EEF Weight Loading Report ===")
        logging.info(f"Total pretrained weights: {load_report['stats']['total_pretrained']}")
        logging.info(f"Directly loaded: {load_report['stats']['loaded']}")
        logging.info(f"Converted (76D->16D EEF): {load_report['stats']['converted']}")
        logging.info(f"Skipped: {load_report['stats']['skipped']}")
        logging.info(f"Reinitialized: {len(load_report['missing_keys'])}")
        logging.info(f"Conversion rate: {(load_report['stats']['loaded'] + load_report['stats']['converted']) / load_report['stats']['total_pretrained'] * 100:.1f}%")

        return load_report

    def forward(self, state_tokens=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        """
        Forward pass for 16D end-effector model

        Args:
            state_tokens: (batch_size, 1, 16) - 只包含16维末端执行器状态
            image_tokens: (batch_size, img_len, img_dim)
            lang_tokens: (batch_size, lang_len, lang_dim)
            lang_attn_mask: (batch_size, lang_len)
        """
        batch_size = state_tokens.shape[0]
        device = state_tokens.device
        dtype = state_tokens.dtype

        # 处理16维状态和动作
        img_c = self.img_adapter(image_tokens)

        # Process language features
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)

        # 16D末端执行器专用编码
        state_traj = self.action_encoder.encode_state(state_tokens)

        # 对于16D模型，我们需要生成对应的噪声动作
        # 创建与state_traj匹配的随机动作
        action_noise = torch.randn(batch_size, self.pred_horizon, 16, device=device, dtype=dtype)
        action_traj = self.action_encoder.encode_action(action_noise)

        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        # Sample timesteps
        timesteps = self.timestep_sampler.sample((batch_size,)).to(device)

        # Flow matching noise
        noise = torch.randn_like(action_noise)
        broadcasted = timesteps.view(-1, 1, 1)
        noisy_action = (action_noise * broadcasted + noise * (1 - broadcasted)).to(dtype=dtype)
        action_traj = self.action_encoder.encode_action(noisy_action)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        # Model forward
        pred = self.model(state_action_traj, timesteps, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)
        target = action_noise - noise

        # 计算损失（只针对16维末端执行器）
        loss = F.mse_loss(pred, target)

        return {
            "loss": loss,
            "diff_loss": loss,
            "pred": pred,
            "target": target
        }

    def compute_loss(self, state_tokens=None, action_gt=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        """
        Compute training loss for 16D end-effector model

        Args:
            img_tokens: (batch_size, img_len, img_token_dim)
            state_tokens: (batch_size, 1, 16), current state
            action_gt: (batch_size, chunk_size, 16), ground-truth actions for supervision
            lang_tokens: (batch_size, L, hidden_size), language features (unpooled)
            lang_attn_mask: (batch_size, L), attention mask for language tokens
        """
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        # Generate noise
        noise = torch.randn(action_gt.shape, dtype=dtype, device=device)
        timesteps = self.timestep_sampler.sample((batch_size,)).to(device)

        # Flow matching: interpolate between action and noise
        broadcasted = timesteps.view(-1, 1, 1)
        noisy_action = (action_gt * broadcasted + noise * (1 - broadcasted)).to(dtype=dtype)

        # Process features
        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        # 16D state/action encoding
        state_traj = self.action_encoder.encode_state(state_tokens)
        action_traj = self.action_encoder.encode_action(noisy_action)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)

        # Model forward
        pred = self.model(state_action_traj, timesteps, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)
        target = action_gt - noise

        # Compute loss (only 16D end-effector)
        diff_loss = F.mse_loss(pred, target)

        loss_dict = {
            "diff_loss": diff_loss,
            "loss": diff_loss,
            "eef_loss": diff_loss  # For compatibility with existing code
        }

        return loss_dict

    @torch.no_grad()
    def predict_action(self, state_tokens=None, image_tokens=None, lang_tokens=None, lang_attn_mask=None):
        """16D末端执行器动作预测 - 与标准版本保持一致的推理算法"""
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        dtype = image_tokens.dtype

        img_c = self.img_adapter(image_tokens)

        # Process language features - handle None case
        lang_c = None
        if lang_tokens is not None:
            lang_c = self.lang_adapter(lang_tokens)  # [B, L, D] - keep unpooled for cross attention

        # 16D末端执行器专用编码
        state_traj = self.action_encoder.encode_state(state_tokens)
        noisy_action = torch.randn((batch_size, self.pred_horizon, 16), dtype=dtype, device=device)
        timestep = torch.tensor([0.0], dtype=dtype, device=device)
        step_size = 1.0 / self.num_inference_timesteps

        for _ in range(self.num_inference_timesteps):
            action_traj = self.action_encoder.encode_action(noisy_action)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            pred = self.model(state_action_traj, timestep, img_c=img_c, lang_c=lang_c, lang_attn_mask=lang_attn_mask)

            # 标准的逐步去噪算法
            noisy_action = pred * step_size + noisy_action
            timestep = timestep + step_size

        return noisy_action

    def gradient_checkpointing_enable(self, value: bool = True):
        """Enable gradient checkpointing to save memory"""
        self.gradient_checkpointing = value
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(value)