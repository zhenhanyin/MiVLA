#!/usr/bin/env python3
"""
真机数据语言嵌入创建脚本
参考 H_RDT/datasets/robotwin2/encode_lang_batch.py
直接在代码中设置任务名称和指令，不使用CSV文件
"""

import os
import torch
import yaml
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.encoder.t5_encoder import T5Embedder

def create_real_robot_embeddings():
    """
    为真机任务创建语言嵌入
    """
    
    # 配置参数
    GPU = 5
    MODEL_PATH = os.environ.get('T5_MODEL_PATH')
    CONFIG_PATH = project_root / 'configs/finetune_real_robot_14d.yaml'
    
    # 输出目录 - 保存到real_robot的lang_embeddings中
    OUTPUT_DIR = Path(__file__).parent / 'lang_embeddings'
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"T5模型路径: {MODEL_PATH}")
    print(f"配置文件路径: {CONFIG_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("-" * 60)
    
    # 加载配置
    try:
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        print("✓ 配置文件加载成功")
    except FileNotFoundError:
        print(f"✗ 配置文件未找到: {CONFIG_PATH}")
        return
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}")
        return
    
    # 初始化T5嵌入器
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        print(f"尝试加载T5模型，路径: {MODEL_PATH}")
        print(f"配置中的tokenizer_max_length: {config['dataset']['tokenizer_max_length']}")
        print(f"设备: {device}")
        
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device,
            use_offload_folder=None
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        print("✓ T5嵌入器初始化成功")
    except Exception as e:
        print(f"✗ T5嵌入器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查T5_MODEL_PATH环境变量和模型可用性")
        return
    
    # 在这里直接设置你的真机任务和指令
    real_robot_tasks = [
        {
            "task_name": "put_the_ball",
            "instruction": "One claw holds the paper box in the middle, while the other claw grabs the small ball and puts it into the paper box."
        },
        # 如果你有其他任务，可以在这里添加
        # {
        #     "task_name": "your_second_task",
        #     "instruction": "your second instruction here"
        # }
    ]
    
    print(f"\\n开始处理 {len(real_robot_tasks)} 个真机任务...")
    
    # 处理每个任务
    success_count = 0
    total_tasks = len(real_robot_tasks)
    
    for idx, task_info in enumerate(real_robot_tasks):
        task_name = task_info["task_name"]
        instruction = task_info["instruction"]
        
        print(f"\\n处理 [{idx+1}/{total_tasks}]: {task_name}")
        print(f"指令: {instruction}")
        
        try:
            # Tokenize指令
            tokens = tokenizer(
                instruction, 
                return_tensors="pt",
                padding="longest",
                truncation=True
            )["input_ids"].to(device)
            
            tokens = tokens.view(1, -1)
            print(f"Tokens形状: {tokens.shape}")
            
            # 生成嵌入
            with torch.no_grad():
                embeddings = text_encoder(tokens).last_hidden_state.detach().cpu()
            
            print(f"嵌入形状: {embeddings.shape}")
            print(f"嵌入维度: {embeddings.shape[-1]}")
            print(f"序列长度: {embeddings.shape[1]}")
            
            # 保存嵌入
            save_path = OUTPUT_DIR / f"{task_name}.pt"
            embedding_data = {
                "task_name": task_name,
                "instruction": instruction,
                "embeddings": embeddings
            }
            
            torch.save(embedding_data, save_path)
            
            print(f"✓ 嵌入已保存到: {save_path}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 处理任务 {task_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\\n" + "="*60)
    print("真机数据语言嵌入创建完成!")
    print(f"成功处理: {success_count}/{total_tasks} 个任务")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if success_count < total_tasks:
        print(f"警告: {total_tasks - success_count} 个任务处理失败")
    
    # 显示使用说明
    print("\\n" + "="*60)
    print("使用说明:")
    print("1. 语言嵌入文件已保存到 lang_embeddings/ 目录")
    print("2. 在数据集初始化时设置 task_name='pick_trash_exp'")
    print("3. 数据集会自动加载对应的语言嵌入")
    print("4. 确保微调配置中的 state_dim 和 action_dim 与真机数据匹配")

if __name__ == "__main__":
    create_real_robot_embeddings()