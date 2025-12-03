#!/usr/bin/env python3
"""
Script to generate text embedding for a single task
Based on H_RDT/datasets/real_robot/create_real_robot_language_embedding.py
"""

import os
import torch
import yaml
import sys
from pathlib import Path

# Add the project root to sys.path for importing models
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.encoder.t5_encoder import T5Embedder

def create_single_task_embedding(task_name, instruction):
    """
    Generate text embedding for a single task
    """
    
    # Configuration
    GPU = 0
    MODEL_PATH = os.environ.get('T5_MODEL_PATH', './t5-v1_1-xxl')
    CONFIG_PATH = os.environ.get('HRDT_CONFIG_PATH', project_root / 'configs/hrdt_finetune.yaml')
    
    # Get current script directory
    current_dir = Path(__file__).parent
    OUTPUT_DIR = current_dir / 'lang_embeddings'
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"Task name: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"T5 model path: {MODEL_PATH}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)
    
    # Load configuration
    try:
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        print("✓ Config file loaded successfully")
    except FileNotFoundError:
        print(f"✗ Config file not found: {CONFIG_PATH}")
        return None
    except Exception as e:
        print(f"✗ Error loading config file: {e}")
        return None
    
    # Initialize T5 embedder
    device = torch.device(f"cuda:{GPU}")
    print(f"Using device: {device}")
    
    try:
        print(f"Attempting to load T5 model from: {MODEL_PATH}")
        print(f"Tokenizer max length: {config['dataset']['tokenizer_max_length']}")
        
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device,
            use_offload_folder=None
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        print("✓ T5 embedder initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing T5 embedder: {e}")
        import traceback
        traceback.print_exc()
        print("Please check T5_MODEL_PATH environment variable and model availability")
        return None
    
    try:
        # Tokenize instruction
        tokens = tokenizer(
            instruction, 
            return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)
        
        tokens = tokens.view(1, -1)
        print(f"Tokens shape: {tokens.shape}")
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = text_encoder(tokens).last_hidden_state.detach().cpu()
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[-1]}")
        print(f"Sequence length: {embeddings.shape[1]}")
        
        # Save embeddings
        save_path = OUTPUT_DIR / f"{task_name}.pt"
        embedding_data = {
            "task_name": task_name,
            "instruction": instruction,
            "embeddings": embeddings
        }
        
        torch.save(embedding_data, save_path)
        
        print(f"✓ Embedding saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"✗ Error processing task {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function - modify task_name and instruction here
    """
    
    # Modify these parameters for your task
    task_name = "scan_object"  # Change this to your task name
    instruction = "Use one arm to pick the scanner and use the other arm to pick the object, and use the scanner to scan the object"  # Change this to your instruction
    
    print("=" * 60)
    print("Single Task Text Embedding Generation")
    print("=" * 60)
    
    result = create_single_task_embedding(task_name, instruction)
    
    if result:
        print("\n" + "=" * 60)
        print("✓ Single task embedding generation completed successfully!")
        print(f"Output file: {result}")
        print("\nUsage:")
        print("1. The language embedding file is saved to lang_embeddings/ directory")
        print("2. Set task_name='scan_object' when initializing your dataset")
        print("3. The dataset will automatically load the corresponding language embedding")
    else:
        print("\n" + "=" * 60)
        print("✗ Failed to generate embedding")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()