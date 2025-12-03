# RoboTwin2 Inference Setup

## Setup Steps

1. Copy MiVLA folder to RoboTwin/policy/
```bash
cp -r MiVLA /path/to/RoboTwin/policy/
```

2. Copy bak folder to MiVLA/
```bash
cp -r MiVLA/bak /path/to/RoboTwin/policy/MiVLA/
```

3. Create checkpoints directory and copy model files
```bash
mkdir -p /path/to/RoboTwin/policy/MiVLA/checkpoints/folder_name/
cp MiVLA/checkpoints/*/config.json /path/to/RoboTwin/policy/MiVLA/checkpoints/folder_name/
cp MiVLA/checkpoints/*/pytorch_model.bin /path/to/RoboTwin/policy/MiVLA/checkpoints/folder_name/
```

## Run Inference

1. Modify eval.sh configuration
```bash
cd /path/to/RoboTwin/policy/MiVLA
# Edit eval.sh parameters:
# - ckpt_setting="checkpoints/folder_name"  # Set checkpoint path
# - task_name="your_task"                 # Set task name
# - task_config="your_config"             # Set task config
```

2. Run inference
```bash
bash eval.sh
```
