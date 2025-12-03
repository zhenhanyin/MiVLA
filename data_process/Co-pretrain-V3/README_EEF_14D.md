# 末端执行器14D位姿生成脚本

## 概述

这个脚本根据Egodex人类手部动作数据，生成虚拟双臂机器人的末端执行器位姿数据。与原脚本不同，这个脚本**不进行逆运动学求解**，直接输出末端执行器的笛卡尔位姿。

## 核心功能

### 输入数据
- **人类手部变换数据**: `/transforms/leftThumbIntermediateBase` 和 `/transforms/rightThumbIntermediateBase`
- **数据格式**: 4x4变换矩阵 [R|t]

### 输出数据
- **末端执行器位姿**: `eef_14d` 数据集
- **数据格式**: `[N, 14]` 数组，其中：
  - `0-6`: 左臂末端执行器位姿 `[x, y, z, qx, qy, qz, qw]`
  - `7-13`: 右臂末端执行器位姿 `[x, y, z, qx, qy, qz, qw]`

## 坐标变换

### 1. 坐标系映射
```python
# Egodes坐标系 → PyBullet机器人坐标系
transform_matrix = [
    [0, 0, -1],  # 人类-Z -> 机器人X
    [-1, 0, 0],  # 人类-X -> 机器人Y
    [0, 1, 0]    # 人类Y -> 机器人Z
]
```

### 2. 位置计算
```
robot_target_pos = robot_start_pos + transform_matrix @ (human_current_pos - human_start_pos)
```

### 3. 姿态计算
```
robot_quat = Rotation.from_matrix(transform_matrix @ human_rotation_matrix).as_quat()
```

## 使用方法

### 1. 基本使用
```bash
python precompute_eef_14d_final.py \
    --input_root "/path/to/Egodex-full" \
    --output_root "/path/to/output" \
    --num_processes 10
```

### 2. 参数说明
- `--input_root`: 输入根目录（Egodex-full数据路径）
- `--output_root`: 输出根目录
- `--num_processes`: 并行处理进程数

### 3. 快速测试
```bash
# 运行示例脚本（包含测试功能）
./run_eef_14d_example.sh
```

## 输出文件结构

每个输出HDF5文件包含：
```
output_file.hdf5
├── eef_14d              # [N, 14] 末端执行器位姿数据
└── attributes
    ├── source_file       # 原始输入文件路径
    ├── total_frames      # 总帧数
    ├── coordinate_transform_matrix  # 坐标变换矩阵
    ├── left_robot_start_pos        # 左臂起始位置
    └── right_robot_start_pos       # 右臂起始位置
```

## 测试和验证

### 1. 单文件测试
```bash
python test_eef_14d.py path/to/output/file.hdf5
```

### 2. 带可视化测试
```bash
python test_eef_14d.py path/to/output/file.hdf5 --visualize --output_dir ./plots
```

### 3. 测试输出包括
- 数据统计信息（位置范围、四元数模长等）
- 3D轨迹可视化图
- 位置随时间变化曲线

## 与原脚本的区别

| 特性 | 原脚本 | 新脚本 |
|------|--------|--------|
| **输出维度** | 14D (关节角度+夹爪) | 14D (位置+四元数) |
| **IK求解** | ✅ 需要PyBullet IK | ❌ 无需IK求解 |
| **依赖** | 需要PyBullet | 仅需NumPy/SciPy |
| **计算复杂度** | 高（IK迭代） | 低（矩阵变换） |
| **处理速度** | 较慢 | 更快 |
| **输出内容** | 关节空间 | 笛卡尔空间 |

## 应用场景

### 1. 末端执行器轨迹学习
- 直接学习末端执行器的运动轨迹
- 适用于任务空间的学习算法

### 2. 位姿预测任务
- 预测机器人末端执行器的目标位姿
- 用于轨迹规划和控制

### 3. 跨机器人迁移学习
- 笛卡尔空间的位姿数据具有更好的泛化性
- 可以适配不同机器人构型

## 技术细节

### 1. 数据预处理
- 使用第0帧作为坐标变换基准
- 异常帧采用上一帧数据填充
- 第一帧异常使用默认位姿

### 2. 坐标系一致性
- 与原始脚本使用相同的坐标变换矩阵
- 保持与机器人模型的兼容性

### 3. 并行处理
- 支持多进程并行处理
- 进度条显示处理状态
- 错误处理和容错机制

## 故障排除

### 1. 常见问题
- **输入文件不存在**: 检查输入路径是否正确
- **权限错误**: 确保有读写权限
- **内存不足**: 减少进程数量

### 2. 调试建议
- 先用少量文件测试
- 检查原始数据格式是否正确
- 使用测试脚本验证输出

## 版本信息
- **版本**: 1.0
- **依赖**: NumPy, SciPy, h5py, tqdm
- **Python版本**: 3.7+
- **作者**: Claude Assistant