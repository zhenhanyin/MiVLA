#!/bin/bash

# 末端执行器14D位姿生成示例脚本

echo "🤖 末端执行器14D位姿生成示例"
echo "=================================="

# 设置输入和输出路径
INPUT_ROOT="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/yinzhenhan/data/dataset/Egodex-full"
OUTPUT_ROOT="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/yinzhenhan/data/dataset/Egodex-eef-14d-pretrain"

# 检查输入目录是否存在
if [ ! -d "$INPUT_ROOT" ]; then
    echo "❌ 输入目录不存在: $INPUT_ROOT"
    echo "请确保输入路径正确或修改脚本中的INPUT_ROOT变量"
    exit 1
fi

echo "📁 输入目录: $INPUT_ROOT"
echo "📁 输出目录: $OUTPUT_ROOT"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 示例1: 处理小批量数据进行测试
echo "🔬 示例1: 处理少量文件进行功能测试"
echo "----------------------------------------"

# 创建临时目录存放少量测试文件
TEMP_INPUT_DIR="./temp_test_input"
TEMP_OUTPUT_DIR="./temp_test_output"

mkdir -p "$TEMP_INPUT_DIR"

# 复制前几个文件用于测试（如果输入目录中有文件的话）
find "$INPUT_ROOT" -name "*.hdf5" -type f | head -5 | while read file; do
    rel_path=$(realpath --relative-to="$INPUT_ROOT" "$file")
    mkdir -p "$TEMP_INPUT_DIR/$(dirname "$rel_path")"
    cp "$file" "$TEMP_INPUT_DIR/$rel_path"
    echo "📋 复制测试文件: $(basename "$file")"
done

if [ -z "$(ls -A $TEMP_INPUT_DIR 2>/dev/null)" ]; then
    echo "⚠️  没有找到可测试的HDF5文件，尝试直接处理输入目录"
    TEMP_INPUT_DIR="$INPUT_ROOT"
    TEMP_OUTPUT_DIR="./test_output_eef_14d"
else
    echo "🚀 开始处理测试文件..."
    python precompute_eef_14d_final.py \
        --input_root "$TEMP_INPUT_DIR" \
        --output_root "$TEMP_OUTPUT_DIR" \
        --num_processes 2
fi

echo ""

# 示例2: 完整处理（需要大量时间和存储空间）
echo "🔥 示例2: 完整处理所有数据（需要较长时间）"
echo "----------------------------------------"

read -p "是否要开始处理完整数据集？这可能需要几个小时 (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 开始完整处理..."
    python precompute_eef_14d_final.py \
        --input_root "$INPUT_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --num_processes 10

    echo ""
    echo "✅ 完整处理完成！"
    echo "📁 结果保存在: $OUTPUT_ROOT"

    # 显示处理结果统计
    echo ""
    echo "📊 处理结果统计:"
    find "$OUTPUT_ROOT" -name "eef_14d" -type f | wc -l | xargs echo "   成功生成的文件数量:"

    # 显示目录大小
    if command -v du &> /dev/null; then
        du -sh "$OUTPUT_ROOT" | xargs echo "   输出目录大小:"
    fi

    echo ""
    echo "🧪 测试生成的文件..."
    # 随机选择一个生成的文件进行测试
    test_file=$(find "$OUTPUT_ROOT" -name "*.hdf5" -type f | head -1)
    if [ ! -z "$test_file" ]; then
        echo "📄 测试文件: $(basename "$test_file")"
        python test_eef_14d.py "$test_file" --visualize --output_dir "./eef_test_plots"
    fi
else
    echo "⏭️  跳过完整处理"
fi

echo ""
echo "🎯 使用说明:"
echo "   1. 输出文件格式: HDF5文件，包含数据集 'eef_14d'"
echo "   2. 数据维度: [帧数, 14]，其中14维 = 左臂7维 + 右臂7维"
echo "   3. 左臂7维: [x, y, z, qx, qy, qz, qw] (位置+四元数)"
echo "   4. 右臂7维: [x, y, z, qx, qy, qz, qw] (位置+四元数)"
echo "   5. 使用测试脚本验证: python test_eef_14d.py <输出文件>"
echo ""
echo "🧪 测试生成的数据:"
echo "   python test_eef_14d.py <某个输出文件> --visualize"

# 清理临时文件
if [ -d "./temp_test_input" ]; then
    echo ""
    read -p "是否删除临时测试文件？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "./temp_test_input" "./temp_test_output"
        echo "🧹 临时文件已清理"
    fi
fi

echo ""
echo "✅ 脚本执行完成！"