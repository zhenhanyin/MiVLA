import h5py

# 替换为你的文件路径（例如：'data.h5' 或 'your_file.hdf5'）
file_path = "/home/yinzhenhan/Human_Pre/dataset/Robotwin_gen_48d_process/adjust_bottle/aloha-agilex_clean_50/data/episode0.hdf5"

# 需要查看的数据集路径列表
datasets_to_check = [
    "endpose/left_endpose",
    "endpose/right_endpose",

]

try:
    # 打开HDF5文件（只读模式）
    with h5py.File(file_path, 'r') as f:
        for dataset_path in datasets_to_check:
            # 检查数据集是否存在
            if dataset_path not in f:
                print(f"警告：数据集 {dataset_path} 不存在")
                continue
            
            # 读取数据集
            dataset = f[dataset_path]
            # 获取前3帧数据（根据shape取前3行）
            first_3_frames = dataset[:3]  # 等价于 dataset[0:3]
            
            # 打印信息
            print(f"=== 数据集: {dataset_path} ===")
            print(f"形状: {dataset.shape}, 数据类型: {dataset.dtype}")
            print("前3帧内容:")
            print(first_3_frames)
            print("-" * 50)  # 分隔线

except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到，请检查路径是否正确")
except Exception as e:
    print(f"读取文件时出错：{str(e)}")