
import h5py
import numpy as np

def inspect_hdf5_file(file_path):
    """
    打开并检查一个HDF5文件的结构、数据形状和属性。
    """
    print(f"--- 正在检查文件: {file_path} ---\n")

    try:
        # 使用'r'模式（只读）打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            
            # === 1. 查看语言元数据 (LLM Descriptions) ===
            print(">>> 语言元数据 (Attributes):")
            
            # 检查 'llm_description' 属性是否存在并打印
            if 'llm_description' in f.attrs:
                print(f"  llm_description: {f.attrs['llm_description']}")
            else:
                print("  未找到 'llm_description' 属性。")

            # 检查可逆任务相关的属性
            if 'llm_description2' in f.attrs:
                print(f"  llm_description2: {f.attrs['llm_description2']}")
            if 'which_llm_description' in f.attrs:
                print(f"  which_llm_description: {f.attrs['which_llm_description']}")
            
            print("-" * 20)

            # === 2. 递归遍历并打印文件结构 ===
            print("\n>>> 文件内部结构 (Datasets and Groups):")
            
            def print_structure(name, obj):
                # 根据对象类型打印信息
                if isinstance(obj, h5py.Dataset):
                    # 如果是数据集，打印其名称、形状和数据类型
                    print(f"  - [Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    # 如果是组，只打印名称
                    print(f"  - [Group] {name}")

            # .visititems() 是一个方便的函数，可以递归访问所有对象
            f.visititems(print_structure)

            print("-" * 20)
            
            # === 3. 查看具体数据集的示例数据 ===
            print("\n>>> 示例数据预览:")
            
            # 尝试打印 'transforms/camera' 的第一帧数据
            camera_transform_path = 'transforms/camera'
            if camera_transform_path in f:
                camera_transform_data = f[camera_transform_path]
                print(f"  '{camera_transform_path}' 的第一帧数据 (shape={camera_transform_data.shape}):")
                # 使用 [0] 来获取第一帧的数据
                print(camera_transform_data[0])
            else:
                print(f"  未找到数据集 '{camera_transform_path}'。")

            # 尝试打印 'transforms/leftHand' 的第一帧数据
            left_hand_path = 'transforms/leftHand'
            if left_hand_path in f:
                left_hand_data = f[left_hand_path]
                print(f"\n  '{left_hand_path}' 的第一帧数据 (shape={left_hand_data.shape}):")
                print(left_hand_data[0])
            else:
                print(f"  未找到数据集 '{left_hand_path}'。")
                
    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{file_path}'")
    except Exception as e:
        print(f"发生错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 将这里的文件路径替换成你的文件路径
    # target_file = "/home/wangxuanhan/Dataset/HRDT_smoke_test/part1/add_remove_lid/0.hdf5"

    target_file = "/home/yinzhenhan/Human_Pre/H_RDT/datasets/robotwin2/single_task_clean/blocks_ranking_size/data/episode0.hdf5"
    
    
    inspect_hdf5_file(target_file)    


