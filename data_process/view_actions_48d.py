import h5py
import numpy as np
import argparse

def print_action_data(action_data, start_frame=0, num_frames=5):
    """
    以格式化的方式打印 actions_48d 数据，并解释每个维度的含义。
    """
     
    total_frames, num_dims = action_data.shape
    
    if num_dims != 48:
        print(f"错误：数据集的维度不是48，而是 {num_dims}。")
        return
        
    print("="*80)
    print("           actions_48d 数据维度含义 (H-RDT 论文定义)")
    print("="*80)
    print("前18维: 双手手腕6D位姿 (位置 + 旋转)")
    print("  - 0-2  : 左手手腕位置 (X, Y, Z)")
    print("  - 3-8  : 左手手腕旋转 (6D连续表示, aa_representation)")
    print("  - 9-11 : 右手手腕位置 (X, Y, Z)")
    print("  - 12-17: 右手手腕旋转 (6D连续表示)")
    print("-"*80)
    print("后30维: 双手指尖3D位置 (各5个指尖)")
    print("  - 18-20: 左手拇指尖 (X, Y, Z)")
    print("  - 21-23: 左手食指尖 (X, Y, Z)")
    print("  - 24-26: 左手中指尖 (X, Y, Z)")
    print("  - 27-29: 左手无名指尖 (X, Y, Z)")
    print("  - 30-32: 左手小指尖 (X, Y, Z)")
    print("  - 33-35: 右手拇指尖 (X, Y, Z)")
    print("  - 36-38: 右手食指尖 (X, Y, Z)")
    print("  - 39-41: 右手中指尖 (X, Y, Z)")
    print("  - 42-44: 右手无名指尖 (X, Y, Z)")
    print("  - 45-47: 右手小指尖 (X, Y, Z)")
    print("="*80)

    # 设置numpy打印选项，使其更易读
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    end_frame = min(start_frame + num_frames, total_frames)
    
    print(f"\n正在显示第 {start_frame} 到 {end_frame-1} 帧的数据 (共 {total_frames} 帧):\n")

    for i in range(start_frame, end_frame):
        print(f"--- Frame {i} ---")
        frame_data = action_data[i]
        
        # 打印手腕位姿
        print("  Left Wrist Pose:")
        print(f"    Position (0-2):   {frame_data[0:3]}")
        print(f"    Rotation (3-8):   {frame_data[3:9]}")
        print("  Right Wrist Pose:")
        print(f"    Position (9-11):  {frame_data[9:12]}")
        print(f"    Rotation (12-17): {frame_data[12:18]}")
        
        # 打印指尖位置
        print("  Left Fingertips (18-32):")
        print(f"    Thumb:  {frame_data[18:21]}")
        print(f"    Index:  {frame_data[21:24]}")
        
        print("  Right Fingertips (33-47):")
        print(f"    Thumb:  {frame_data[33:36]}")
        print(f"    Index:  {frame_data[36:39]}")
        print("-" * 20 + "\n")


def main():
    parser = argparse.ArgumentParser(description='查看HDF5文件中的 actions_48d 数据集。')
    parser.add_argument('hdf5_path', type=str, help='HDF5文件的路径')
    parser.add_argument('--start_frame', type=int, default=0, help='从哪一帧开始显示')
    parser.add_argument('--num_frames', type=int, default=90, help='要显示的帧数 (设置为-1以显示全部)')
    
    args = parser.parse_args()
    
    try:
        with h5py.File(args.hdf5_path, 'r') as f:
            if 'actions_48d' in f:
                action_data = f['actions_48d'][:]
                
                num_to_show = action_data.shape[0] if args.num_frames == -1 else args.num_frames
                
                print_action_data(action_data, args.start_frame, num_to_show)
                
            else:
                print(f"错误：在文件 '{args.hdf5_path}' 中找不到数据集 'actions_48d'。")
                print("可用的数据集/组包括:", list(f.keys()))
                
    except FileNotFoundError:
        print(f"错误：文件不存在 '{args.hdf5_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    main()