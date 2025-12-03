#!/usr/bin/env python3
import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from functools import partial
import time
import argparse
import sys

def process_single_file(file_path):
    """
    Process a single HDF5 file to extract action statistics
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        tuple: (file_path, success, min_vals, max_vals, error_msg, outlier_dims, action_dim)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            action_data = None
            
            # Try different possible action data paths for real robot data
            if "action" in f:
                action_data = f["action"][:]
            elif "actions" in f:
                action_data = f["actions"][:]
            elif "joint_action" in f:
                # Prefer to use individual joint components for more accurate statistics
                if "left_arm" in f["joint_action"] and "right_arm" in f["joint_action"]:
                    # Handle multi-joint action format
                    left_arm = f["joint_action/left_arm"][:]
                    left_gripper = f["joint_action/left_gripper"][:]
                    right_arm = f["joint_action/right_arm"][:]
                    right_gripper = f["joint_action/right_gripper"][:]
                    
                    # Handle dimension mismatch
                    if len(left_arm.shape) == 2 and len(left_gripper.shape) == 1:
                        left_gripper = left_gripper.reshape(-1, 1)
                    if len(right_arm.shape) == 2 and len(right_gripper.shape) == 1:
                        right_gripper = right_gripper.reshape(-1, 1)
                    
                    action_data = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                    print(f"Using individual joint components for {file_path}")
                elif "vector" in f["joint_action"]:
                    # Fallback to vector if individual components not available
                    action_data = f["joint_action/vector"][:]
                    print(f"Using joint_action/vector for {file_path}")
                else:
                    action_data = f["joint_action"][:]
            
            if action_data is None:
                return (file_path, False, None, None, "No action data found", [], 0)
            
            # Ensure action data is 2D
            if len(action_data.shape) == 1:
                action_data = action_data.reshape(-1, len(action_data))
            
            # Get action dimension
            action_dim = action_data.shape[1]
            
            # Check for outliers (absolute value > 4)
            outlier_dims = []
            if np.any(np.abs(action_data) > 4):
                outlier_dims = np.where(np.any(np.abs(action_data) > 4, axis=0))[0].tolist()
            
            # Calculate min/max for this file
            file_min = np.min(action_data, axis=0)
            file_max = np.max(action_data, axis=0)
            
            return (file_path, True, file_min, file_max, None, outlier_dims, action_dim)
            
    except Exception as e:
        return (file_path, False, None, None, str(e), [], 0)

def process_task_files(task_info):
    """
    Process all HDF5 files in a single task directory

    Args:
        task_info: tuple of (task_name, task_dir)

    Returns:
        dict: Results for this task
    """
    task_name, task_dir = task_info

    # Find all HDF5 files in this task directory (recursive search for nested, or flat structure)
    if task_name == "flat_task":
        # Flat structure: HDF5 files directly in the directory
        hdf5_files = glob(os.path.join(task_dir, "*.hdf5"))
    else:
        # Nested structure: search recursively in task directory
        hdf5_files = glob(os.path.join(task_dir, "**/*.hdf5"), recursive=True)
    
    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'error_files': [],
        'outlier_files': [],
        'global_min': None,
        'global_max': None,
        'action_dim': None
    }
    
    print(f"Processing task '{task_name}' with {len(hdf5_files)} files...")
    
    # Process each file in this task
    for file_path in tqdm(hdf5_files, desc=f"Task {task_name}"):
        file_path, success, file_min, file_max, error_msg, outlier_dims, action_dim = process_single_file(file_path)
        
        results['file_count'] += 1
        
        if success:
            results['success_count'] += 1
            
            # Set action dimension for this task
            if results['action_dim'] is None:
                results['action_dim'] = action_dim
            elif results['action_dim'] != action_dim:
                print(f"Warning: Inconsistent action dimensions in {file_path}. Expected {results['action_dim']}, got {action_dim}")
            
            # Update global min/max for this task
            if results['global_min'] is None:
                results['global_min'] = file_min
                results['global_max'] = file_max
            else:
                results['global_min'] = np.minimum(results['global_min'], file_min)
                results['global_max'] = np.maximum(results['global_max'], file_max)
            
            # Record outlier files
            if outlier_dims:
                results['outlier_files'].append((file_path, outlier_dims))
        else:
            results['error_files'].append((file_path, error_msg))
    
    return results

def collect_action_stats_multiprocess(root_dir, output_path, outlier_path, num_processes=4):
    """
    Collect action statistics from all real robot HDF5 files using multiprocessing

    Args:
        root_dir: Root directory of HDF5 files
        output_path: Output JSON file path
        outlier_path: Output text file path for outlier files
        num_processes: Number of processes to use
    """
    print(f"Starting multiprocess statistics calculation with {num_processes} processes...")

    # Check if directory structure is flat (direct HDF5 files) or nested (task directories)
    hdf5_files_direct = glob(os.path.join(root_dir, "*.hdf5"))
    task_dirs = [d for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]

    if len(hdf5_files_direct) > 0:
        # Flat structure: HDF5 files directly in root_dir
        print(f"Found {len(hdf5_files_direct)} HDF5 files in flat directory structure")
        task_infos = [("flat_task", root_dir)]
    else:
        # Nested structure: Task directories containing HDF5 files
        print(f"Found {len(task_dirs)} task directories")
        task_infos = [(task, os.path.join(root_dir, task)) for task in task_dirs]
    
    # Record start time
    start_time = time.time()
    
    # Process tasks in parallel
    with mp.Pool(processes=num_processes) as pool:
        task_results = pool.map(process_task_files, task_infos)
    
    # Aggregate results from all tasks
    total_file_count = 0
    total_success_count = 0
    all_error_files = []
    all_outlier_files = []
    global_min = None
    global_max = None
    final_action_dim = None
    
    for result in task_results:
        total_file_count += result['file_count']
        total_success_count += result['success_count']
        all_error_files.extend(result['error_files'])
        all_outlier_files.extend(result['outlier_files'])
        
        # Update global min/max across all tasks
        if result['global_min'] is not None:
            if global_min is None:
                global_min = result['global_min']
                global_max = result['global_max']
                final_action_dim = result['action_dim']
            else:
                global_min = np.minimum(global_min, result['global_min'])
                global_max = np.maximum(global_max, result['global_max'])
                
                # Check for consistent action dimensions
                if result['action_dim'] != final_action_dim:
                    print(f"Warning: Task {result['task_name']} has different action dimension: {result['action_dim']} vs {final_action_dim}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Generate statistics dictionary
    stat_dict = {
        "real_robot": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
            "file_count": total_success_count,
            "total_files_scanned": total_file_count,
            "action_dim": final_action_dim if final_action_dim is not None else 0,
            "processing_time_seconds": elapsed_time,
            "num_processes_used": num_processes,
            "tasks_processed": len(task_results)
        }
    }
    
    # Save statistics results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stat_dict, f, indent=4, ensure_ascii=False)
    
    # Save outlier files list
    with open(outlier_path, 'w', encoding='utf-8') as f:
        f.write(f"Outlier files (absolute value > 4) - Total: {len(all_outlier_files)}\n")
        f.write("=" * 80 + "\n")
        for file_path, dims in all_outlier_files:
            f.write(f"{file_path} - Outlier dimensions: {dims}\n")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"REAL ROBOT STATISTICS CALCULATION COMPLETED")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processes used: {num_processes}")
    print(f"Average time per process: {elapsed_time/num_processes:.2f} seconds")
    print(f"Files per second: {total_file_count/elapsed_time:.2f}")
    print(f"\nResults:")
    print(f"- Total files scanned: {total_file_count}")
    print(f"- Successfully processed: {total_success_count}")
    print(f"- Failed files: {len(all_error_files)}")
    print(f"- Files with outliers: {len(all_outlier_files)}")
    print(f"- Action dimensions: {final_action_dim if final_action_dim is not None else 'N/A'}")
    if global_min is not None:
        print(f"- Min values: {global_min}")
        print(f"- Max values: {global_max}")
    
    # Show task-wise breakdown
    print(f"\nTask-wise breakdown:")
    for result in task_results:
        print(f"- {result['task_name']}: {result['success_count']}/{result['file_count']} files, action_dim={result['action_dim']}")
    
    # Show some error examples
    if all_error_files:
        print(f"\nError examples (showing first 10 out of {len(all_error_files)}):")
        for i, (path, err) in enumerate(all_error_files[:10]):
            print(f"- {path}: {err}")
        if len(all_error_files) > 10:
            print(f"... and {len(all_error_files) - 10} more errors")

def main():
    """Main function to run the multiprocess statistics calculation"""
    parser = argparse.ArgumentParser(description='Calculate Real Robot dataset statistics')
    parser.add_argument('--root_dir', type=str, 
                       default="/home/yinzhenhan/Human_Pre/dataset/zhenji/put_the_cup_on_the_coaster",
                       help='Root directory of HDF5 files')
    parser.add_argument('--output_path', type=str, 
                       default="/home/yinzhenhan/Human_Pre/H_RDT/datasets/real_robot/stats_cup_to_coaster.json",
                       help='Output JSON file path for statistics')
    parser.add_argument('--outlier_path', type=str, 
                       default="/home/yinzhenhan/Human_Pre/H_RDT/datasets/real_robot/outlier_files_cup_to_coaster.txt",
                       help='Output text file path for outlier files')
    parser.add_argument('--num_processes', type=int, default=32,
                       help='Number of processes to use')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"- Root directory: {args.root_dir}")
    print(f"- Output file: {args.output_path}")
    print(f"- Outlier file: {args.outlier_path}")
    print(f"- Number of processes: {args.num_processes}")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.outlier_path), exist_ok=True)
    
    # Check if root directory exists
    if not os.path.exists(args.root_dir):
        print(f"Error: Root directory {args.root_dir} does not exist!")
        sys.exit(1)
    
    # Run the calculation
    collect_action_stats_multiprocess(
        root_dir=args.root_dir,
        output_path=args.output_path,
        outlier_path=args.outlier_path,
        num_processes=args.num_processes
    )
    
    print(f"\nResults saved to:")
    print(f"- Statistics: {args.output_path}")
    print(f"- Outliers: {args.outlier_path}")

if __name__ == "__main__":
    # Ensure proper multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()