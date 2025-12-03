#!/usr/bin/env python3
"""
Extract language instructions from LIBERO HDF5 files and save to CSV format.

This script processes all HDF5 files in LIBERO datasets, extracts the language_instruction
from problem_info, and creates a CSV file with format:
    normalized_instruction,original_instruction

Usage:
    python extract_language_instructions.py
"""

import h5py
import json
import os
import csv
from pathlib import Path


def normalize_instruction(instruction):
    """Convert language instruction to normalized format (lowercase, underscores instead of spaces)"""
    # Remove quotes if present
    instruction = instruction.strip('"\'')
    # Convert to lowercase and replace spaces with underscores
    normalized = instruction.lower().replace(' ', '_')
    # Remove any remaining special characters that might cause issues
    normalized = normalized.replace(',', '').replace('.', '').replace('!', '').replace('?', '')
    return normalized


def extract_instructions_from_hdf5(hdf5_path):
    """Extract language instruction from a single HDF5 file"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' in f and hasattr(f['data'], 'attrs'):
                problem_info = f['data'].attrs['problem_info']

                # Parse problem_info (it might be JSON string or dict)
                if isinstance(problem_info, str):
                    problem_info = json.loads(problem_info)

                if 'language_instruction' in problem_info:
                    original_instruction = problem_info['language_instruction']
                    normalized_instruction = normalize_instruction(original_instruction)

                    return normalized_instruction, original_instruction
                else:
                    print(f"⚠️  No language_instruction found in {hdf5_path}")
                    return None, None
            else:
                print(f"⚠️  No data group or attributes found in {hdf5_path}")
                return None, None

    except Exception as e:
        print(f"❌ Error processing {hdf5_path}: {str(e)}")
        return None, None


def find_hdf5_files(libero_datasets_path):
    """Find all HDF5 files in LIBERO datasets directory"""
    hdf5_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(libero_datasets_path):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))

    return sorted(hdf5_files)


def get_filename_without_extension(hdf5_path):
    """Extract filename without extension and _demo suffix"""
    filename = os.path.basename(hdf5_path)
    # Remove .hdf5 extension
    filename = filename.replace('.hdf5', '')
    return filename


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    libero_datasets_path = "/home/yinzhenhan/LIBERO/datasets"  # Fixed path
    output_csv_path = os.path.join(script_dir, 'task_instructions.csv')

    print(f"🔍 Searching for HDF5 files in: {libero_datasets_path}")

    # Check if LIBERO datasets directory exists
    if not os.path.exists(libero_datasets_path):
        print(f"❌ LIBERO datasets directory not found: {libero_datasets_path}")
        print("Please make sure the LIBERO repository is properly set up.")
        return

    # Find all HDF5 files
    hdf5_files = find_hdf5_files(libero_datasets_path)

    if not hdf5_files:
        print("❌ No HDF5 files found in LIBERO datasets directory")
        return

    print(f"📁 Found {len(hdf5_files)} HDF5 files")

    # Extract instructions
    instructions = []
    successful_count = 0

    for i, hdf5_path in enumerate(hdf5_files, 1):
        print(f"📄 [{i}/{len(hdf5_files)}] Processing: {os.path.basename(hdf5_path)}")

        normalized, original = extract_instructions_from_hdf5(hdf5_path)

        if normalized and original:
            filename_without_ext = get_filename_without_extension(hdf5_path)
            # Add period at the end of instruction if not already present
            original_with_period = original.strip()
            if not original_with_period.endswith('.'):
                original_with_period += '.'
            instructions.append((filename_without_ext, original_with_period))
            successful_count += 1
            print(f"✅ Extracted: '{original_with_period}'")
        else:
            print(f"❌ Failed to extract instruction from {os.path.basename(hdf5_path)}")

    # Sort instructions by filename for consistency
    instructions.sort(key=lambda x: x[0])

    # Write to CSV
    print(f"\n💾 Writing {len(instructions)} instructions to: {output_csv_path}")

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['filename', 'original_instruction'])

            # Write instructions
            for filename, original in instructions:
                writer.writerow([filename, original])

        print(f"✅ Successfully wrote {len(instructions)} instructions to {output_csv_path}")

        # Show statistics
        print(f"\n📊 Statistics:")
        print(f"   Total HDF5 files processed: {len(hdf5_files)}")
        print(f"   Successful extractions: {successful_count}")
        print(f"   Failed extractions: {len(hdf5_files) - successful_count}")
        print(f"   Success rate: {successful_count/len(hdf5_files)*100:.1f}%")

        # Show sample instructions
        if instructions:
            print(f"\n📝 Sample instructions (first 5):")
            for i, (filename, original) in enumerate(instructions[:5], 1):
                print(f"   {i}. {filename}, \"{original}\"")

            if len(instructions) > 5:
                print(f"   ... and {len(instructions) - 5} more instructions")

    except Exception as e:
        print(f"❌ Error writing CSV file: {str(e)}")


if __name__ == "__main__":
    main()