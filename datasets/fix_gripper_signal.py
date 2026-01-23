
"""
Fix gripper signal in npz files.

Usage:
    python fix_gripper_signal.py --input /mnt/sda/datasets/real_data
"""

import os
import argparse
import numpy as np
from pathlib import Path

def fix_task_gripper(task_dir: str, gripper_value: int, dry_run: bool = False):
    """Fix gripper signal in all npz files of a task directory.

    Args:
        task_dir: Path to task directory containing npz files
        gripper_value: Target gripper value (0 or 1)
        dry_run: If True, only print what would be changed
    """
    npz_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.npz')])

    print(f"Processing task: {os.path.basename(task_dir)}")
    print(f"  Found {len(npz_files)} npz files")
    print(f"  Target gripper value: {gripper_value}")
    print(f"  Dry run: {dry_run}")
    print()

    modified_count = 0
    for npz_file in npz_files:
        npz_path = os.path.join(task_dir, npz_file)

        
        data = np.load(npz_path, allow_pickle=True)
        action = data['action']  

        
        unique_gripper = np.unique(action[:, -1])
        original_values = action[:, -1].copy()

        
        action[:, -1] = gripper_value

        
        n_changed = np.sum(original_values != gripper_value)
        if n_changed > 0:
            print(f"  {npz_file}: {n_changed}/{len(action)} frames changed")
            print(f"    Original gripper values: {unique_gripper}")

            if not dry_run:
                
                
                arrays = {key: data[key] for key in data.files}
                arrays['action'] = action
                np.savez_compressed(npz_path, **arrays)
                modified_count += 1

    print(f"\nModified {modified_count} files" if not dry_run else f"\nWould modify {modified_count} files")

def main():
    parser = argparse.ArgumentParser(description="Fix gripper signal in npz files")
    parser.add_argument("--input", type=str, default="/mnt/sda/datasets/real_data",
                        help="Input directory containing task folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be changed without modifying files")

    args = parser.parse_args()

    
    tasks = [
        ("Fill a cup one-third full with cola.", 1),   
        ("Close drawer-type parts box", 0),             
    ]

    print("=" * 60)
    print("Gripper Signal Fix Script")
    print("=" * 60)
    print()

    for task_name, gripper_value in tasks:
        task_dir = os.path.join(args.input, task_name)
        if os.path.exists(task_dir):
            print(f"{'=' * 60}")
            fix_task_gripper(task_dir, gripper_value, args.dry_run)
            print()
        else:
            print(f"WARNING: Task directory not found: {task_dir}")
            print()

    print("=" * 60)
    print("Done!")
    if args.dry_run:
        print("This was a dry run. Run without --dry-run to apply changes.")
    print("=" * 60)

if __name__ == "__main__":
    main()
