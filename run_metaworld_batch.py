import numpy as np
import time
import gc
import torch
from tabulate import tabulate
from tqdm import tqdm

# Import the rollout function from run_metaworld.py
import sys
sys.path.insert(0, '/home/syr/code/prediction_with_action')
from run_metaworld import run_single_rollout, set_random_seed
from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, META_CONFIG


def main():
    # Configuration
    ROLLOUT_NUM = META_CONFIG['rollout_num']  # Use META_CONFIG for rollout number
    # Use META_CONFIG task_list for selective testing
    task_list = META_CONFIG.get('task_list', list(INSTRUCTIONS.keys()))

    print(f"Total tasks to run: {len(task_list)}")
    print(f"Tasks: {task_list}")
    print(f"Rollouts per task: {ROLLOUT_NUM}")
    print(f"Save videos: False (batch mode)")

    # Build agent - align with run_metaworld.py
    agent = DiffusionAgent(
        ckpt_path=META_CONFIG['ckpt_path'],
        vae_path=META_CONFIG['vae_path'],
        clip_path=META_CONFIG['clip_path'],
        denoise_steps=META_CONFIG['denoise_steps'],
        device_id=META_CONFIG.get('gpu_id', 0)  # Use META_CONFIG
    )

    # Store results
    results = []

    # Run all tasks
    print("\n" + "="*80)
    print("STARTING BATCH TESTING")
    print("="*80)

    for selected_id, task in enumerate(task_list):
        print(f"\n{'='*60}")
        print(f"Task: {task} (ID: {selected_id})")
        print(f"{'='*60}")

        success_count = 0
        for traj_idx in tqdm(range(ROLLOUT_NUM), desc=f"{task}"):
            # Call the exact same rollout function as run_metaworld.py
            # but with save_video=False to skip video saving
            success = run_single_rollout(
                agent, task, selected_id, traj_idx,
                META_CONFIG, INSTRUCTIONS,
                save_video=False  # Don't save videos in batch mode
            )
            if success:
                success_count += 1

            # Explicit cleanup after each rollout to match standalone behavior
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        success_rate = success_count / ROLLOUT_NUM
        results.append({
            'ID': selected_id,
            'Task': task,
            'Success': success_count,
            'Total': ROLLOUT_NUM,
            'Rate': f"{success_rate:.1%}"
        })
        print(f"  Result: {success_count}/{ROLLOUT_NUM} = {success_rate:.1%}")

    # Print results table
    print("\n" + "="*80)
    print("BATCH TESTING RESULTS")
    print("="*80)

    headers = ['ID', 'Task', 'Success/Total', 'Success Rate']
    table_data = [[r['ID'], r['Task'], f"{r['Success']}/{r['Total']}", r['Rate']] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Calculate and print overall statistics
    total_success = sum(r['Success'] for r in results)
    total_trials = sum(r['Total'] for r in results)
    overall_rate = total_success / total_trials if total_trials > 0 else 0

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total tasks: {len(task_list)}")
    print(f"Total trials: {total_trials}")
    print(f"Total successes: {total_success}")
    print(f"Overall success rate: {overall_rate:.2%}")
    print("="*80)

    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BATCH TESTING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid') + "\n\n")
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total tasks: {len(task_list)}\n")
        f.write(f"Total trials: {total_trials}\n")
        f.write(f"Total successes: {total_success}\n")
        f.write(f"Overall success rate: {overall_rate:.2%}\n")
        f.write("="*80 + "\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
