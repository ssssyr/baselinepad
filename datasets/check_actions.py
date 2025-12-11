import json
import os
import numpy as np


def show_stat(name, arr):
    arr = np.array(arr)
    print(
        f"{name}: mean {arr.mean(0)}, std {arr.std(0)}, "
        f"min {arr.min(0)}, max {arr.max(0)}"
    )


def main():
    feature_dir = "/mnt/sda/datasets/metaworldcorner3-features_button_press_v2"
    skip_step = 4   # keep consistent with training config

    with open(os.path.join(feature_dir, "dataset_rgb_s_d.json"), "r") as f:
        steps = json.load(f)

    episodes = {}
    for s in steps:
        ep = int(s["episode"])
        episodes.setdefault(ep, []).append(s)
    for ep in episodes:
        episodes[ep] = sorted(episodes[ep], key=lambda x: int(x.get("frame", 0)))

    poses, grips, deltas = [], [], []
    jump_records = []
    for ep, fs in episodes.items():
        for i in range(len(fs)):
            p0 = np.array(fs[i]["state"], dtype=float)
            poses.append(p0[:3])
            grips.append(p0[3])
            j = i + skip_step
            if j < len(fs):
                p1 = np.array(fs[j]["state"], dtype=float)
                d = p1[:3] - p0[:3]
                deltas.append(d)
                jump_records.append((np.linalg.norm(d), ep, i, j, d))

    print(f"Episodes: {len(episodes)}, Frames: {len(steps)}")
    show_stat("Pose(xyz)", poses)
    show_stat("Delta@skip_step(xyz)", deltas)
    grips_arr = np.array(grips)
    print(
        f"Gripper: mean {grips_arr.mean():.4f}, std {grips_arr.std():.4f}, "
        f"min {grips_arr.min():.4f}, max {grips_arr.max():.4f}"
    )

    # Top-10 largest jumps
    print("\nTop 10 largest pose jumps (by norm) across skip_step:")
    for k, (norm, ep, i, j, d) in enumerate(sorted(jump_records, reverse=True)[:10], 1):
        print(f"Top{k}: |delta|={norm:.4f}, ep={ep}, frame {i}->{j}, delta={d}")


if __name__ == "__main__":
    main()
