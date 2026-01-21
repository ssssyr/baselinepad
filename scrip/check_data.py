import numpy as np

# 读取NPZ文件
data = np.load('/mnt/sda/datasets/real_data/Squeeze hand sanitizer foam from bottle./episode_0000.npz')

print("Keys in NPZ file:")
for key in data.keys():
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# 查看force数据
if 'force' in data.keys():
    force = data['force']
    print(f"\nForce data shape: {force.shape}")
    print(f"Force first few rows:\n{force[:5]}")
elif 'ee_force' in data.keys():
    force = data['ee_force']
    print(f"\nee_force data shape: {force.shape}")
    print(f"ee_force first few rows:\n{force[:5]}")
else:
    print("\nSearching for force-related keys...")
    for key in data.keys():
        if 'force' in key.lower():
            print(f"Found: {key}")
