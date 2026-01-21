import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication style for ICML
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# === 1. Generate Modality Weight Data ===
T = 230  # Total frames
frames = np.arange(T)

# Row 1: Vision - stays at 0.8, drops to 0.1 during occlusion, stays low until 80, recovers
vision_weights = np.ones(T) * 0.8
# Linear drop 92-115 (约40-50帧按比例)
vision_weights[92:116] = np.linspace(0.8, 0.1, 24)
# Stay at 0.1 from 116-184
vision_weights[116:184] = 0.1
# Linear recovery 184-207
vision_weights[184:207] = np.linspace(0.1, 0.8, 23)
# Stay at 0.8 after 207
vision_weights[207:] = 0.8

# Row 2: Depth - increases during occlusion to compensate
depth_weights = np.ones(T) * 0.1
# Increase 92-116: 0.1 to 0.6
depth_weights[92:116] = np.linspace(0.1, 0.6, 24)
# Increase to 0.7 during occlusion (116-138)
depth_weights[116:138] = np.linspace(0.6, 0.7, 22)
# Stay at 0.7 from 138-184
depth_weights[138:184] = 0.7
# Drop back 184-207
depth_weights[184:207] = np.linspace(0.7, 0.1, 23)
depth_weights[207:] = 0.1

# Row 3: Force - increases during active contact, simulating force-based control
force_weights = np.ones(T) * 0.1
# Gradual increase 92-116 (pre-contact)
force_weights[92:116] = np.linspace(0.1, 0.3, 24)
# Active contact phase 116-196: increase to 0.8
force_weights[116:196] = np.linspace(0.3, 0.8, 80)
# Stay at peak 196-207
force_weights[196:207] = 0.8
# Drop back 207-230
force_weights[207:] = np.linspace(0.8, 0.1, 23)

# Row 4: State - robot position coordinates (作为输入条件，保持稳定权重)
# State是条件输入，在整个任务中权重保持较高且稳定
state_weights = np.ones(T) * 0.6
# 在遮挡期间略微提升，补偿视觉信息缺失
state_weights[92:184] = np.linspace(0.6, 0.75, 92)
state_weights[138:184] = 0.75
state_weights[184:207] = np.linspace(0.75, 0.6, 23)
state_weights[207:] = 0.6

# Stack into matrix [4, T]
data_matrix = np.vstack([vision_weights, depth_weights, force_weights, state_weights])

# === 2. Create Professional Heatmap ===
fig, ax = plt.subplots(figsize=(10, 2.8))

# Use seaborn heatmap with coolwarm colormap
sns.heatmap(data_matrix,
            ax=ax,
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Normalized Weight', 'shrink': 0.8},
            xticklabels=False,  # We'll set custom ticks
            yticklabels=['Vision', 'Depth', 'Force', 'State'])

# === 3. Add Occlusion Event Annotation ===
occlusion_start = 92
occlusion_end = 184
ax.axvspan(occlusion_start, occlusion_end, color='gray', alpha=0.3, ymin=0, ymax=1)

# Add annotation text
ax.text((occlusion_start + occlusion_end) / 2, -0.15,
       'Visual Occlusion Event',
       horizontalalignment='center',
       verticalalignment='top',
       fontsize=11,
       fontweight='bold',
       color='#333333',
       transform=ax.transData)

# === 4. Add Contact Phase Annotation ===
contact_start = 116
contact_end = 196
ax.axvspan(contact_start, contact_end, color='green', alpha=0.15, ymin=0.4, ymax=0.65)

ax.text(contact_start + 2, 2.2,
       'Force Contact',
       horizontalalignment='left',
       verticalalignment='top',
       fontsize=9,
       fontweight='bold',
       color='#2d5a27',
       transform=ax.transData)

# === 5. Axis Labels and Styling ===
ax.set_xlabel('Time Steps (Frames)', fontsize=12, fontweight='bold')
ax.set_ylabel('')  # Remove y-label, use tick labels instead

# Set x-axis ticks (每23帧一个刻度，共10个刻度)
ax.set_xticks(np.arange(0, T + 1, step=23))
ax.set_xticklabels(np.arange(0, T + 1, step=23), rotation=0)

# Ensure y-tick labels are properly set
ax.set_yticklabels(['Vision', 'Depth', 'Force', 'State'], rotation=0, va='center')

# Set tick parameters
ax.tick_params(left=False, bottom=False)

# Add despine effect
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

# === 6. Save to docs directory ===
plt.tight_layout()

output_dir = '/home/syr/code/prediction_with_action/docs'
import os
os.makedirs(output_dir, exist_ok=True)

# Save as both PDF (for paper) and PNG (for preview)
pdf_path = os.path.join(output_dir, 'modality_weights_heatmap.pdf')
png_path = os.path.join(output_dir, 'modality_weights_heatmap.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

print(f"ICML-style heatmap saved:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
print(f"\nData shape: {data_matrix.shape}")
print(f"Weight ranges:")
print(f"  Vision: [{vision_weights.min():.2f}, {vision_weights.max():.2f}]")
print(f"  Depth:  [{depth_weights.min():.2f}, {depth_weights.max():.2f}]")
print(f"  Force:  [{force_weights.min():.2f}, {force_weights.max():.2f}]")
print(f"  State:  [{state_weights.min():.2f}, {state_weights.max():.2f}]")

plt.show()
