import matplotlib.pyplot as plt

# ===== 数据 =====
dense_x = [25, 75, 115, 235]
dense_y = [59.2, 64.8, 71.3, 78.5]

# ✅ Dense 标签改成两行：第一行数字，第二行 Dense
dense_txt = ["131.8M\nDense", "460.4M\nDense", "678M\nDense", "1.53B\nDense"]

# ✅ 两个 MoE 点 x 完全相同（上下对齐）
moe_x = [135, 135]
moe_y = [83.2, 84.4]
moe_txt = ["1.57B MoE (4E)", "2.17B MoE (8E)"]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=150)
ax.set_facecolor("white")

ax.grid(True, linestyle="--", linewidth=1, alpha=0.45)
ax.set_xlim(0, 250)
ax.set_ylim(55, 90)
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([55, 60, 65, 70, 75, 80, 85, 90])

ax.set_xlabel("Inference Compute (GFLOPs)", fontsize=28, labelpad=12)
ax.set_ylabel("MT50 Success Rate (%)", fontsize=28, labelpad=12)
ax.tick_params(axis="both", labelsize=26, width=1.5, length=6)
# 加粗坐标轴线
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Dense line
ax.plot(
    dense_x, dense_y, "-o",
    linewidth=3.5, markersize=9,
    color="#3f4a52",
    label="Dense Baseline",
    zorder=3
)

# ✅ Dense 标签位置（你原来的逻辑保留）
dense_offsets = [
    (-40, 12),   # 131.8M：更靠近点
    (2, 10),     # 460.4M：左上
    (0, 16),     # 678M：正上
    (-30, -22),  # 1.53B：右下，避开最后线段
]
dense_align = [
    ("left", "bottom"),
    ("right", "bottom"),
    ("center", "bottom"),
    ("left", "top"),
]

for (x, y, t), (dx, dy), (ha, va) in zip(
    zip(dense_x, dense_y, dense_txt),
    dense_offsets,
    dense_align
):
    ax.annotate(
        t,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha, va=va,
        fontsize=26, color="black",
        linespacing=0.9,
        multialignment="center",
        zorder=6
    )

# MoE points
ax.plot(
    moe_x, moe_y, "--D",
    linewidth=3.0, markersize=9,
    color="#b22222",
    label="Demuse (MoE)",
    zorder=4
)

# MoE 文本位置
moe_offsets = [(-20, -6), (10, 14)]
moe_align = [("right", "bottom"), ("left", "bottom")]
for (x, y, t), (dx, dy), (ha, va) in zip(
    zip(moe_x, moe_y, moe_txt),
    moe_offsets,
    moe_align
):
    ax.annotate(
        t,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=26, color="#b22222",
        ha=ha, va=va,
        zorder=6
    )

# Efficiency arrow (从 Dense 右端指向 MoE 点)
ax.annotate(
    "",
    xy=(135, 83.2),
    xytext=(235, 78.5),
    arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=5),
    zorder=5
)
ax.text(145, 83.5, "Efficiency Gain", fontsize=26, color="#2ca02c", weight="bold")

# legend
leg = ax.legend(loc="lower right", fontsize=24, frameon=True)
leg.get_frame().set_edgecolor("#c0c0c0")
leg.get_frame().set_linewidth(1.2)
leg.get_frame().set_alpha(1.0)

# =========================
# ✅ 关键：保证标题/坐标轴标题不被裁剪
# 1) tight_layout 给顶部留空间（避免标题挤出）
plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.94])

# 2) 保存时不要用 bbox_inches="tight"（它最容易裁标题/轴标题）
plt.savefig("scaling_analysis_white.png", dpi=300, facecolor="white")
plt.savefig("scaling_analysis_white.svg", facecolor="white")

plt.show()
