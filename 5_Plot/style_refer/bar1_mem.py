import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.style.use("ggplot")

# ========================================================
# ==================  在这里修改数据  ======================
# ========================================================

# 数据集名称（对应每个子图）
datasets = ["longMemEval", "FNSPID", "Paper","GIST1M"]

# 算法名称（对应每个柱子 / 图例项），顺序即图例顺序
methods = ["TimeFirst", "iRangeGraph", "DIGRA", "UNIFY", "Wow", "RecencyBucket"]

# 每个数据集下，每个算法对应的数值（Index Time, 单位: 秒）
# 顺序需与 methods 一一对应
data = {
   "longMemEval":  [3.51, 6.07, 6.13, 6.22, 5.39, 5.52],
    "FNSPID": [2.93, 5.48, 6.02, 6.37, 5.39, 5.48],
    "Paper":  [1.52, 2.89, 3.67, 2.63, 2.95, 3.06],
    "GIST1M": [3.58, 4.37, 4.89, 4.65, 4.28, 4.48],
    
}
# 注意：如果 methods 有 6 个，请确保上面每个数据集的列表长度也是 6，
# 这里示例数据长度不一致，仅作占位，请按需修改为与 methods 等长的数值列表。

# 每个数据集 y 轴的显示范围（下限, 上限）。留空 (None, None) 则根据该数据集的最大值
# 自动计算（下限为 0，上限留出 headroom 以容纳柱顶数值标签），适合数值范围较窄、
# 各数据集量级又不完全一致的情况；如需手动固定范围，直接填 (0, 550) 这样的数值即可。
ylims = {
    "longMemEval":  (None, None),
    "FNSPID": (None, None),
    "Paper": (None, None),
    "GIST1M": (None, None),
   
}

# 柱顶数值标签相对柱高预留的空间比例（上限 = 该子图最大值 * 此系数）
headroom_ratio = 1.15


colors = ["#7f7f7f","#3d85c6", "#8e7cc3", "#6aa84f", "#f6b26b", "#e74c3c"]
hatches = ["xx", "//", "\\\\", "xx", "..", "//"]

bar_width = 0.8
group_gap = 1.0  # 每组（数据集内每个算法）之间的间距系数
edge_linewidth = 2.0   # 柱子边框粗细
hatch_linewidth = 2.0  # 柱内斜纹线条粗细（需通过 rcParams 设置，bar() 的 linewidth 只控制边框）
plt.rcParams["hatch.linewidth"] = hatch_linewidth

# ========================================================
# ==================  绘图主体（一般无需修改） ===============
# ========================================================

n_methods = len(methods)
x_positions = np.arange(n_methods) * group_gap

# 四个数据集放在同一张图里，横向排列子图
dataset_groups = [datasets]

bars_for_legend = []
for group_idx, group in enumerate(dataset_groups, start=1):
    n_ds = len(group)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 6))
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, group):
        values = data.get(ds, [np.nan] * n_methods)

        ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for x, v, method, color, hatch in zip(x_positions, values, methods, colors, hatches):
            bar = ax.bar(
                x, v,
                width=bar_width,
                facecolor="white",
                edgecolor=color,
                hatch=hatch,
                linewidth=edge_linewidth,
                zorder=3,
            )
            if group_idx == 1 and ax is axes[0]:
                bars_for_legend.append(bar[0])

        ylo, yhi = ylims.get(ds, (None, None))
        if yhi is None:
            finite_values = [v for v in values if not np.isnan(v)]
            yhi = max(finite_values) * headroom_ratio if finite_values else 1.0
        if ylo is None:
            ylo = 0
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.ticklabel_format(style="plain", axis="y")
        ax.set_xticks([])
        ax.set_xlabel(ds, fontsize=26)
        ax.set_ylabel("Memory Overhead(GB)", fontsize=26)
        ax.tick_params(axis="y", labelsize=22)

    plt.tight_layout()
    out_name = f"mem_{'_'.join(group)}.png"
    fig.savefig(out_name, dpi=200, bbox_inches="tight")

# ── 图例单独生成一张图片 ────────────────────────────────────────
fig_leg = plt.figure(figsize=(max(8, 2 * n_methods), 1.0))
fig_leg.legend(
    bars_for_legend,
    methods,
    loc="center",
    ncol=n_methods,
    frameon=False,
    fontsize=26,
    handlelength=2.0,
    columnspacing=0.8,
)
plt.axis("off")
fig_leg.savefig("mem_legend.png", dpi=200, bbox_inches="tight", pad_inches=0.05)

plt.show()