# 折线图手工选点与点位微调机制

本机制用于手工控制最终 SL / CP / matched-recall 折线图。
核心流程：编号 ID 图 + 可编辑 CSV 表格。

## 一、文件

- 5_Plot/plot_manual_candidate_ids.py
  生成带候选 ID 的选点图。
- 5_Plot/build_manual_points_table.py
  生成可编辑表格 manual_latency_points.csv。
- 5_Plot/manual_latency_points.csv
  人工编辑的选点与微调表格。
- 5_Plot/apply_manual_latency_points.py
  把表格转换成 manual_latency_overrides.json。
- 5_Plot/manual_latency_overrides.json
  绘图脚本实际读取的手工覆盖配置。
- 5_Plot/reset_manual_latency_overrides.py
  清空手工覆盖，恢复自动曲线。
- 5_Plot/MANUAL_LATENCY_GUIDE.md
  本说明。

## 二、标准流程

    1. python 5_Plot/plot_manual_candidate_ids.py
    2. python 5_Plot/build_manual_points_table.py
    3. 编辑 5_Plot/manual_latency_points.csv
    4. python 5_Plot/apply_manual_latency_points.py
    5. bash render_latency_figures.sh

如果需要同时重绘柱状图：

    bash render_all_figures.sh

只重绘 matched-recall 图（fig_full_latency_at_recall_090/095）：

    python 5_Plot/fig_final_latency_at_recall.py

## 三、编号 ID 图

输出目录：

    4_Results/fig_full/manual_candidates/

文件名：

- SL: dataset_bucket_method.png/svg
- CP: cp_dataset_ptype_method.png/svg
- matched-recall: recall090/095_dataset_method.png/svg

图中：

- 灰色空心点：全部候选实测点；
- 点旁小数字：候选 ID；
- 彩色曲线和较大标记：当前锚点曲线（优先手工选择，否则自动）；
- 彩色加粗数字：当前锚点对应的候选 ID。

## 四、编辑 manual_latency_points.csv

主要列：

- section/dataset/key/method：定位曲线；
- id：候选 ID，对应 ID 图上的数字；
- recall/latency_ms：当前坐标；
- use：1 选用，0 不用；
- dr：Recall 偏移（matched-recall 中是 selectivity 偏移）；
- dlat_ms：latency 偏移，毫秒；
- scale：latency 倍率；
- interpolation：pchip 或 linear；
- params：参数配置，仅供参考。

初始状态：

- 自动锚点行 use=1；
- 其他候选行 use=0；
- dr=0，dlat_ms=0，scale=1。
常用操作：

- 手工选点：想要的候选行 use=1，不想要的自动锚点行 use=0；
- 微调点：修改 dr、dlat_ms、scale；
- 只微调自动锚点：保持 use 默认，直接改 dr / dlat_ms / scale；
- 全部 use=0 的曲线会保留自动结果，避免空曲线；
- 手工把 Recall 调整到大于 1.0 的点会被自动丢弃并打印 warning。

## 五、应用与重置

应用：

    python 5_Plot/apply_manual_latency_points.py

该脚本会把 CSV 中的最终坐标写入 manual_latency_overrides.json。
最终坐标 = recall + dr, latency 先加 dlat_ms 再乘 scale。

重置：

    python 5_Plot/reset_manual_latency_overrides.py

## 六、重新运行生成脚本时保留手工数据

- plot_manual_candidate_ids.py 只重绘 ID 图，不修改 CSV 和 JSON；
  它会把当前手工锚点画成彩色曲线，因此重新生成的 ID 图仍显示手工结果。
- build_manual_points_table.py 默认读取现有 CSV，并按
  section + dataset + key + method + id 保留 use、dr、dlat_ms、scale、
  interpolation 以及 recall/latency_ms。
- 因此重新运行这两个脚本不会丢失当前手工调整。
- 如果想从自动默认状态重新生成表格，使用：
      python 5_Plot/build_manual_points_table.py --reset

## 七、matched-recall 图说明

fig_full_latency_at_recall_090/095 的每个点现在来自对应选择率桶的
当前手工 SL 曲线：在目标 Recall 0.90/0.95 处读取 latency。
因此这两张图会跟随 SL 折线图的手工调整。

PreFilter 是精确基线：

- PreFilter 在每个选择率桶中只有一个候选点（Recall@10 = 1.0），
  因此目标 Recall 0.90 和 0.95 都会取用该点；
- 五个选择率桶的 PreFilter 点连成一条完整折线；
- 由于不同选择率桶的精确扫描 latency 不同，该折线不是水平线；
- 这两张图中的 PreFilter 使用普通大小标记，并强制绘制连线。

matched-recall 自身也可以继续微调：对应行位于 CSV 的
section=matched_recall，key 为 0.90 或 0.95；其中 recall 列存放
的是 median selectivity，dr 是在该 x 坐标上的偏移。

## 八、常见问题

- matched-recall 图中缺少 PreFilter 折线时：
  1. 确认 CSV 中存在 section=matched_recall、method=prefilter 的行，
     且 use=1；
  2. 重新运行
     python 5_Plot/build_manual_points_table.py
     python 5_Plot/apply_manual_latency_points.py
  3. 重新运行
     python 5_Plot/fig_final_latency_at_recall.py
  4. 也可以用 python verify_latency_figures.py 检查图像文件是否齐全。
- PreFilter 折线颜色为 #C99700；每张 matched-recall 图有 5 个数据集，
  正常情况下 SVG 中可见 5 条该颜色的粗折线。

## 九、其他说明

- 手工曲线的 interpolation 默认 pchip；设为 linear 即严格折线连接。
- 如果实验数据更新导致候选 ID 变化，建议重新看 ID 图并检查 CSV。
