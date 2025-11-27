#!/usr/bin/env python3
"""
可视化 SnapKV TopK 索引选择的脚本

使用方法:
    python visualize_topk_indices.py --input topk_indices_logs/snapkv_layer0_indices.json
    python visualize_topk_indices.py --input_dir topk_indices_logs --layer 0
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_indices_data(json_file):
    """加载保存的索引数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_indices_heatmap(data, save_path=None):
    """
    绘制 TopK 索引的 Heatmap

    每一行代表一个 attention head，显示该 head 选取的所有 token 索引
    """
    indices = np.array(data['indices'])  # [bsz, num_heads, topk]
    layer_idx = data['layer_idx']
    group_info = data.get('group_info', None)

    # 只取第一个 batch
    indices_single = indices[0]  # [num_heads, topk]
    num_heads, topk = indices_single.shape

    # 创建一个 binary matrix: [num_heads, max_seq_len]
    # 假设最大序列长度从索引中推断
    max_idx = indices_single.max() + 1

    binary_matrix = np.zeros((num_heads, max_idx))
    for head_idx in range(num_heads):
        selected_indices = indices_single[head_idx]
        binary_matrix[head_idx, selected_indices] = 1

    # 绘制 heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, num_heads * 0.3)))
    im = ax.imshow(binary_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Attention Head', fontsize=12)

    # 标题根据是否有 group 信息调整
    if group_info:
        title = f'Layer {layer_idx}: TopK Token Selection Pattern (GQA: {group_info["num_groups"]} groups)\n' \
                f'(Red/Dark = Selected, Yellow/Light = Not Selected)'
    else:
        title = f'Layer {layer_idx}: TopK Token Selection Pattern\n' \
                f'(Red/Dark = Selected, Yellow/Light = Not Selected)'
    ax.set_title(title, fontsize=14, pad=20)

    # 设置 y 轴刻度和标签
    if group_info:
        # 如果有 group 信息，显示 "Head X (G Y)" 格式
        group_assignments = group_info['group_assignments']
        ax.set_yticks(np.arange(num_heads))
        ax.set_yticklabels([f'Head {i} (G{group_assignments[i]})' for i in range(num_heads)])

        # 添加 group 分隔线
        heads_per_group = group_info['heads_per_group']
        for g in range(1, group_info['num_groups']):
            # 在每个 group 的边界处画一条水平线
            ax.axhline(y=g * heads_per_group - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
    else:
        # 没有 group 信息，只显示 head 编号
        ax.set_yticks(np.arange(num_heads))
        ax.set_yticklabels([f'Head {i}' for i in range(num_heads)])

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selected', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_indices_distribution(data, save_path=None):
    """
    绘制索引分布的柱状图

    统计每个位置被选中的频率（跨所有 head）
    """
    indices = np.array(data['indices'])  # [bsz, num_heads, topk]
    layer_idx = data['layer_idx']
    group_info = data.get('group_info', None)

    # 只取第一个 batch
    indices_single = indices[0]  # [num_heads, topk]

    # 统计每个索引被选中的次数
    max_idx = indices_single.max() + 1
    selection_count = np.zeros(max_idx)

    for head_indices in indices_single:
        for idx in head_indices:
            selection_count[idx] += 1

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(selection_count))
    bars = ax.bar(x, selection_count, color='steelblue', alpha=0.7)

    # 高亮显示被所有 head 选中的位置
    num_heads = indices_single.shape[0]
    for i, count in enumerate(selection_count):
        if count == num_heads:
            bars[i].set_color('red')
        elif count >= num_heads * 0.8:
            bars[i].set_color('orange')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Selection Count (across heads)', fontsize=12)

    # 标题根据是否有 group 信息调整
    if group_info:
        title = f'Layer {layer_idx}: Token Selection Frequency (GQA: {group_info["num_groups"]} groups)\n' \
                f'(Red = All heads, Orange = ≥80% heads)'
    else:
        title = f'Layer {layer_idx}: Token Selection Frequency\n' \
                f'(Red = All heads, Orange = ≥80% heads)'
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)

    # 添加图例
    red_patch = mpatches.Patch(color='red', label='Selected by all heads')
    orange_patch = mpatches.Patch(color='orange', label='Selected by ≥80% heads')
    blue_patch = mpatches.Patch(color='steelblue', label='Other')
    ax.legend(handles=[red_patch, orange_patch, blue_patch], loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_scores(data, save_path=None):
    """
    绘制注意力分数的分布（如果有保存）
    """
    if 'attn_scores' not in data:
        print("No attention scores found in data.")
        return

    attn_scores = np.array(data['attn_scores'])  # [bsz, num_heads, seq_len]
    layer_idx = data['layer_idx']

    # 只取第一个 batch
    scores_single = attn_scores[0]  # [num_heads, seq_len]

    # 绘制所有 head 的平均分数
    mean_scores = scores_single.mean(axis=0)
    std_scores = scores_single.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 子图1: 平均注意力分数
    x = np.arange(len(mean_scores))
    ax1.plot(x, mean_scores, linewidth=2, color='steelblue', label='Mean score')
    ax1.fill_between(x, mean_scores - std_scores, mean_scores + std_scores,
                      alpha=0.3, color='steelblue', label='±1 std')
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Attention Score (pooled)', fontsize=12)
    ax1.set_title(f'Layer {layer_idx}: Average Attention Scores Across Heads', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 子图2: 每个 head 的注意力分数 heatmap
    im = ax2.imshow(scores_single, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('Attention Head', fontsize=12)
    ax2.set_title(f'Layer {layer_idx}: Attention Scores per Head', fontsize=14)

    num_heads = scores_single.shape[0]
    ax2.set_yticks(np.arange(num_heads))
    ax2.set_yticklabels([f'Head {i}' for i in range(num_heads)])

    plt.colorbar(im, ax=ax2, label='Attention Score')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention scores plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_head_indices(data, num_heads_to_plot=4, save_path=None):
    """
    为每个 head 单独绘制其选择的索引（散点图）
    """
    indices = np.array(data['indices'])  # [bsz, num_heads, topk]
    layer_idx = data['layer_idx']

    # 只取第一个 batch
    indices_single = indices[0]  # [num_heads, topk]
    num_heads = indices_single.shape[0]

    # 选择要绘制的 head 数量
    heads_to_plot = min(num_heads_to_plot, num_heads)

    fig, axes = plt.subplots(heads_to_plot, 1, figsize=(16, 3 * heads_to_plot))
    if heads_to_plot == 1:
        axes = [axes]

    for i in range(heads_to_plot):
        ax = axes[i]
        head_indices = indices_single[i]

        # 绘制散点图
        y = np.ones_like(head_indices) * (i + 1)
        ax.scatter(head_indices, y, alpha=0.6, s=50, color='steelblue')

        ax.set_ylabel(f'Head {i}', fontsize=11)
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

        if i == 0:
            ax.set_title(f'Layer {layer_idx}: Selected Token Indices per Head', fontsize=14)

        if i == heads_to_plot - 1:
            ax.set_xlabel('Token Position', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-head indices plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_group_overlap_comparison(data, save_path=None):
    """
    计算并可视化同一个 GQA group 内重复选到的 token 百分比

    对于每个 group，计算：
    - 交集大小（所有 heads 都选中的 tokens 数量）
    - 重复率 = 交集 / topk (表示在 topk 个选择中，有多少比例是所有 heads 都同意的)
    - 全局重复率：所有 heads 都选中的 tokens 占 topk 的比例
    """
    indices = np.array(data['indices'])  # [bsz, num_heads, topk]
    layer_idx = data['layer_idx']
    group_info = data.get('group_info', None)

    if group_info is None:
        print("No GQA group information found. Skipping group overlap plot.")
        return

    # 只取第一个 batch
    indices_single = indices[0]  # [num_heads, topk]
    num_heads, topk = indices_single.shape

    # 获取分组信息
    heads_per_group = group_info['heads_per_group']
    num_groups = group_info['num_groups']

    # 为每个 group 计算统计信息
    group_intersections = []  # 交集大小（重复选到的数量）
    group_overlap_percentages = []  # 重复百分比（交集/topk）

    for group_idx in range(num_groups):
        # 获取该 group 内的所有 heads
        start_head = group_idx * heads_per_group
        end_head = start_head + heads_per_group
        group_heads_indices = indices_single[start_head:end_head]  # [heads_per_group, topk]

        # 转换为集合列表
        sets = [set(head_indices.tolist()) for head_indices in group_heads_indices]

        # 计算交集（所有 heads 都选中的 tokens）
        intersection = set.intersection(*sets) if sets else set()
        intersection_size = len(intersection)

        # 计算重复百分比：重复选到的数量 / topk
        overlap_percentage = intersection_size / topk if topk > 0 else 0

        group_intersections.append(intersection_size)
        group_overlap_percentages.append(overlap_percentage)

    # 计算全局重复率（所有 heads 都选中的 tokens）
    all_heads_sets = [set(head_indices.tolist()) for head_indices in indices_single]
    global_intersection = set.intersection(*all_heads_sets) if all_heads_sets else set()
    global_intersection_size = len(global_intersection)
    global_overlap_percentage = global_intersection_size / topk if topk > 0 else 0

    # 创建柱状图（包含各个 group 和全局）
    fig, ax = plt.subplots(figsize=(14, 6))

    # x 轴包含各个 group 和一个 "Global" 柱子
    x = np.arange(num_groups + 1)
    width = 0.6

    # 准备数据：各个 group + 全局
    all_percentages = group_overlap_percentages + [global_overlap_percentage]
    all_intersections = group_intersections + [global_intersection_size]

    # 为不同类型的柱子设置不同颜色
    colors = ['steelblue'] * num_groups + ['crimson']  # 全局用红色突出显示

    # 绘制柱状图
    bars = ax.bar(x, all_percentages, width, color=colors, alpha=0.8, edgecolor='black')

    # 设置标签
    ax.set_xlabel('Group Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overlap Ratio (Intersection / TopK)', fontsize=14, fontweight='bold')
    ax.set_title(f'Layer {layer_idx}: Token Overlap Ratio per Group + Global\n(TopK={topk}, Heads per Group={heads_per_group}, Total Heads={num_heads})',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Group {i}' for i in range(num_groups)] + ['Global\n(All Heads)'], fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上显示百分比和具体数量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 显示百分比
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 显示具体数量 (交集/topk)
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{all_intersections[i]}/{topk}',
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Per-Group Overlap'),
        Patch(facecolor='crimson', edgecolor='black', label='Global Overlap (All Heads)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved group overlap comparison plot to {save_path}")

        # 同时打印统计信息
        print(f"\nGroup Overlap Statistics (Layer {layer_idx}):")
        print(f"TopK = {topk}, Heads per Group = {heads_per_group}, Total Heads = {num_heads}")
        print(f"{'Group':<15} {'Intersection':<15} {'Overlap Ratio':<20}")
        print("-" * 55)
        for i in range(num_groups):
            print(f"Group {i:<9} {group_intersections[i]:<3}/{topk:<10} {group_overlap_percentages[i]*100:>6.2f}%")
        print("-" * 55)
        print(f"{'Global (All)':<15} {global_intersection_size:<3}/{topk:<10} {global_overlap_percentage*100:>6.2f}%")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize TopK indices from SnapKV')
    parser.add_argument('--input', type=str, help='Path to input JSON file')
    parser.add_argument('--input_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--layer', type=int, help='Specific layer to visualize (if using --input_dir)')
    parser.add_argument('--sample', type=int, help='Specific sample to visualize (0-9)')
    parser.add_argument('--dataset', type=str, help='Specific dataset name (if using organized folder structure)')
    parser.add_argument('--output_dir', type=str, default='kv_cache_logs/visualizations/topk_indices',
                       help='Directory to save visualization plots')
    parser.add_argument('--plot_types', nargs='+',
                       choices=['heatmap', 'distribution', 'attention', 'per_head', 'group_overlap', 'all'],
                       default=['all'],
                       help='Types of plots to generate')

    args = parser.parse_args()

    # 确定输入文件
    if args.input:
        json_files = [args.input]
    elif args.input_dir:
        input_dir = Path(args.input_dir)

        # 构建glob模式，支持新的文件夹结构（timestamp/dataset/cap_xxx/）
        # 使用递归 glob 模式 **/ 来支持任意深度的文件夹
        pattern_parts = []

        # 如果指定了数据集名称，在路径中添加数据集过滤
        # 否则使用 **/ 匹配任意深度
        if args.dataset:
            # 匹配: .../dataset_name/.../file.json
            pattern_parts.append(f'**/{args.dataset}/**/')
        else:
            # 匹配任意深度
            pattern_parts.append('**/')

        # 文件名模式
        if args.sample is not None:
            pattern_parts.append(f'*_sample{args.sample:03d}_')
        else:
            pattern_parts.append('*_sample*_')

        if args.layer is not None:
            pattern_parts.append(f'layer{args.layer}_indices.json')
        else:
            pattern_parts.append('*_indices.json')

        pattern = ''.join(pattern_parts)
        json_files = list(input_dir.glob(pattern))

        # 如果没有找到，尝试不带sample的旧格式
        if not json_files:
            pattern_parts = ['**/']
            if args.layer is not None:
                pattern_parts.append(f'*_layer{args.layer}_indices.json')
            else:
                pattern_parts.append('*_indices.json')
            pattern = ''.join(pattern_parts)
            json_files = list(input_dir.glob(pattern))
    else:
        print("Error: Must specify either --input or --input_dir")
        return

    if not json_files:
        print(f"No JSON files found!")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理每个文件
    for json_file in json_files:
        print(f"\nProcessing {json_file}...")
        data = load_indices_data(json_file)

        layer_idx = data['layer_idx']
        sample_id = data.get('sample_id', None)  # 兼容旧格式
        dataset_name = data.get('dataset_name', None)  # 数据集名称
        capacity = data.get('capacity', None)  # capacity 参数

        # 如果 JSON 中没有 dataset_name，但用户通过 --dataset 参数指定了，使用用户指定的
        if not dataset_name and args.dataset:
            dataset_name = args.dataset

        # 构建输出基础路径：output_dir/{dataset_name}/cap_{capacity}/
        output_base_dir = args.output_dir
        if dataset_name:
            output_base_dir = os.path.join(output_base_dir, dataset_name)
        if capacity is not None:
            output_base_dir = os.path.join(output_base_dir, f"cap_{capacity}")

        # 构建文件名（不包含 dataset_name，因为已经在文件夹中）
        name_parts = []
        if sample_id is not None:
            name_parts.append(f"sample{sample_id:03d}")
        name_parts.append(f"layer{layer_idx}")

        base_name = "_".join(name_parts)

        plot_types = args.plot_types
        if 'all' in plot_types:
            plot_types = ['heatmap', 'distribution', 'attention', 'per_head', 'group_overlap']

        # 生成各种图表，每种类型保存到独立的子文件夹
        # 文件夹结构：output_dir/{dataset_name}/cap_{capacity}/{plot_type}/sample_layer.png
        if 'heatmap' in plot_types:
            heatmap_dir = os.path.join(output_base_dir, 'heatmap')
            os.makedirs(heatmap_dir, exist_ok=True)
            save_path = os.path.join(heatmap_dir, f'{base_name}.png')
            plot_indices_heatmap(data, save_path)

        if 'distribution' in plot_types:
            distribution_dir = os.path.join(output_base_dir, 'distribution')
            os.makedirs(distribution_dir, exist_ok=True)
            save_path = os.path.join(distribution_dir, f'{base_name}.png')
            plot_indices_distribution(data, save_path)

        if 'attention' in plot_types:
            attention_dir = os.path.join(output_base_dir, 'attention')
            os.makedirs(attention_dir, exist_ok=True)
            save_path = os.path.join(attention_dir, f'{base_name}.png')
            plot_attention_scores(data, save_path)

        if 'per_head' in plot_types:
            per_head_dir = os.path.join(output_base_dir, 'per_head')
            os.makedirs(per_head_dir, exist_ok=True)
            save_path = os.path.join(per_head_dir, f'{base_name}.png')
            plot_per_head_indices(data, num_heads_to_plot=8, save_path=save_path)

        if 'group_overlap' in plot_types:
            group_overlap_dir = os.path.join(output_base_dir, 'group_overlap')
            os.makedirs(group_overlap_dir, exist_ok=True)
            save_path = os.path.join(group_overlap_dir, f'{base_name}.png')
            plot_group_overlap_comparison(data, save_path)

    print(f"\n✓ Visualization complete! Plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
