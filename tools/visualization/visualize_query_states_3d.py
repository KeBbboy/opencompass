#!/usr/bin/env python3
"""
3D可视化 Query States 的脚本（Window部分）

数据形状: [num_query_heads, window_size, head_dim]
注意: Query states 保存的是 window 部分，即参与注意力重要性计算的 query

3D坐标轴:
  - X轴: Token Position (window中的位置, 0 到 window_size-1)
  - Y轴: Head Dimension (head_dim 的维度索引, 0 到 head_dim-1)
  - Z轴: Query Value (该维度的实际值，不降维)

使用方法:
    # 单个head的3D可视化（不降维）
    python visualize_query_states_3d.py --input kv_cache_logs/query_states_logs/snapkv_sample000_layer0_querystates.json --mode matplotlib --head 0

    # 使用 plotly 交互式3D图
    python visualize_query_states_3d.py --input kv_cache_logs/query_states_logs/snapkv_sample000_layer0_querystates.json --mode plotly --head 0

    # 批量处理
    python visualize_query_states_3d.py --input_dir kv_cache_logs/query_states_logs --layer 0 --mode plotly --head 0
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_query_states_data(json_file):
    """加载保存的key states数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_query_states_3d_matplotlib(data, save_path=None, head_idx=0, sample_tokens=None, sample_dims=None, plot_style='wireframe'):
    """
    使用 matplotlib 绘制 Query States 的 3D 图
    X轴: Token Position, Y轴: Head Dimension, Z轴: Query Value (不降维)

    Args:
        data: 加载的数据字典
        save_path: 保存路径
        head_idx: 要可视化的 head 索引（必须指定）
        sample_tokens: 采样间隔（token维度），如果序列太长可以采样
        sample_dims: 采样间隔（dimension维度），如果维度太多可以采样
        plot_style: 绘图风格，'wireframe'（细线网格）或 'surface'（曲面）
    """
    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_query_heads:
        print(f"Warning: head_idx {head_idx} >= num_query_heads {num_query_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = query_states_single[head_idx]  # [window_size, head_dim]

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, window_size // 100)  # 最多显示100个tokens
    if sample_dims is None:
        sample_dims = max(1, head_dim // 64)  # 最多显示64个维度

    # 采样
    token_indices = np.arange(0, window_size, sample_tokens)
    dim_indices = np.arange(0, head_dim, sample_dims)

    # 创建网格
    X, Y = np.meshgrid(token_indices, dim_indices)
    Z_original = selected_keys[token_indices][:, dim_indices].T  # [len(dim_indices), len(token_indices)]

    # 检查是否有负数，如果有则同时生成平移版本
    has_negative = Z_original.min() < 0

    # 保存两个版本
    for version in ['original', 'shifted'] if has_negative else ['original']:
        if version == 'shifted':
            # 将数据向上平移，使最小值为0
            Z = Z_original - Z_original.min()
            # 修改保存路径，放在shifted子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                shifted_dir = os.path.join(dir_path, 'shifted')
                os.makedirs(shifted_dir, exist_ok=True)
                current_save_path = os.path.join(shifted_dir, filename)
            else:
                current_save_path = None
            title_suffix = " (Shifted to Non-negative)"
        else:
            Z = Z_original
            # 原始版本放在original子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                original_dir = os.path.join(dir_path, 'original')
                os.makedirs(original_dir, exist_ok=True)
                current_save_path = os.path.join(original_dir, filename)
            else:
                current_save_path = None
            title_suffix = ""

        # 创建 3D 图
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 根据 plot_style 选择绘图方式
        if plot_style == 'wireframe':
            # 使用 wireframe 绘制细线网格
            surf = ax.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.5, alpha=0.7)
            # 添加散点以显示数据点
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()
            scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', marker='o', s=8, alpha=0.8)
        else:
            # 使用 surface plot（原来的方式）
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='k', linewidth=0.3)
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()
            scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', marker='o', s=8, alpha=0.6)

        # 设置坐标轴标签
        ax.set_xlabel('Token Position in Sequence', fontsize=11)
        ax.set_ylabel('Head Dimension Index', fontsize=11)
        ax.set_zlabel('Query Value', fontsize=11)

        # 设置标题
        is_gqa = group_info.get('is_gqa', False)
        if is_gqa:
            title = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}{title_suffix}\n' \
                    f'(GQA: {group_info["num_query_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)\n' \
                    f'Sampled: {len(token_indices)}/{window_size} tokens, {len(dim_indices)}/{head_dim} dims'
        else:
            title = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}{title_suffix}\n' \
                    f'Sampled: {len(token_indices)}/{window_size} tokens, {len(dim_indices)}/{head_dim} dims'
        ax.set_title(title, fontsize=13, pad=20)

        # 添加 colorbar
        if plot_style == 'wireframe':
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        else:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Query Value', rotation=270, labelpad=20)

        # 调整视角
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if current_save_path:
            plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {current_save_path}")
        else:
            plt.show()

        plt.close()


def plot_query_states_3d_plotly(data, save_path=None, head_idx=0, sample_tokens=None, sample_dims=None):
    """
    使用 plotly 绘制交互式 Query States 的 3D 图
    X轴: Token Position, Y轴: Head Dimension, Z轴: Query Value (不降维)

    Args:
        data: 加载的数据字典
        save_path: 保存路径（HTML文件）
        head_idx: 要可视化的 head 索引
        sample_tokens: 采样间隔（token维度）
        sample_dims: 采样间隔（dimension维度）
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Error: plotly not installed. Please install it with: pip install plotly")
        return

    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_query_heads:
        print(f"Warning: head_idx {head_idx} >= num_query_heads {num_query_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = query_states_single[head_idx]  # [window_size, head_dim]

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, window_size // 200)  # plotly可以处理更多数据点
    if sample_dims is None:
        sample_dims = max(1, head_dim // 128)

    # 采样
    token_indices = np.arange(0, window_size, sample_tokens)
    dim_indices = np.arange(0, head_dim, sample_dims)

    # 创建网格
    X, Y = np.meshgrid(token_indices, dim_indices)
    Z = selected_keys[token_indices][:, dim_indices].T  # [len(dim_indices), len(token_indices)]

    # 创建 surface trace
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Query Value'),
        hovertemplate='Token: %{x}<br>Dim: %{y}<br>Value: %{z:.4f}<extra></extra>'
    )

    # 创建 figure
    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title_text = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}<br>' \
                     f'(GQA: {group_info["num_query_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)<br>' \
                     f'Sampled: {len(token_indices)}/{window_size} tokens, {len(dim_indices)}/{head_dim} dims'
    else:
        title_text = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}<br>' \
                     f'Sampled: {len(token_indices)}/{window_size} tokens, {len(dim_indices)}/{head_dim} dims'

    fig = go.Figure(data=[surface])

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='Token Position in Sequence',
            yaxis_title='Head Dimension Index',
            zaxis_title='Query Value',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1200,
        height=900,
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive 3D plot to {save_path}")
    else:
        fig.show()


def plot_query_states_heatmap(data, save_path=None, head_idx=0):
    """
    绘制 Query States 的 2D heatmap
    X轴: Token Position, Y轴: Head Dimension, 颜色: Query Value
    """
    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_query_heads:
        print(f"Warning: head_idx {head_idx} >= num_query_heads {num_query_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = query_states_single[head_idx]  # [window_size, head_dim]

    # 转置以便 Y轴是 head_dim
    heatmap_data = selected_keys.T  # [head_dim, window_size]

    # 创建 heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, head_dim * 0.05)))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Head Dimension Index', fontsize=12)

    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}: Query States Heatmap\n' \
                f'(GQA: {group_info["num_query_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
    else:
        title = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}: Query States Heatmap'
    ax.set_title(title, fontsize=14, pad=20)

    # 设置 y 轴刻度（如果维度太多，只显示部分）
    if head_dim <= 32:
        ax.set_yticks(np.arange(head_dim))
        ax.set_yticklabels([f'Dim {i}' for i in range(head_dim)])
    else:
        # 显示每隔N个维度
        step = max(1, head_dim // 32)
        yticks = np.arange(0, head_dim, step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'Dim {i}' for i in yticks])

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Query Value', rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_all_heads_subplots(data, save_path=None, sample_tokens=None, sample_dims=None, plot_style='wireframe'):
    """
    将同一层的所有heads放在同一张图中进行对比（subplot网格）

    Args:
        data: 加载的数据字典
        save_path: 保存路径
        sample_tokens: 采样间隔（token维度）
        sample_dims: 采样间隔（dimension维度）
        plot_style: 绘图风格
    """
    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single.shape

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, window_size // 50)  # 更少的采样点用于subplot
    if sample_dims is None:
        sample_dims = max(1, head_dim // 32)

    # 检查是否有负数
    has_negative = query_states_single.min() < 0

    # 保存两个版本
    for version in ['original', 'shifted'] if has_negative else ['original']:
        if version == 'shifted':
            # 将数据向上平移，使最小值为0
            query_states_to_plot = query_states_single - query_states_single.min()
            # 修改保存路径，放在shifted子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                shifted_dir = os.path.join(dir_path, 'shifted')
                os.makedirs(shifted_dir, exist_ok=True)
                current_save_path = os.path.join(shifted_dir, filename)
            else:
                current_save_path = None
            title_suffix = " (Shifted to Non-negative)"
        else:
            query_states_to_plot = query_states_single
            # 原始版本放在original子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                original_dir = os.path.join(dir_path, 'original')
                os.makedirs(original_dir, exist_ok=True)
                current_save_path = os.path.join(original_dir, filename)
            else:
                current_save_path = None
            title_suffix = ""

        # 计算subplot网格布局
        ncols = min(4, num_query_heads)  # 每行最多4个subplot
        nrows = (num_query_heads + ncols - 1) // ncols  # 向上取整

        # 创建figure和subplots
        fig = plt.figure(figsize=(ncols * 5, nrows * 4))

        for head_idx in range(num_query_heads):
            # 创建subplot
            ax = fig.add_subplot(nrows, ncols, head_idx + 1, projection='3d')

            # 获取该head的数据
            selected_keys = query_states_to_plot[head_idx]  # [window_size, head_dim]

            # 采样
            token_indices = np.arange(0, window_size, sample_tokens)
            dim_indices = np.arange(0, head_dim, sample_dims)

            # 创建网格
            X, Y = np.meshgrid(token_indices, dim_indices)
            Z = selected_keys[token_indices][:, dim_indices].T

            # 绘图
            if plot_style == 'wireframe':
                ax.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.3, alpha=0.6)
                X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
                scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', marker='o', s=5, alpha=0.7)
            else:
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.2)

            # 设置标签和标题
            ax.set_xlabel('Token Pos', fontsize=8)
            ax.set_ylabel('Dim', fontsize=8)
            ax.set_zlabel('Value', fontsize=8)
            ax.set_title(f'Head {head_idx}', fontsize=10, pad=5)
            ax.tick_params(labelsize=7)
            ax.view_init(elev=20, azim=45)

        # 总标题
        is_gqa = group_info.get('is_gqa', False)
        if is_gqa:
            suptitle = f'Sample {sample_id} - Layer {layer_idx}: All {num_query_heads} KV Heads Comparison{title_suffix}\n' \
                       f'(GQA: {group_info["num_query_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
        else:
            suptitle = f'Sample {sample_id} - Layer {layer_idx}: All {num_query_heads} Heads Comparison{title_suffix}'
        fig.suptitle(suptitle, fontsize=14, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if current_save_path:
            plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved all heads comparison subplot to {current_save_path}")
        else:
            plt.show()

        plt.close()


def plot_all_heads_comparison(data, save_path=None, dim_reduction='norm'):
    """
    绘制所有 heads 的对比图（使用降维后的值）
    X轴: Token Position, Y轴: KV Head Index, 颜色: Query Value (降维后)
    """
    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single.shape

    # 维度降维
    if dim_reduction == 'norm':
        z_values = np.linalg.norm(query_states_single, axis=2)  # [num_query_heads, window_size]
        z_label = 'L2 Norm of Key Vector'
    elif dim_reduction == 'mean':
        z_values = np.mean(query_states_single, axis=2)
        z_label = 'Mean of Key Vector'
    elif dim_reduction == 'max':
        z_values = np.max(query_states_single, axis=2)
        z_label = 'Max of Key Vector'
    else:
        z_values = np.linalg.norm(query_states_single, axis=2)
        z_label = 'L2 Norm of Key Vector'

    # 创建 heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, num_query_heads * 0.4)))
    im = ax.imshow(z_values, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('KV Head Index', fontsize=12)

    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title = f'Sample {sample_id} - Layer {layer_idx}: All Heads Comparison\n' \
                f'(GQA: {group_info["num_query_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
    else:
        title = f'Sample {sample_id} - Layer {layer_idx}: All Heads Comparison'
    ax.set_title(title, fontsize=14, pad=20)

    # 设置 y 轴刻度
    ax.set_yticks(np.arange(num_query_heads))
    ax.set_yticklabels([f'Head {i}' for i in range(num_query_heads)])

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(z_label, rotation=270, labelpad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_group_comparison(data, save_path=None, group_idx=0, sample_tokens=None, sample_dims=None, plot_style='wireframe', comparison_type='overlay'):
    """
    对比同一个 GQA group 内不同 query heads 的差异

    Args:
        data: 加载的数据字典
        save_path: 保存路径
        group_idx: 要可视化的 group 索引
        sample_tokens: 采样间隔（token维度）
        sample_dims: 采样间隔（dimension维度）
        plot_style: 绘图风格
        comparison_type: 对比类型 ('overlay', 'subplots', 'difference')
    """
    query_states = np.array(data['query_states'])  # [bsz, num_query_heads, window_size, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    query_states_single_original = query_states[0]  # [num_query_heads, window_size, head_dim]
    num_query_heads, window_size, head_dim = query_states_single_original.shape

    # 获取 GQA 信息
    num_key_value_groups = group_info.get('num_key_value_groups', 1)
    num_kv_heads = group_info.get('num_kv_heads', num_query_heads)

    if num_key_value_groups <= 1:
        print("Warning: This is not a GQA model or group info not available. Showing all heads.")
        heads_in_group = list(range(num_query_heads))
        group_idx = 0
    else:
        # 计算这个 group 包含哪些 query heads
        # 例如: 32 query heads, 4 kv heads, 8 groups -> 每个 group 有 8 个 query heads
        heads_per_group = num_query_heads // num_kv_heads
        start_head = group_idx * heads_per_group
        end_head = start_head + heads_per_group
        heads_in_group = list(range(start_head, min(end_head, num_query_heads)))

    print(f"Group {group_idx}: Query heads {heads_in_group[0]}-{heads_in_group[-1]} (total {len(heads_in_group)} heads)")

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, window_size // 50)
    if sample_dims is None:
        sample_dims = max(1, head_dim // 32)

    # 采样
    token_indices = np.arange(0, window_size, sample_tokens)
    dim_indices = np.arange(0, head_dim, sample_dims)

    # 检查是否有负数，决定是否生成 shifted 版本
    has_negative = query_states_single_original.min() < 0

    # 处理 original 和 shifted 两个版本
    for version in ['original', 'shifted'] if has_negative else ['original']:
        if version == 'shifted':
            # 将数据向上平移，使最小值为0
            query_states_single = query_states_single_original - query_states_single_original.min()
            # 修改保存路径，放在 comparison_type/shifted 子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                # 移除文件名中的 comparison_type 部分（因为已经在文件夹名中了）
                filename = filename.replace(f'_{comparison_type}', '')
                shifted_dir = os.path.join(dir_path, comparison_type, 'shifted')
                os.makedirs(shifted_dir, exist_ok=True)
                current_save_path = os.path.join(shifted_dir, filename)
            else:
                current_save_path = None
            title_suffix = " (Shifted to Non-negative)"
        else:
            query_states_single = query_states_single_original.copy()
            # 原始版本放在 comparison_type/original 子文件夹
            if save_path:
                dir_path = os.path.dirname(save_path)
                filename = os.path.basename(save_path)
                # 移除文件名中的 comparison_type 部分（因为已经在文件夹名中了）
                filename = filename.replace(f'_{comparison_type}', '')
                original_dir = os.path.join(dir_path, comparison_type, 'original')
                os.makedirs(original_dir, exist_ok=True)
                current_save_path = os.path.join(original_dir, filename)
            else:
                current_save_path = None
            title_suffix = ""

        if comparison_type == 'overlay':
            # 叠加模式：在同一个 3D 图中显示多个 heads
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(111, projection='3d')

            colors = plt.cm.rainbow(np.linspace(0, 1, len(heads_in_group)))

            for i, head_idx in enumerate(heads_in_group):
                selected_queries = query_states_single[head_idx]  # [window_size, head_dim]

                # 创建网格
                X, Y = np.meshgrid(token_indices, dim_indices)
                Z = selected_queries[token_indices][:, dim_indices].T

                # 绘制
                if plot_style == 'wireframe':
                    ax.plot_wireframe(X, Y, Z, color=colors[i], linewidth=0.5, alpha=0.6, label=f'Head {head_idx}')
                else:
                    ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.4, edgecolor='k', linewidth=0.1)

            ax.set_xlabel('Token Position in Window', fontsize=11)
            ax.set_ylabel('Head Dimension Index', fontsize=11)
            ax.set_zlabel('Query Value', fontsize=11)
            ax.set_title(f'Sample {sample_id} - Layer {layer_idx} - Group {group_idx} Overlay{title_suffix}\n' +
                        f'Query Heads {heads_in_group[0]}-{heads_in_group[-1]} ({len(heads_in_group)} heads)',
                        fontsize=13, pad=20)
            ax.legend(loc='upper right', fontsize=8)
            ax.view_init(elev=25, azim=45)

        elif comparison_type == 'subplots':
            # Subplots 模式：每个 head 一个 subplot
            ncols = min(4, len(heads_in_group))
            nrows = (len(heads_in_group) + ncols - 1) // ncols

            fig = plt.figure(figsize=(ncols * 5, nrows * 4))

            for i, head_idx in enumerate(heads_in_group):
                ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

                selected_queries = query_states_single[head_idx]
                X, Y = np.meshgrid(token_indices, dim_indices)
                Z = selected_queries[token_indices][:, dim_indices].T

                if plot_style == 'wireframe':
                    ax.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.5, alpha=0.7)
                else:
                    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='k', linewidth=0.2)

                ax.set_xlabel('Token Pos', fontsize=8)
                ax.set_ylabel('Dim', fontsize=8)
                ax.set_zlabel('Value', fontsize=8)
                ax.set_title(f'Query Head {head_idx}', fontsize=10, pad=5)
                ax.tick_params(labelsize=7)
                ax.view_init(elev=20, azim=45)

            fig.suptitle(f'Sample {sample_id} - Layer {layer_idx} - Group {group_idx} Comparison{title_suffix}\n' +
                        f'Query Heads {heads_in_group[0]}-{heads_in_group[-1]}',
                        fontsize=14, y=0.98)

        elif comparison_type == 'difference':
            # 差异模式：计算与第一个 head 的差异
            reference_head = heads_in_group[0]
            reference_data = query_states_single[reference_head]  # [window_size, head_dim]

            ncols = min(3, len(heads_in_group) - 1)
            nrows = (len(heads_in_group) - 1 + ncols - 1) // ncols

            fig = plt.figure(figsize=(ncols * 5, nrows * 4))

            for i, head_idx in enumerate(heads_in_group[1:]):  # 跳过参考 head
                ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

                selected_queries = query_states_single[head_idx]

                # 计算差异
                diff = selected_queries - reference_data

                X, Y = np.meshgrid(token_indices, dim_indices)
                Z = diff[token_indices][:, dim_indices].T

                # 使用发散色图显示差异
                surf = ax.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.8, edgecolor='k', linewidth=0.2)

                ax.set_xlabel('Token Pos', fontsize=8)
                ax.set_ylabel('Dim', fontsize=8)
                ax.set_zlabel('Difference', fontsize=8)
                ax.set_title(f'Head {head_idx} - Head {reference_head}', fontsize=10, pad=5)
                ax.tick_params(labelsize=7)
                ax.view_init(elev=20, azim=45)

                # 添加 colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            fig.suptitle(f'Sample {sample_id} - Layer {layer_idx} - Group {group_idx} Difference{title_suffix}\n' +
                        f'Reference: Head {reference_head}',
                        fontsize=14, y=0.98)

        elif comparison_type == 'token_concat':
            # Token拼接模式：在token维度上拼接所有heads的数据
            # 每个 head 的数据: [window_size, head_dim]
            # 拼接后: [window_size * num_heads_in_group, head_dim]

            # 收集所有 heads 的数据并在 token 维度上拼接
            concatenated_data = []
            for head_idx in heads_in_group:
                concatenated_data.append(query_states_single[head_idx])  # [window_size, head_dim]

            # 在 token 维度 (axis=0) 上拼接
            concat_queries = np.concatenate(concatenated_data, axis=0)  # [window_size * num_heads, head_dim]
            concat_window_size, concat_head_dim = concat_queries.shape

            # 创建标记，用于在图中显示不同 head 的边界
            head_boundaries = [i * window_size for i in range(1, len(heads_in_group))]

            # 采样（如果序列太长）
            token_sample = max(1, concat_window_size // 200)  # 拼接后可能很长，适当增加采样
            token_indices = np.arange(0, concat_window_size, token_sample)
            dim_indices = np.arange(0, concat_head_dim, sample_dims)

            # 创建网格
            X, Y = np.meshgrid(token_indices, dim_indices)
            Z = concat_queries[token_indices][:, dim_indices].T  # [len(dim_indices), len(token_indices)]

            # 创建 3D 图
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection='3d')

            # 绘制
            if plot_style == 'wireframe':
                ax.plot_wireframe(X, Y, Z, color='steelblue', linewidth=0.5, alpha=0.7)
                X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
                scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', marker='o', s=5, alpha=0.8)
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            else:
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='k', linewidth=0.3)
                cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            # 添加垂直分隔线标记不同 head 的边界
            z_min, z_max = Z.min(), Z.max()
            for boundary in head_boundaries:
                # 只在采样点中画线
                if boundary in token_indices:
                    boundary_idx = np.where(token_indices == boundary)[0][0]
                    # 画一条垂直于XY平面的线
                    for dim_idx in range(0, len(dim_indices), max(1, len(dim_indices)//10)):
                        ax.plot([token_indices[boundary_idx]] * 2,
                               [dim_indices[dim_idx]] * 2,
                               [z_min, z_max],
                               'r--', linewidth=2, alpha=0.7)

            # 设置坐标轴标签
            ax.set_xlabel('Concatenated Token Position\n(Heads stacked along token dimension)', fontsize=11)
            ax.set_ylabel('Head Dimension Index', fontsize=11)
            ax.set_zlabel('Query Value', fontsize=11)

            # 添加 X 轴的 head 标记
            # 在每个 head 的中间位置添加文本标记
            for i, head_idx in enumerate(heads_in_group):
                mid_pos = i * window_size + window_size // 2
                if mid_pos in token_indices:
                    ax.text(mid_pos, 0, z_min, f'H{head_idx}',
                           fontsize=9, color='red', fontweight='bold')

            # 设置标题
            title = f'Sample {sample_id} - Layer {layer_idx} - Group {group_idx} Token Concatenation{title_suffix}\n' \
                   f'Query Heads {heads_in_group[0]}-{heads_in_group[-1]} ({len(heads_in_group)} heads concatenated)\n' \
                   f'Total tokens: {concat_window_size} ({window_size} per head × {len(heads_in_group)} heads)'
            ax.set_title(title, fontsize=13, pad=20)

            cbar.set_label('Query Value', rotation=270, labelpad=20)
            ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if current_save_path:
            plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
            print(f"  [{version}] Saved group comparison plot to {current_save_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Query States in 3D')
    parser.add_argument('--input', type=str, help='Path to single JSON file')
    parser.add_argument('--input_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='kv_cache_logs/visualizations/query_states',
                        help='Output directory for plots')
    parser.add_argument('--layer', type=int, help='Filter by layer index')
    parser.add_argument('--sample', type=int, help='Filter by sample id')
    parser.add_argument('--mode', type=str, default='matplotlib',
                        choices=['matplotlib', 'plotly', 'both', 'heatmap', 'comparison', 'subplots', 'group_comparison'],
                        help='Visualization mode (subplots: all heads in one figure, group_comparison: compare heads in same GQA group)')
    parser.add_argument('--head', type=int, default=0,
                        help='Head index to visualize (required for 3D and heatmap modes)')
    parser.add_argument('--all_heads', action='store_true',
                        help='Visualize all heads (will ignore --head parameter)')
    parser.add_argument('--group_idx', type=int, default=0,
                        help='GQA group index to visualize (for group_comparison mode)')
    parser.add_argument('--comparison_type', type=str, default='subplots',
                        choices=['overlay', 'subplots', 'difference', 'token_concat'],
                        help='Type of group comparison: overlay (all in one 3D plot), subplots (separate plots), difference (show differences), token_concat (concatenate heads along token dimension)')
    parser.add_argument('--sample_tokens', type=int, default=None,
                        help='Sample every N tokens (auto if not specified)')
    parser.add_argument('--sample_dims', type=int, default=None,
                        help='Sample every N dimensions (auto if not specified)')
    parser.add_argument('--plot_style', type=str, default='wireframe',
                        choices=['wireframe', 'surface'],
                        help='Plot style for matplotlib: wireframe (thin lines) or surface (solid)')

    args = parser.parse_args()

    # 在output_dir后面添加mode子文件夹
    if args.mode == 'both':
        # 'both'模式会生成matplotlib和plotly两种，不添加子文件夹
        final_output_dir = args.output_dir
    else:
        # 其他模式添加mode子文件夹
        final_output_dir = os.path.join(args.output_dir, args.mode)

    # 创建输出目录
    os.makedirs(final_output_dir, exist_ok=True)

    # 确定要处理的文件列表
    files_to_process = []

    if args.input:
        if os.path.exists(args.input):
            files_to_process.append(args.input)
        else:
            print(f"Error: File {args.input} does not exist")
            return
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory {args.input_dir} does not exist")
            return

        # 查找所有 querystates.json 文件
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('_querystates.json'):
                    files_to_process.append(os.path.join(root, file))
    else:
        print("Error: Please specify either --input or --input_dir")
        return

    if not files_to_process:
        print("No key states files found")
        return

    print(f"Found {len(files_to_process)} key states file(s)")

    # 处理每个文件
    for json_file in files_to_process:
        print(f"\nProcessing {json_file}")

        try:
            data = load_query_states_data(json_file)

            # 过滤
            if args.layer is not None and data['layer_idx'] != args.layer:
                print(f"  Skipping (layer {data['layer_idx']} != {args.layer})")
                continue

            if args.sample is not None and data['sample_id'] != args.sample:
                print(f"  Skipping (sample {data['sample_id']} != {args.sample})")
                continue

            # 生成输出文件名
            base_name = Path(json_file).stem  # 去掉 .json 扩展名

            # 确定要处理的heads
            if args.all_heads:
                # 获取所有head的数量
                query_states = np.array(data['query_states'])
                num_query_heads = query_states.shape[1]  # [bsz, num_query_heads, window_size, head_dim]
                heads_to_process = list(range(num_query_heads))
                print(f"  Processing all {num_query_heads} heads...")
            else:
                heads_to_process = [args.head]

            # 对每个head进行可视化
            for head_idx in heads_to_process:
                # 根据模式生成可视化
                if args.mode in ['matplotlib', 'both']:
                    if args.mode == 'both':
                        mode_dir = os.path.join(args.output_dir, 'matplotlib')
                        os.makedirs(mode_dir, exist_ok=True)
                        output_file = os.path.join(mode_dir, f"{base_name}_3d_head{head_idx}_matplotlib.png")
                    else:
                        output_file = os.path.join(final_output_dir, f"{base_name}_3d_head{head_idx}_matplotlib.png")
                    print(f"    Generating matplotlib 3D plot for head {head_idx}...")
                    plot_query_states_3d_matplotlib(data, output_file, head_idx, args.sample_tokens, args.sample_dims, args.plot_style)

                if args.mode in ['plotly', 'both']:
                    if args.mode == 'both':
                        mode_dir = os.path.join(args.output_dir, 'plotly')
                        os.makedirs(mode_dir, exist_ok=True)
                        output_file = os.path.join(mode_dir, f"{base_name}_3d_head{head_idx}_plotly.html")
                    else:
                        output_file = os.path.join(final_output_dir, f"{base_name}_3d_head{head_idx}_plotly.html")
                    print(f"    Generating plotly 3D plot for head {head_idx}...")
                    plot_query_states_3d_plotly(data, output_file, head_idx, args.sample_tokens, args.sample_dims)

                if args.mode == 'heatmap':
                    output_file = os.path.join(final_output_dir, f"{base_name}_heatmap_head{head_idx}.png")
                    print(f"    Generating heatmap for head {head_idx}...")
                    plot_query_states_heatmap(data, output_file, head_idx)

            # comparison 模式不需要遍历head
            if args.mode == 'comparison':
                output_file = os.path.join(final_output_dir, f"{base_name}_comparison_allheads.png")
                print(f"    Generating comparison plot for all heads...")
                plot_all_heads_comparison(data, output_file, dim_reduction='norm')

            # subplots 模式：将所有heads放在同一张图中
            if args.mode == 'subplots':
                output_file = os.path.join(final_output_dir, f"{base_name}_subplots_allheads.png")
                print(f"    Generating subplots for all heads in one figure...")
                plot_all_heads_subplots(data, output_file, args.sample_tokens, args.sample_dims, args.plot_style)

            # group_comparison 模式：对比同一 GQA group 内的 query heads
            if args.mode == 'group_comparison':
                output_file = os.path.join(final_output_dir, f"{base_name}_group{args.group_idx}_{args.comparison_type}.png")
                print(f"    Generating group comparison plot for group {args.group_idx} ({args.comparison_type} mode)...")
                plot_group_comparison(data, output_file, args.group_idx, args.sample_tokens, args.sample_dims, args.plot_style, args.comparison_type)

        except Exception as e:
            print(f"  Error processing file: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ All visualizations saved!")
    if args.mode == 'both':
        print(f"   📁 matplotlib outputs:")
        print(f"      - Original: {os.path.join(args.output_dir, 'matplotlib', 'original')}/")
        print(f"      - Shifted:  {os.path.join(args.output_dir, 'matplotlib', 'shifted')}/")
        print(f"   📁 plotly outputs: {os.path.join(args.output_dir, 'plotly')}/")
    elif args.mode in ['matplotlib', 'subplots', 'heatmap', 'group_comparison']:
        print(f"   📁 Base directory: {final_output_dir}/")
        print(f"      - Original: {os.path.join(final_output_dir, 'original')}/")
        print(f"      - Shifted:  {os.path.join(final_output_dir, 'shifted')}/ (if has negative values)")
        if args.mode == 'group_comparison':
            print(f"   📊 Group comparison mode: {args.comparison_type}")
            print(f"   🎯 GQA Group: {args.group_idx}")
    else:
        print(f"   📁 Output directory: {final_output_dir}/")

    print(f"\n📊 3D坐标轴说明:")
    print("  - X轴: Token Position (序列中的位置, 0 到 window_size-1)")
    print("  - Y轴: Head Dimension Index (head_dim 的维度索引, 0 到 head_dim-1)")
    print("  - Z轴: Query Value (该维度的实际值，不降维)")
    print(f"\n💡 提示:")
    print("  - 如果数据有负数，会自动生成两个版本：")
    print("    • original/ - 原始数据（保留负数）")
    print("    • shifted/  - 平移数据（最小值=0，更容易看出差异）")


if __name__ == '__main__':
    main()
