#!/usr/bin/env python3
"""
3D可视化 Key States 的脚本

数据形状: [num_kv_heads, seq_len, head_dim]
3D坐标轴:
  - X轴: Token Position (序列中的位置, 0 到 seq_len-1)
  - Y轴: Head Dimension (head_dim 的维度索引, 0 到 head_dim-1)
  - Z轴: Key Value (该维度的实际值，不降维)

使用方法:
    # 单个head的3D可视化（不降维）
    python visualize_key_states_3d.py --input kv_cache_logs/key_states_logs/snapkv_sample000_layer0_keystates.json --mode matplotlib --head 0

    # 使用 plotly 交互式3D图
    python visualize_key_states_3d.py --input kv_cache_logs/key_states_logs/snapkv_sample000_layer0_keystates.json --mode plotly --head 0

    # 批量处理
    python visualize_key_states_3d.py --input_dir kv_cache_logs/key_states_logs --layer 0 --mode plotly --head 0
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_key_states_data(json_file):
    """加载保存的key states数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_key_states_3d_matplotlib(data, save_path=None, head_idx=0, sample_tokens=None, sample_dims=None, plot_style='wireframe'):
    """
    使用 matplotlib 绘制 Key States 的 3D 图
    X轴: Token Position, Y轴: Head Dimension, Z轴: Key Value (不降维)

    Args:
        data: 加载的数据字典
        save_path: 保存路径
        head_idx: 要可视化的 head 索引（必须指定）
        sample_tokens: 采样间隔（token维度），如果序列太长可以采样
        sample_dims: 采样间隔（dimension维度），如果维度太多可以采样
        plot_style: 绘图风格，'wireframe'（细线网格）或 'surface'（曲面）
    """
    key_states = np.array(data['key_states'])  # [bsz, num_kv_heads, seq_len, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    key_states_single = key_states[0]  # [num_kv_heads, seq_len, head_dim]
    num_kv_heads, seq_len, head_dim = key_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_kv_heads:
        print(f"Warning: head_idx {head_idx} >= num_kv_heads {num_kv_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = key_states_single[head_idx]  # [seq_len, head_dim]

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, seq_len // 100)  # 最多显示100个tokens
    if sample_dims is None:
        sample_dims = max(1, head_dim // 64)  # 最多显示64个维度

    # 采样
    token_indices = np.arange(0, seq_len, sample_tokens)
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
        ax.set_zlabel('Key Value', fontsize=11)

        # 设置标题
        is_gqa = group_info.get('is_gqa', False)
        if is_gqa:
            title = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}{title_suffix}\n' \
                    f'(GQA: {group_info["num_kv_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)\n' \
                    f'Sampled: {len(token_indices)}/{seq_len} tokens, {len(dim_indices)}/{head_dim} dims'
        else:
            title = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}{title_suffix}\n' \
                    f'Sampled: {len(token_indices)}/{seq_len} tokens, {len(dim_indices)}/{head_dim} dims'
        ax.set_title(title, fontsize=13, pad=20)

        # 添加 colorbar
        if plot_style == 'wireframe':
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        else:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Key Value', rotation=270, labelpad=20)

        # 调整视角
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if current_save_path:
            plt.savefig(current_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {current_save_path}")
        else:
            plt.show()

        plt.close()


def plot_key_states_3d_plotly(data, save_path=None, head_idx=0, sample_tokens=None, sample_dims=None):
    """
    使用 plotly 绘制交互式 Key States 的 3D 图
    X轴: Token Position, Y轴: Head Dimension, Z轴: Key Value (不降维)

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

    key_states = np.array(data['key_states'])  # [bsz, num_kv_heads, seq_len, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    key_states_single = key_states[0]  # [num_kv_heads, seq_len, head_dim]
    num_kv_heads, seq_len, head_dim = key_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_kv_heads:
        print(f"Warning: head_idx {head_idx} >= num_kv_heads {num_kv_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = key_states_single[head_idx]  # [seq_len, head_dim]

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, seq_len // 200)  # plotly可以处理更多数据点
    if sample_dims is None:
        sample_dims = max(1, head_dim // 128)

    # 采样
    token_indices = np.arange(0, seq_len, sample_tokens)
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
        colorbar=dict(title='Key Value'),
        hovertemplate='Token: %{x}<br>Dim: %{y}<br>Value: %{z:.4f}<extra></extra>'
    )

    # 创建 figure
    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title_text = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}<br>' \
                     f'(GQA: {group_info["num_kv_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)<br>' \
                     f'Sampled: {len(token_indices)}/{seq_len} tokens, {len(dim_indices)}/{head_dim} dims'
    else:
        title_text = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}<br>' \
                     f'Sampled: {len(token_indices)}/{seq_len} tokens, {len(dim_indices)}/{head_dim} dims'

    fig = go.Figure(data=[surface])

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='Token Position in Sequence',
            yaxis_title='Head Dimension Index',
            zaxis_title='Key Value',
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


def plot_key_states_heatmap(data, save_path=None, head_idx=0):
    """
    绘制 Key States 的 2D heatmap
    X轴: Token Position, Y轴: Head Dimension, 颜色: Key Value
    """
    key_states = np.array(data['key_states'])  # [bsz, num_kv_heads, seq_len, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    key_states_single = key_states[0]  # [num_kv_heads, seq_len, head_dim]
    num_kv_heads, seq_len, head_dim = key_states_single.shape

    # 检查 head_idx 是否有效
    if head_idx >= num_kv_heads:
        print(f"Warning: head_idx {head_idx} >= num_kv_heads {num_kv_heads}, using head 0 instead")
        head_idx = 0

    # 获取指定 head 的数据
    selected_keys = key_states_single[head_idx]  # [seq_len, head_dim]

    # 转置以便 Y轴是 head_dim
    heatmap_data = selected_keys.T  # [head_dim, seq_len]

    # 创建 heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, head_dim * 0.05)))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Head Dimension Index', fontsize=12)

    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title = f'Sample {sample_id} - Layer {layer_idx} - KV Head {head_idx}: Key States Heatmap\n' \
                f'(GQA: {group_info["num_kv_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
    else:
        title = f'Sample {sample_id} - Layer {layer_idx} - Head {head_idx}: Key States Heatmap'
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
    cbar.set_label('Key Value', rotation=270, labelpad=20)

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
    key_states = np.array(data['key_states'])  # [bsz, num_kv_heads, seq_len, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    key_states_single = key_states[0]  # [num_kv_heads, seq_len, head_dim]
    num_kv_heads, seq_len, head_dim = key_states_single.shape

    # 自动采样设置
    if sample_tokens is None:
        sample_tokens = max(1, seq_len // 50)  # 更少的采样点用于subplot
    if sample_dims is None:
        sample_dims = max(1, head_dim // 32)

    # 检查是否有负数
    has_negative = key_states_single.min() < 0

    # 保存两个版本
    for version in ['original', 'shifted'] if has_negative else ['original']:
        if version == 'shifted':
            # 将数据向上平移，使最小值为0
            key_states_to_plot = key_states_single - key_states_single.min()
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
            key_states_to_plot = key_states_single
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
        ncols = min(4, num_kv_heads)  # 每行最多4个subplot
        nrows = (num_kv_heads + ncols - 1) // ncols  # 向上取整

        # 创建figure和subplots
        fig = plt.figure(figsize=(ncols * 5, nrows * 4))

        for head_idx in range(num_kv_heads):
            # 创建subplot
            ax = fig.add_subplot(nrows, ncols, head_idx + 1, projection='3d')

            # 获取该head的数据
            selected_keys = key_states_to_plot[head_idx]  # [seq_len, head_dim]

            # 采样
            token_indices = np.arange(0, seq_len, sample_tokens)
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
            suptitle = f'Sample {sample_id} - Layer {layer_idx}: All {num_kv_heads} KV Heads Comparison{title_suffix}\n' \
                       f'(GQA: {group_info["num_kv_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
        else:
            suptitle = f'Sample {sample_id} - Layer {layer_idx}: All {num_kv_heads} Heads Comparison{title_suffix}'
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
    X轴: Token Position, Y轴: KV Head Index, 颜色: Key Value (降维后)
    """
    key_states = np.array(data['key_states'])  # [bsz, num_kv_heads, seq_len, head_dim]
    layer_idx = data['layer_idx']
    sample_id = data['sample_id']
    group_info = data.get('group_info', {})

    # 只取第一个 batch
    key_states_single = key_states[0]  # [num_kv_heads, seq_len, head_dim]
    num_kv_heads, seq_len, head_dim = key_states_single.shape

    # 维度降维
    if dim_reduction == 'norm':
        z_values = np.linalg.norm(key_states_single, axis=2)  # [num_kv_heads, seq_len]
        z_label = 'L2 Norm of Key Vector'
    elif dim_reduction == 'mean':
        z_values = np.mean(key_states_single, axis=2)
        z_label = 'Mean of Key Vector'
    elif dim_reduction == 'max':
        z_values = np.max(key_states_single, axis=2)
        z_label = 'Max of Key Vector'
    else:
        z_values = np.linalg.norm(key_states_single, axis=2)
        z_label = 'L2 Norm of Key Vector'

    # 创建 heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, num_kv_heads * 0.4)))
    im = ax.imshow(z_values, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('KV Head Index', fontsize=12)

    is_gqa = group_info.get('is_gqa', False)
    if is_gqa:
        title = f'Sample {sample_id} - Layer {layer_idx}: All Heads Comparison\n' \
                f'(GQA: {group_info["num_kv_heads"]} KV heads, {group_info["num_query_heads"]} Q heads)'
    else:
        title = f'Sample {sample_id} - Layer {layer_idx}: All Heads Comparison'
    ax.set_title(title, fontsize=14, pad=20)

    # 设置 y 轴刻度
    ax.set_yticks(np.arange(num_kv_heads))
    ax.set_yticklabels([f'Head {i}' for i in range(num_kv_heads)])

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


def main():
    parser = argparse.ArgumentParser(description='Visualize Key States in 3D')
    parser.add_argument('--input', type=str, help='Path to single JSON file')
    parser.add_argument('--input_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='kv_cache_logs/visualizations/key_states',
                        help='Output directory for plots')
    parser.add_argument('--layer', type=int, help='Filter by layer index')
    parser.add_argument('--sample', type=int, help='Filter by sample id')
    parser.add_argument('--mode', type=str, default='matplotlib',
                        choices=['matplotlib', 'plotly', 'both', 'heatmap', 'comparison', 'subplots'],
                        help='Visualization mode (subplots: all heads in one figure)')
    parser.add_argument('--head', type=int, default=0,
                        help='Head index to visualize (required for 3D and heatmap modes)')
    parser.add_argument('--all_heads', action='store_true',
                        help='Visualize all heads (will ignore --head parameter)')
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

        # 查找所有 keystates.json 文件
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith('_keystates.json'):
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
            data = load_key_states_data(json_file)

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
                key_states = np.array(data['key_states'])
                num_kv_heads = key_states.shape[1]  # [bsz, num_kv_heads, seq_len, head_dim]
                heads_to_process = list(range(num_kv_heads))
                print(f"  Processing all {num_kv_heads} heads...")
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
                    plot_key_states_3d_matplotlib(data, output_file, head_idx, args.sample_tokens, args.sample_dims, args.plot_style)

                if args.mode in ['plotly', 'both']:
                    if args.mode == 'both':
                        mode_dir = os.path.join(args.output_dir, 'plotly')
                        os.makedirs(mode_dir, exist_ok=True)
                        output_file = os.path.join(mode_dir, f"{base_name}_3d_head{head_idx}_plotly.html")
                    else:
                        output_file = os.path.join(final_output_dir, f"{base_name}_3d_head{head_idx}_plotly.html")
                    print(f"    Generating plotly 3D plot for head {head_idx}...")
                    plot_key_states_3d_plotly(data, output_file, head_idx, args.sample_tokens, args.sample_dims)

                if args.mode == 'heatmap':
                    output_file = os.path.join(final_output_dir, f"{base_name}_heatmap_head{head_idx}.png")
                    print(f"    Generating heatmap for head {head_idx}...")
                    plot_key_states_heatmap(data, output_file, head_idx)

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
    elif args.mode in ['matplotlib', 'subplots', 'heatmap']:
        print(f"   📁 Base directory: {final_output_dir}/")
        print(f"      - Original: {os.path.join(final_output_dir, 'original')}/")
        print(f"      - Shifted:  {os.path.join(final_output_dir, 'shifted')}/ (if has negative values)")
    else:
        print(f"   📁 Output directory: {final_output_dir}/")

    print(f"\n📊 3D坐标轴说明:")
    print("  - X轴: Token Position (序列中的位置, 0 到 seq_len-1)")
    print("  - Y轴: Head Dimension Index (head_dim 的维度索引, 0 到 head_dim-1)")
    print("  - Z轴: Key Value (该维度的实际值，不降维)")
    print(f"\n💡 提示:")
    print("  - 如果数据有负数，会自动生成两个版本：")
    print("    • original/ - 原始数据（保留负数）")
    print("    • shifted/  - 平移数据（最小值=0，更容易看出差异）")


if __name__ == '__main__':
    main()
