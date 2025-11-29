#!/usr/bin/env python3
"""
TTFT 结果分析脚本
用于分析 TTFT 测量结果并生成统计报告
"""

import json
import glob
import os
import sys
from collections import defaultdict
import argparse


def load_ttft_logs(log_dir):
    """加载所有 TTFT 日志文件"""
    pattern = os.path.join(log_dir, '**/*.json')
    log_files = glob.glob(pattern, recursive=True)

    data = []
    for file in log_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                entry['file_path'] = file
                data.append(entry)
        except Exception as e:
            print(f"警告: 无法读取文件 {file}: {e}")

    return data


def analyze_ttft(data):
    """分析 TTFT 数据"""
    if not data:
        print("没有找到 TTFT 数据")
        return

    # 提取 TTFT 值
    ttft_values = [entry.get('ttft_seconds', 0) for entry in data if 'ttft_seconds' in entry]

    if not ttft_values:
        print("没有有效的 TTFT 数据")
        return

    # 计算统计信息
    import statistics

    stats = {
        'count': len(ttft_values),
        'mean': statistics.mean(ttft_values),
        'median': statistics.median(ttft_values),
        'stdev': statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0,
        'min': min(ttft_values),
        'max': max(ttft_values),
    }

    # 计算百分位数
    sorted_values = sorted(ttft_values)
    stats['p50'] = sorted_values[int(len(sorted_values) * 0.50)]
    stats['p90'] = sorted_values[int(len(sorted_values) * 0.90)]
    stats['p95'] = sorted_values[int(len(sorted_values) * 0.95)]
    stats['p99'] = sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 100 else sorted_values[-1]

    return stats


def group_by_method(data):
    """按方法分组统计"""
    grouped = defaultdict(list)

    for entry in data:
        # 从 extra_info 中获取 method
        method = entry.get('extra_info', {}).get('method', 'unknown')
        if 'ttft_seconds' in entry:
            grouped[method].append(entry['ttft_seconds'])

    return grouped


def print_report(data, output_file=None):
    """打印分析报告"""
    output = []

    def log(msg):
        output.append(msg)
        print(msg)

    log("=" * 80)
    log("TTFT 分析报告")
    log("=" * 80)

    # 整体统计
    log("\n整体统计:")
    log("-" * 80)
    overall_stats = analyze_ttft(data)

    if overall_stats:
        log(f"  样本数量: {overall_stats['count']}")
        log(f"  平均 TTFT: {overall_stats['mean']:.4f}s")
        log(f"  中位数 TTFT: {overall_stats['median']:.4f}s")
        log(f"  标准差: {overall_stats['stdev']:.4f}s")
        log(f"  最小值: {overall_stats['min']:.4f}s")
        log(f"  最大值: {overall_stats['max']:.4f}s")
        log(f"  P50: {overall_stats['p50']:.4f}s")
        log(f"  P90: {overall_stats['p90']:.4f}s")
        log(f"  P95: {overall_stats['p95']:.4f}s")
        log(f"  P99: {overall_stats['p99']:.4f}s")

    # 按方法分组统计
    log("\n按方法分组统计:")
    log("-" * 80)
    grouped = group_by_method(data)

    for method, ttft_values in sorted(grouped.items()):
        if not ttft_values:
            continue

        import statistics
        log(f"\n  方法: {method}")
        log(f"    样本数: {len(ttft_values)}")
        log(f"    平均: {statistics.mean(ttft_values):.4f}s")
        log(f"    中位数: {statistics.median(ttft_values):.4f}s")
        log(f"    最小值: {min(ttft_values):.4f}s")
        log(f"    最大值: {max(ttft_values):.4f}s")

    # 按任务分组
    log("\n按任务分组统计 (前10个任务):")
    log("-" * 80)
    task_grouped = defaultdict(list)

    for entry in data:
        task = entry.get('task_name', 'unknown')
        if 'ttft_seconds' in entry:
            task_grouped[task].append(entry['ttft_seconds'])

    for i, (task, ttft_values) in enumerate(sorted(task_grouped.items(), key=lambda x: len(x[1]), reverse=True)[:10]):
        import statistics
        log(f"  {i+1}. {task}: {len(ttft_values)} 样本, 平均 {statistics.mean(ttft_values):.4f}s")

    log("\n" + "=" * 80)

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))
        log(f"\n报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析 TTFT 测量结果')
    parser.add_argument('log_dir', nargs='?', default='./ttft_logs', help='TTFT 日志目录 (默认: ./ttft_logs)')
    parser.add_argument('-o', '--output', help='保存报告到文件')
    parser.add_argument('--csv', help='导出 CSV 格式数据')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"错误: 目录不存在: {args.log_dir}")
        sys.exit(1)

    print(f"正在读取 TTFT 日志: {args.log_dir}")
    data = load_ttft_logs(args.log_dir)

    if not data:
        print(f"在 {args.log_dir} 中没有找到 TTFT 日志文件")
        sys.exit(1)

    print(f"找到 {len(data)} 个 TTFT 记录\n")

    # 打印报告
    print_report(data, args.output)

    # 导出 CSV
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'task_name', 'ttft_seconds', 'total_time_seconds',
                           'total_tokens', 'method', 'input_tokens', 'output_tokens'])

            for entry in data:
                writer.writerow([
                    entry.get('timestamp', ''),
                    entry.get('task_name', ''),
                    entry.get('ttft_seconds', ''),
                    entry.get('total_time_seconds', ''),
                    entry.get('total_tokens', ''),
                    entry.get('extra_info', {}).get('method', ''),
                    entry.get('extra_info', {}).get('input_tokens', ''),
                    entry.get('extra_info', {}).get('output_tokens', ''),
                ])
        print(f"\nCSV 数据已导出到: {args.csv}")


if __name__ == '__main__':
    main()
