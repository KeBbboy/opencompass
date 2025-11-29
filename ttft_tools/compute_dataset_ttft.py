#!/usr/bin/env python3
"""
计算每个数据集的平均 TTFT
按数据集分组统计并生成汇总文件
"""

import json
import glob
import os
import sys
from collections import defaultdict
import argparse
import csv


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
            print(f"警告: 无法读取文件 {file}: {e}", file=sys.stderr)

    return data


def extract_dataset_name(task_name):
    """
    从 task_name 中提取数据集名称
    例如: 'LongBench_narrativeqa' -> 'narrativeqa'
          'LongBench_hotpotqa' -> 'hotpotqa'
    """
    if '_' in task_name:
        # 通常格式为 'LongBench_datasetname' 或 'benchmark_datasetname'
        parts = task_name.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[1:])  # 去掉前缀，保留数据集名
    return task_name


def compute_dataset_stats(data):
    """计算每个数据集的 TTFT 统计信息"""
    dataset_ttfts = defaultdict(list)
    dataset_throughputs = defaultdict(list)
    dataset_total_times = defaultdict(list)
    dataset_info = defaultdict(lambda: {
        'total_samples': 0,
        'total_tokens': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'method': None,
    })

    for entry in data:
        task_name = entry.get('task_name', 'unknown')
        dataset = extract_dataset_name(task_name)

        # 收集 TTFT 数据
        if 'ttft_seconds' in entry:
            dataset_ttfts[dataset].append(entry['ttft_seconds'])

        # 收集吞吐量数据
        if 'decode_throughput_tokens_per_sec' in entry:
            dataset_throughputs[dataset].append(entry['decode_throughput_tokens_per_sec'])

        # 收集总时间数据
        if 'total_time_seconds' in entry:
            dataset_total_times[dataset].append(entry['total_time_seconds'])

        # 收集其他信息
        extra_info = entry.get('extra_info', {})
        dataset_info[dataset]['total_samples'] += 1
        dataset_info[dataset]['total_tokens'] += entry.get('total_tokens', 0)
        dataset_info[dataset]['input_tokens'] += extra_info.get('input_tokens', 0)
        dataset_info[dataset]['output_tokens'] += extra_info.get('output_tokens', 0)

        if dataset_info[dataset]['method'] is None:
            dataset_info[dataset]['method'] = extra_info.get('method', 'unknown')

    # 计算统计信息
    results = []
    for dataset, ttft_values in sorted(dataset_ttfts.items()):
        if not ttft_values:
            continue

        import statistics

        avg_ttft = statistics.mean(ttft_values)
        median_ttft = statistics.median(ttft_values)
        min_ttft = min(ttft_values)
        max_ttft = max(ttft_values)
        std_ttft = statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0

        # 计算吞吐量统计
        throughput_values = dataset_throughputs.get(dataset, [])
        avg_throughput = statistics.mean(throughput_values) if throughput_values else 0
        median_throughput = statistics.median(throughput_values) if throughput_values else 0

        # 计算总时间统计
        total_time_values = dataset_total_times.get(dataset, [])
        avg_total_time = statistics.mean(total_time_values) if total_time_values else 0

        info = dataset_info[dataset]

        results.append({
            'dataset': dataset,
            'samples': len(ttft_values),
            'avg_ttft': avg_ttft,
            'median_ttft': median_ttft,
            'min_ttft': min_ttft,
            'max_ttft': max_ttft,
            'std_ttft': std_ttft,
            'avg_throughput': avg_throughput,
            'median_throughput': median_throughput,
            'avg_total_time': avg_total_time,
            'total_tokens': info['total_tokens'],
            'avg_input_tokens': info['input_tokens'] / info['total_samples'] if info['total_samples'] > 0 else 0,
            'avg_output_tokens': info['output_tokens'] / info['total_samples'] if info['total_samples'] > 0 else 0,
            'method': info['method'],
        })

    return results


def save_to_txt(results, output_file):
    """保存为文本格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 130 + "\n")
        f.write("数据集性能统计报告 (TTFT + 吞吐量)\n")
        f.write("=" * 130 + "\n\n")

        f.write(f"总数据集数量: {len(results)}\n")
        f.write(f"总样本数量: {sum(r['samples'] for r in results)}\n\n")

        f.write("-" * 130 + "\n")
        f.write(f"{'数据集':<25} {'样本':<6} {'平均TTFT':<11} {'中位TTFT':<11} {'平均吞吐':<13} {'中位吞吐':<13} {'平均总时':<11} {'方法':<15}\n")
        f.write(f"{'':25} {'':6} {'(秒)':<11} {'(秒)':<11} {'(tok/s)':<13} {'(tok/s)':<13} {'(秒)':<11} {'':<15}\n")
        f.write("-" * 130 + "\n")

        for r in results:
            f.write(
                f"{r['dataset']:<25} "
                f"{r['samples']:<6} "
                f"{r['avg_ttft']:<11.4f} "
                f"{r['median_ttft']:<11.4f} "
                f"{r['avg_throughput']:<13.2f} "
                f"{r['median_throughput']:<13.2f} "
                f"{r['avg_total_time']:<11.4f} "
                f"{r['method']:<15}\n"
            )

        f.write("-" * 130 + "\n\n")

        # 添加详细信息
        f.write("详细信息:\n")
        f.write("=" * 100 + "\n")

        for r in results:
            f.write(f"\n数据集: {r['dataset']}\n")
            f.write(f"  样本数: {r['samples']}\n")
            f.write(f"  平均 TTFT: {r['avg_ttft']:.4f}s\n")
            f.write(f"  中位数 TTFT: {r['median_ttft']:.4f}s\n")
            f.write(f"  TTFT 范围: [{r['min_ttft']:.4f}s - {r['max_ttft']:.4f}s]\n")
            f.write(f"  标准差: {r['std_ttft']:.4f}s\n")
            f.write(f"  平均吞吐量: {r['avg_throughput']:.2f} tokens/s\n")
            f.write(f"  中位数吞吐量: {r['median_throughput']:.2f} tokens/s\n")
            f.write(f"  平均总时间: {r['avg_total_time']:.4f}s\n")
            f.write(f"  平均输入 tokens: {r['avg_input_tokens']:.0f}\n")
            f.write(f"  平均输出 tokens: {r['avg_output_tokens']:.0f}\n")
            f.write(f"  方法: {r['method']}\n")

    print(f"✓ 文本报告已保存到: {output_file}")


def save_to_csv(results, output_file):
    """保存为 CSV 格式"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'dataset', 'samples', 'avg_ttft', 'median_ttft', 'min_ttft',
            'max_ttft', 'std_ttft', 'avg_throughput', 'median_throughput',
            'avg_total_time', 'avg_input_tokens', 'avg_output_tokens', 'method'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"✓ CSV 报告已保存到: {output_file}")


def save_to_json(results, output_file):
    """保存为 JSON 格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON 报告已保存到: {output_file}")


def print_summary(results):
    """打印摘要到终端"""
    print("\n" + "=" * 130)
    print("数据集性能统计摘要 (TTFT + 吞吐量)")
    print("=" * 130)

    print(f"\n总数据集数量: {len(results)}")
    print(f"总样本数量: {sum(r['samples'] for r in results)}")

    print("\n" + "-" * 130)
    print(f"{'数据集':<25} {'样本':<6} {'平均TTFT':<11} {'中位TTFT':<11} {'平均吞吐':<13} {'中位吞吐':<13} {'平均总时':<11} {'方法':<15}")
    print(f"{'':25} {'':6} {'(秒)':<11} {'(秒)':<11} {'(tok/s)':<13} {'(tok/s)':<13} {'(秒)':<11} {'':<15}")
    print("-" * 130)

    for r in sorted(results, key=lambda x: x['avg_ttft'], reverse=True):
        print(
            f"{r['dataset']:<25} "
            f"{r['samples']:<6} "
            f"{r['avg_ttft']:<11.4f} "
            f"{r['median_ttft']:<11.4f} "
            f"{r['avg_throughput']:<13.2f} "
            f"{r['median_throughput']:<13.2f} "
            f"{r['avg_total_time']:<11.4f} "
            f"{r['method']:<15}"
        )

    print("-" * 130)

    # 整体统计
    all_ttfts = [r['avg_ttft'] for r in results]
    all_throughputs = [r['avg_throughput'] for r in results if r['avg_throughput'] > 0]

    if all_ttfts:
        import statistics
        print(f"\n整体统计:")
        print(f"  平均 TTFT: {statistics.mean(all_ttfts):.4f}s")
        print(f"  中位数 TTFT: {statistics.median(all_ttfts):.4f}s")
        print(f"  TTFT 最小值: {min(all_ttfts):.4f}s (数据集: {min(results, key=lambda x: x['avg_ttft'])['dataset']})")
        print(f"  TTFT 最大值: {max(all_ttfts):.4f}s (数据集: {max(results, key=lambda x: x['avg_ttft'])['dataset']})")

        if all_throughputs:
            print(f"  平均吞吐量: {statistics.mean(all_throughputs):.2f} tokens/s")
            print(f"  中位数吞吐量: {statistics.median(all_throughputs):.2f} tokens/s")
            print(f"  吞吐量最小值: {min(all_throughputs):.2f} tokens/s")
            print(f"  吞吐量最大值: {max(all_throughputs):.2f} tokens/s")

    print("=" * 130 + "\n")


def compare_all_methods(base_dir, output_txt=None, output_csv=None):
    """比较所有方法的性能"""
    # 获取所有子目录（方法）
    method_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            method_dirs.append((item, item_path))

    if not method_dirs:
        print(f"错误: 在 {base_dir} 中没有找到方法目录")
        sys.exit(1)

    print(f"找到 {len(method_dirs)} 个方法目录")

    # 收集所有方法的统计数据
    all_method_results = {}
    all_datasets = set()

    for method_name, method_path in sorted(method_dirs):
        print(f"\n处理方法: {method_name}")
        data = load_ttft_logs(method_path)

        if not data:
            print(f"  警告: 没有找到 TTFT 日志文件")
            continue

        print(f"  找到 {len(data)} 个 TTFT 记录")
        results = compute_dataset_stats(data)

        if not results:
            print(f"  警告: 没有有效的统计数据")
            continue

        # 转换为字典格式，以数据集名为键
        method_dict = {r['dataset']: r for r in results}
        all_method_results[method_name] = method_dict
        all_datasets.update(method_dict.keys())

    if not all_method_results:
        print("错误: 没有找到任何有效的统计数据")
        sys.exit(1)

    # 生成比较报告
    sorted_datasets = sorted(all_datasets)
    sorted_methods = sorted(all_method_results.keys())

    # 打印到终端
    print("\n" + "=" * 150)
    print("所有方法性能比较 (平均 TTFT)")
    print("=" * 150)

    # 表头
    header = f"{'数据集':<25} "
    for method in sorted_methods:
        header += f"{method[:20]:<22} "
    print(header)
    print("-" * 150)

    # 数据行
    for dataset in sorted_datasets:
        row = f"{dataset:<25} "
        for method in sorted_methods:
            if dataset in all_method_results[method]:
                ttft = all_method_results[method][dataset]['avg_ttft']
                row += f"{ttft:>8.4f}s ({all_method_results[method][dataset]['samples']:>3}样本) "
            else:
                row += f"{'N/A':<22} "
        print(row)

    print("-" * 150)

    # 计算每个方法的平均值
    print("\n整体平均 TTFT:")
    for method in sorted_methods:
        ttfts = [r['avg_ttft'] for r in all_method_results[method].values()]
        if ttfts:
            import statistics
            avg_ttft = statistics.mean(ttfts)
            median_ttft = statistics.median(ttfts)
            print(f"  {method:<30}: 平均={avg_ttft:.4f}s, 中位数={median_ttft:.4f}s, 数据集数={len(ttfts)}")

    print("=" * 150 + "\n")

    # 保存到文本文件
    if output_txt:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 150 + "\n")
            f.write("所有方法性能比较 (平均 TTFT)\n")
            f.write("=" * 150 + "\n\n")

            # 表头
            header = f"{'数据集':<25} "
            for method in sorted_methods:
                header += f"{method[:20]:<22} "
            f.write(header + "\n")
            f.write("-" * 150 + "\n")

            # 数据行
            for dataset in sorted_datasets:
                row = f"{dataset:<25} "
                for method in sorted_methods:
                    if dataset in all_method_results[method]:
                        ttft = all_method_results[method][dataset]['avg_ttft']
                        row += f"{ttft:>8.4f}s ({all_method_results[method][dataset]['samples']:>3}样本) "
                    else:
                        row += f"{'N/A':<22} "
                f.write(row + "\n")

            f.write("-" * 150 + "\n\n")

            # 整体统计
            f.write("整体平均 TTFT:\n")
            for method in sorted_methods:
                ttfts = [r['avg_ttft'] for r in all_method_results[method].values()]
                if ttfts:
                    import statistics
                    avg_ttft = statistics.mean(ttfts)
                    median_ttft = statistics.median(ttfts)
                    f.write(f"  {method:<30}: 平均={avg_ttft:.4f}s, 中位数={median_ttft:.4f}s, 数据集数={len(ttfts)}\n")

            f.write("\n" + "=" * 150 + "\n")

        print(f"✓ 比较报告已保存到: {output_txt}")

    # 保存到 CSV 文件
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['dataset'] + sorted_methods
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for dataset in sorted_datasets:
                row = {'dataset': dataset}
                for method in sorted_methods:
                    if dataset in all_method_results[method]:
                        row[method] = all_method_results[method][dataset]['avg_ttft']
                    else:
                        row[method] = 'N/A'
                writer.writerow(row)

        print(f"✓ CSV 比较报告已保存到: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='计算每个数据集的平均 TTFT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础使用
  python compute_dataset_ttft.py ./ttft_logs

  # 保存为文本文件
  python compute_dataset_ttft.py ./ttft_logs -o dataset_ttft.txt

  # 同时生成 CSV 和 JSON
  python compute_dataset_ttft.py ./ttft_logs --csv dataset_ttft.csv --json dataset_ttft.json

  # 分析特定方法的结果
  python compute_dataset_ttft.py ./ttft_logs/RQA_sum_capacity512 -o rqa_sum_results.txt

  # 比较所有方法的性能
  python compute_dataset_ttft.py ./ttft_logs --all-methods -o results.txt --csv results.csv
        """
    )

    parser.add_argument('log_dir', nargs='?', default='./ttft_logs',
                       help='TTFT 日志目录 (默认: ./ttft_logs)')
    parser.add_argument('-o', '--output', help='保存文本报告到文件')
    parser.add_argument('--csv', help='保存 CSV 格式报告')
    parser.add_argument('--json', help='保存 JSON 格式报告')
    parser.add_argument('--no-summary', action='store_true', help='不打印终端摘要')
    parser.add_argument('--all-methods', action='store_true',
                       help='比较目录下所有方法的性能（将每个子目录视为一个方法）')

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"错误: 目录不存在: {args.log_dir}")
        sys.exit(1)

    # 如果启用了 all-methods 模式
    if args.all_methods:
        compare_all_methods(args.log_dir, args.output, args.csv)
        return

    print(f"正在读取 TTFT 日志: {args.log_dir}")
    data = load_ttft_logs(args.log_dir)

    if not data:
        print(f"错误: 在 {args.log_dir} 中没有找到 TTFT 日志文件")
        sys.exit(1)

    print(f"找到 {len(data)} 个 TTFT 记录")

    # 计算统计信息
    results = compute_dataset_stats(data)

    if not results:
        print("错误: 没有有效的统计数据")
        sys.exit(1)

    # 打印摘要
    if not args.no_summary:
        print_summary(results)

    # 保存到文件
    if args.output:
        save_to_txt(results, args.output)

    if args.csv:
        save_to_csv(results, args.csv)

    if args.json:
        save_to_json(results, args.json)

    # 如果没有指定任何输出文件，默认保存为文本
    if not (args.output or args.csv or args.json):
        default_output = os.path.join(os.path.dirname(args.log_dir) or '.', 'dataset_ttft_summary.txt')
        save_to_txt(results, default_output)


if __name__ == '__main__':
    main()
