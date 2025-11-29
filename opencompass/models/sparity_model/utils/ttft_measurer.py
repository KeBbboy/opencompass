"""
TTFT (Time To First Token) 测量器
完全解耦的设计，通过 HuggingFace 的生成回调机制工作
"""
import time
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from transformers import StoppingCriteria


class TTFTMeasurer(StoppingCriteria):
    """
    TTFT 测量器，作为 StoppingCriteria 集成到 HuggingFace generate 中
    不侵入模型代码，完全解耦
    """
    def __init__(
        self,
        save_dir: str = "./ttft_logs",
        task_name: str = "unknown",
        enable: bool = True,
        save_to_file: bool = True
    ):
        self.enable = enable
        self.save_to_file = save_to_file
        self.save_dir = save_dir
        self.task_name = task_name

        # 测量数据
        self.start_time = None
        self.ttft = None
        self.first_token_time = None
        self.total_tokens = 0
        self.step_times = []

        # 创建保存目录
        if self.save_to_file and self.enable:
            os.makedirs(self.save_dir, exist_ok=True)

    def start(self):
        """在生成开始前调用"""
        if self.enable:
            self.start_time = time.perf_counter()
            self.ttft = None
            self.first_token_time = None
            self.total_tokens = 0
            self.step_times = []

    def __call__(self, input_ids, scores, **kwargs):
        """
        StoppingCriteria 接口，每生成一个 token 都会被调用
        返回 False 表示继续生成
        """
        if not self.enable or self.start_time is None:
            return False

        current_time = time.perf_counter()
        self.total_tokens = input_ids.shape[1]

        # 记录第一个 token 的生成时间（TTFT）
        if self.ttft is None:
            self.ttft = current_time - self.start_time
            self.first_token_time = current_time
            print(f"🕐 TTFT: {self.ttft:.4f}s | Task: {self.task_name}")

        # 记录每步的时间
        self.step_times.append(current_time - self.start_time)

        return False  # 不停止生成

    def save_metrics(self, input_text: str = "", output_text: str = "", extra_info: Dict[str, Any] = None):
        """保存测量指标到文件"""
        if not self.enable or not self.save_to_file or self.start_time is None:
            return

        metrics = self.get_metrics()

        # 添加额外信息
        metrics['input_text'] = input_text[:200] + "..." if len(input_text) > 200 else input_text
        metrics['output_text'] = output_text[:200] + "..." if len(output_text) > 200 else output_text
        metrics['task_name'] = self.task_name

        if extra_info:
            metrics['extra_info'] = extra_info

        # 生成文件名：时间戳_任务名.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_task_name = self.task_name.replace('/', '_').replace('\\', '_')
        filename = f"{timestamp}_{safe_task_name}.json"
        filepath = os.path.join(self.save_dir, filename)

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"📁 TTFT metrics saved to: {filepath}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取所有测量指标"""
        if not self.enable or self.start_time is None:
            return {}

        total_time = time.perf_counter() - self.start_time

        metrics = {
            'ttft_seconds': self.ttft,
            'total_time_seconds': total_time,
            'total_tokens': self.total_tokens,
            'timestamp': datetime.now().isoformat(),
        }

        if self.ttft and self.total_tokens > 1:
            # 计算解码阶段的平均时间
            decode_time = total_time - self.ttft
            decode_tokens = self.total_tokens - 1  # 减去第一个 token

            metrics['decode_time_seconds'] = decode_time
            metrics['decode_tokens'] = decode_tokens
            metrics['avg_decode_time_per_token'] = decode_time / decode_tokens if decode_tokens > 0 else 0
            metrics['decode_throughput_tokens_per_sec'] = decode_tokens / decode_time if decode_time > 0 else 0

        return metrics

    def print_summary(self):
        """打印测量摘要"""
        if not self.enable:
            return

        metrics = self.get_metrics()
        if metrics:
            print("\n" + "="*70)
            print(f"TTFT Measurement Summary - {self.task_name}")
            print("-"*70)
            print(f"  Time to First Token (TTFT): {metrics.get('ttft_seconds', 0):.4f}s")
            print(f"  Total Generation Time:       {metrics.get('total_time_seconds', 0):.4f}s")
            print(f"  Total Tokens Generated:      {metrics.get('total_tokens', 0)}")
            if 'avg_decode_time_per_token' in metrics:
                print(f"  Avg Decode Time/Token:       {metrics['avg_decode_time_per_token']:.4f}s")
                print(f"  Decode Throughput:           {metrics['decode_throughput_tokens_per_sec']:.2f} tokens/s")
            print("="*70 + "\n")


def create_ttft_measurer(
    save_dir: str = "./ttft_logs",
    task_name: str = "unknown",
    enable: bool = True,
    save_to_file: bool = True
) -> TTFTMeasurer:
    """
    创建 TTFT 测量器的工厂函数

    Args:
        save_dir: TTFT 日志保存目录
        task_name: 任务名称，用于文件命名
        enable: 是否启用测量
        save_to_file: 是否保存到文件

    Returns:
        TTFTMeasurer 实例
    """
    return TTFTMeasurer(
        save_dir=save_dir,
        task_name=task_name,
        enable=enable,
        save_to_file=save_to_file
    )
