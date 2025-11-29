"""
TTFT 测量器测试脚本
用于验证 TTFT 功能是否正常工作
"""
import json
import os
from ttft_measurer import create_ttft_measurer


def test_ttft_measurer():
    """测试 TTFT 测量器基本功能"""
    print("=" * 70)
    print("Testing TTFT Measurer")
    print("=" * 70)

    # 创建测量器
    measurer = create_ttft_measurer(
        save_dir="./test_ttft_logs",
        task_name="test_task",
        enable=True,
        save_to_file=True
    )

    # 模拟开始测量
    measurer.start()
    print("✓ TTFT measurer started")

    # 模拟生成过程（通过 __call__ 方法）
    import torch

    # 模拟第一个 token 生成（prefill + 第一个 token）
    import time
    time.sleep(0.1)  # 模拟 prefill 时间

    dummy_input_ids = torch.randint(0, 1000, (1, 10))
    dummy_scores = torch.randn(1, 1000)

    # 第一次调用 - 记录 TTFT
    measurer(dummy_input_ids, dummy_scores)
    print("✓ First token generated (TTFT recorded)")

    # 模拟后续 token 生成
    for i in range(5):
        time.sleep(0.02)  # 模拟每个 token 生成时间
        dummy_input_ids = torch.cat([dummy_input_ids, torch.randint(0, 1000, (1, 1))], dim=1)
        measurer(dummy_input_ids, dummy_scores)

    print("✓ Additional tokens generated")

    # 保存结果
    measurer.save_metrics(
        input_text="This is a test input",
        output_text="This is a test output",
        extra_info={
            'test_mode': True,
            'model': 'test_model'
        }
    )
    print("✓ Metrics saved")

    # 打印摘要
    measurer.print_summary()

    # 验证文件是否生成
    log_files = [f for f in os.listdir("./test_ttft_logs") if f.endswith('.json')]
    if log_files:
        print(f"\n✓ Found {len(log_files)} log file(s)")

        # 读取并验证内容
        with open(os.path.join("./test_ttft_logs", log_files[0]), 'r') as f:
            data = json.load(f)

        print("\nLog file content preview:")
        print(f"  - TTFT: {data.get('ttft_seconds', 'N/A')}s")
        print(f"  - Total time: {data.get('total_time_seconds', 'N/A')}s")
        print(f"  - Total tokens: {data.get('total_tokens', 'N/A')}")
        print(f"  - Task name: {data.get('task_name', 'N/A')}")

        print("\n" + "=" * 70)
        print("✅ TTFT Measurer Test PASSED")
        print("=" * 70)
    else:
        print("\n❌ No log files found - Test FAILED")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_ttft_logs"):
        response = input("\nDelete test log files? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree("./test_ttft_logs")
            print("✓ Test files cleaned up")


if __name__ == "__main__":
    test_ttft_measurer()
