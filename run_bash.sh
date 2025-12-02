#!/bin/bash

# 批量运行实验脚本
# 遍历不同的 method, max_capacity_prompt 和 torch_dtype 参数
# 使用环境变量方式传递参数

# ============ 配置参数 ============
GPU_ID=0
CONFIG_FILE="opencompass/configs/sparity_config/run_Longbench.py"

# 定义要测试的方法
METHODS=(
    # "full"

    # "RQA_sum"
    # "RQA_mean"
    # "RQA_mean_softmax"

    "orthogonal_pivot"

    # "first_group_gqa"
    # "topk_gqa"
    # "topk_gqa_global"
    # "min_max_gqa"
    
    # "min_max_gqa_global"
    # "min_max_gqa_chunk"
    # "min_max_gqa_chunk_global"
    # "sum_gqa"
    # "sum_gqa_global"
    # "sum_gqa_chunk"
    # "sum_gqa_chunk_global"

    # "windowkv"
    # "windowkv_gqa"

    # "snapkv"
    # "pyramidkv"
    # "pyramidkv_gqa"
)

# 定义要测试的 max_capacity_prompt 值
CAPACITIES=(
    # 8192
    # 4096
    # 2048
    # 1024
    512
    256
    # 128
)

# 定义要测试的 torch_dtype 值
DTYPES=(
    # "float16"
    "bfloat16"
)

# ============ TTFT 测量配置 (完全解耦，可选功能) ============
ENABLE_TTFT=False                         # 启用/禁用 TTFT 测量
TTFT_SAVE_DIR="./ttft_logs"              # TTFT 日志保存目录
TTFT_SAVE_TO_FILE=False                   # 是否保存到文件

# ============ 脚本开始 ============

# 创建日志目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment_logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "批量实验开始"
echo "GPU: $GPU_ID"
echo "配置文件: $CONFIG_FILE"
echo "日志目录: $LOG_DIR"
echo "TTFT 测量: $ENABLE_TTFT"
if [ "$ENABLE_TTFT" = "True" ]; then
    echo "TTFT 保存目录: $TTFT_SAVE_DIR"
fi
echo "=========================================="

# 遍历所有组合
for method in "${METHODS[@]}"; do
    for capacity in "${CAPACITIES[@]}"; do
        for dtype in "${DTYPES[@]}"; do
            echo ""
            echo "=========================================="
            echo "运行实验: method=$method, capacity=$capacity, dtype=$dtype"
            echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="

            # 运行实验（通过环境变量传递参数）
            LOG_FILE="$LOG_DIR/${method}_capacity${capacity}_${dtype}.log"
            echo "日志文件: $LOG_FILE"

            # 设置 TTFT 保存路径（包含实验参数信息）
            TTFT_EXP_DIR="${TTFT_SAVE_DIR}/${method}_capacity${capacity}"

            CUDA_VISIBLE_DEVICES=$GPU_ID \
            SPARITY_METHOD=$method \
            MAX_CAPACITY_PROMPT=$capacity \
            TORCH_DTYPE=$dtype \
            RUN_TIMESTAMP=$TIMESTAMP \
            ENABLE_TTFT=$ENABLE_TTFT \
            TTFT_SAVE_DIR=$TTFT_EXP_DIR \
            TTFT_SAVE_TO_FILE=$TTFT_SAVE_TO_FILE \
            python run.py "$CONFIG_FILE" --debug 2>&1 | tee "$LOG_FILE"

            EXIT_CODE=${PIPESTATUS[0]}
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✓ 实验成功完成"
                echo "SUCCESS" >> "$LOG_FILE"
            else
                echo "✗ 实验失败 (退出码: $EXIT_CODE)"
                echo "FAILED: exit code $EXIT_CODE" >> "$LOG_FILE"
            fi

            # 休息几秒，让 GPU 冷却
            sleep 5
        done
    done
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结果保存在: $LOG_DIR"
if [ "$ENABLE_TTFT" = "True" ]; then
    echo "TTFT 日志保存在: $TTFT_SAVE_DIR"
fi
echo "=========================================="

# 生成摘要
echo ""
echo "实验摘要:"
total=0
success=0
failed=0

for log in "$LOG_DIR"/*.log; do
    total=$((total + 1))
    name=$(basename "$log" .log)

    if grep -q "SUCCESS" "$log" 2>/dev/null; then
        echo "  ✓ $name"
        success=$((success + 1))
    else
        echo "  ✗ $name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "总计: $total 个实验"
echo "成功: $success"
echo "失败: $failed"
