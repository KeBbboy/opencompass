#!/bin/bash

# 批量运行实验脚本
# 遍历不同的 method 和 max_capacity_prompt 参数
# 使用环境变量方式传递参数

# ============ 配置参数 ============
GPU_ID=0
CONFIG_FILE="opencompass/configs/sparity_config/run_Longbench.py"

# 定义要测试的方法
METHODS=(
    "full_KIVI"
)

# KIVI 参数：测试不同的量化位数
K_BITS=(
    2
    4
)

V_BITS=(
    2
    4
)

# KIVI 参数：测试不同的 group_size
GROUP_SIZES=(
    32
)

# KIVI 参数：residual_length (保持固定)
RESIDUAL_LENGTH=32

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
echo "=========================================="

# 遍历所有组合
for method in "${METHODS[@]}"; do
    for k_bit in "${K_BITS[@]}"; do
        for v_bit in "${V_BITS[@]}"; do
            for group_size in "${GROUP_SIZES[@]}"; do
                echo ""
                echo "=========================================="
                echo "运行实验: method=$method, k_bit=$k_bit, v_bit=$v_bit, group_size=$group_size, residual_length=$RESIDUAL_LENGTH"
                echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "=========================================="

                # 运行实验（通过环境变量传递参数）
                LOG_FILE="$LOG_DIR/${method}_k${k_bit}_v${v_bit}_group${group_size}_residual${RESIDUAL_LENGTH}.log"
                echo "日志文件: $LOG_FILE"

                CUDA_VISIBLE_DEVICES=$GPU_ID \
                SPARITY_METHOD=$method \
                KIVI_K_BITS=$k_bit \
                KIVI_V_BITS=$v_bit \
                KIVI_GROUP_SIZE=$group_size \
                KIVI_RESIDUAL_LENGTH=$RESIDUAL_LENGTH \
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
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结果保存在: $LOG_DIR"
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
