#!/bin/bash

# 批量分析所有 TTFT 配置的脚本
# 为每个 method_capacity 组合生成独立的数据集统计报告

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# TTFT 日志目录
TTFT_DIR=${1:-"./ttft_logs"}

# 结果保存目录
RESULTS_DIR="ttft_results"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}批量分析 TTFT 配置${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "TTFT 日志目录: $TTFT_DIR"
echo "结果保存目录: $RESULTS_DIR"
echo ""

# 检查 TTFT 目录是否存在
if [ ! -d "$TTFT_DIR" ]; then
    echo "错误: 目录不存在: $TTFT_DIR"
    exit 1
fi

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 统计配置数量
config_count=$(find "$TTFT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ $config_count -eq 0 ]; then
    echo "错误: 在 $TTFT_DIR 中没有找到任何配置目录"
    exit 1
fi

echo -e "找到 ${GREEN}$config_count${NC} 个配置"
echo ""

# 处理计数器
processed=0
success=0
failed=0

# 遍历所有配置目录
for dir in "$TTFT_DIR"/*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi

    config=$(basename "$dir")
    processed=$((processed + 1))

    echo -e "${BLUE}[$processed/$config_count]${NC} 正在分析: ${GREEN}$config${NC}"

    # 设置输出文件
    txt_output="$RESULTS_DIR/${config}_datasets.txt"
    csv_output="$RESULTS_DIR/${config}_datasets.csv"
    json_output="$RESULTS_DIR/${config}_datasets.json"

    # 运行分析
    if python ttft_tools/compute_dataset_ttft.py "$dir" \
        --no-summary \
        -o "$txt_output" \
        --csv "$csv_output" \
        --json "$json_output" 2>&1 | grep -q "ERROR\|错误"; then
        echo "  ✗ 失败"
        failed=$((failed + 1))
    else
        echo "  ✓ 成功"
        echo "    - TXT:  $txt_output"
        echo "    - CSV:  $csv_output"
        echo "    - JSON: $json_output"
        success=$((success + 1))
    fi
    echo ""
done

# 生成汇总报告
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}分析完成！${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "总配置数: $config_count"
echo -e "成功: ${GREEN}$success${NC}"
echo -e "失败: ${failed:-0}"
echo ""
echo "结果保存在: $RESULTS_DIR/"
echo ""

# 列出生成的文件
echo "生成的报告文件:"
ls -1 "$RESULTS_DIR"/*.txt 2>/dev/null | while read file; do
    echo "  - $(basename "$file")"
done

echo ""
echo -e "${GREEN}提示: 可以用以下命令查看报告${NC}"
echo "  cat $RESULTS_DIR/{配置名}_datasets.txt"
echo "  或用 Excel 打开: $RESULTS_DIR/{配置名}_datasets.csv"
