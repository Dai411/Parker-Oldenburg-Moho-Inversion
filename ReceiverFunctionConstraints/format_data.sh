#!/bin/bash

# 用法: ./format_data.sh [输入文件] [输出文件]

INPUT_FILE="${1:-data.mrc}"
OUTPUT_FILE="${2:-formatted_data.txt}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 文件 $INPUT_FILE 不存在"
    exit 1
fi

echo "正在处理文件: $INPUT_FILE"

# 方法1: 将所有空白字符（空格和制表符）统一转换为制表符，然后清理多余的空列
cat "$INPUT_FILE" | \
    # 将连续的空白字符（空格/制表符）替换为单个制表符
    sed 's/[[:space:]]\+/\t/g' | \
    # 删除行首的制表符
    sed 's/^\t//' | \
    # 删除行尾的制表符
    sed 's/\t$//' > "$OUTPUT_FILE"

echo "方法1完成: $OUTPUT_FILE (制表符分隔)"

# 方法2: 生成带表头的CSV文件（更容易在电子表格软件中查看）
CSV_FILE="${INPUT_FILE%.*}.csv"
echo "正在生成CSV文件: $CSV_FILE"

# 根据您的数据，推测列名
echo "X坐标,Y坐标,Z坐标,值1,纬度,经度,站点ID,值2,值3" > "$CSV_FILE"

cat "$INPUT_FILE" | \
    sed 's/[[:space:]]\+/,/g' | \
    sed 's/^,//' | \
    sed 's/,$//' >> "$CSV_FILE"

# 方法3: 创建一个对齐的文本报告（最适合在文本编辑器中查看）
ALIGNED_FILE="${INPUT_FILE%.*}_aligned.txt"

echo "正在生成对齐文本: $ALIGNED_FILE"
echo "================================================================" > "$ALIGNED_FILE"
echo "站点数据 - 格式化的表格" >> "$ALIGNED_FILE"
echo "================================================================" >> "$ALIGNED_FILE"
echo "" >> "$ALIGNED_FILE"

# 使用column命令进行对齐
printf "%-15s %-15s %-8s %-8s %-10s %-10s %-10s %-8s %-8s\n" \
    "X坐标" "Y坐标" "Z坐标" "值1" "纬度" "经度" "站点ID" "值2" "值3" >> "$ALIGNED_FILE"

printf "%s\n" "----------------------------------------------------------------" >> "$ALIGNED_FILE"

cat "$INPUT_FILE" | \
    sed 's/[[:space:]]\+/\t/g' | \
    awk -F'\t' '{
        # 清理空字段
        for(i=1;i<=NF;i++) {
            if($i=="") $i="-"
        }
        # 格式化为对齐的输出
        printf "%-15s %-15s %-8s %-8s %-10s %-10s %-10s %-8s %-8s\n", 
               $1, $2, $3, $4, $5, $6, $7, $8, $9
    }' >> "$ALIGNED_FILE"

echo "所有处理完成！"
echo ""
echo "生成的文件："
echo "  1. $OUTPUT_FILE - 制表符分隔（适合导入数据库）"
echo "  2. $CSV_FILE - CSV格式（适合Excel打开）"
echo "  3. $ALIGNED_FILE - 对齐文本（在编辑器中查看最整齐）"

# 显示前几行预览
echo ""
echo "=== 预览（前5行）==="
head -n 5 "$ALIGNED_FILE"
