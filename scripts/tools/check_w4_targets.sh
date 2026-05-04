#!/usr/bin/env bash
# W4 推理验收脚本：解析 inference_test.py 输出，判定 M2 PASS/FAIL
set -e

LOG_FILE="${1:-/tmp/w4_run.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: log file not found: $LOG_FILE"
    exit 1
fi

mem_ratio=$(grep -oP 'Memory ratio\s*=\s*\K[0-9.]+' "$LOG_FILE" | tail -n1)
tok_speedup=$(grep -oP 'Speedup \(token\)\s*=\s*\K[0-9.]+' "$LOG_FILE" | tail -n1)

if [ -z "$mem_ratio" ] || [ -z "$tok_speedup" ]; then
    echo "ERROR: 无法从日志中提取指标"
    echo "  请确认 inference_test.py 已完整运行"
    exit 1
fi

pass=1
if awk "BEGIN {exit !($mem_ratio > 0.50)}"; then
    echo "FAIL memory: ratio=${mem_ratio} > 0.50 (目标 ≤0.50)"
    pass=0
else
    echo "PASS memory: ratio=${mem_ratio} ≤ 0.50"
fi

if awk "BEGIN {exit !($tok_speedup < 1.10)}"; then
    echo "FAIL speed:  token_speedup=${tok_speedup}x < 1.10x (目标 ≥1.10x)"
    pass=0
else
    echo "PASS speed:  token_speedup=${tok_speedup}x ≥ 1.10x"
fi

if [ "$pass" -eq 1 ]; then
    echo "=== OVERALL PASS ==="
    exit 0
else
    echo "=== OVERALL FAIL ==="
    exit 1
fi
