#!/usr/bin/env bash
# 单次问答（检索 + LLM 生成）
# 用法: ./scripts/query.sh "2025年12月catchii进展" --auto-date
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ $# -lt 1 ]; then
  echo "用法: $0 <问题> [main.py 额外参数]"
  exit 1
fi

QUESTION="$1"
shift
python3 main.py --query "$QUESTION" "$@"
