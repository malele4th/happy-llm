#!/usr/bin/env bash
# 仅检索，不调用 LLM
# 用法: ./scripts/search.sh "catchii家族房" [--auto-date] [--debug]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ $# -lt 1 ]; then
  echo "用法: $0 <查询文本> [main.py 额外参数]"
  exit 1
fi

QUERY="$1"
shift
python3 main.py --search "$QUERY" "$@"
