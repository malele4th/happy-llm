#!/usr/bin/env bash
# 构建或增量更新向量索引
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 main.py --build "$@"
