#!/usr/bin/env bash
# 周报 RAG 通用入口，转发所有参数给 main.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 main.py "$@"
