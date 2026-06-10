#!/usr/bin/env bash
# 交互式问答
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 main.py --chat "$@"
