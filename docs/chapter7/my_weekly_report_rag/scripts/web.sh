#!/usr/bin/env bash
# 启动 Web 服务，供局域网用户通过浏览器访问
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 main.py --web "$@"
