#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "已创建 .env，请填写模型、Embedding 和所需工具配置后重新运行。"
  exit 0
fi

if ! command -v node >/dev/null 2>&1; then
  echo "未找到 Node.js。请安装 Node 24.21.0 后重试。"
  exit 1
fi

node -e '
const [major, minor] = process.versions.node.split(".").map(Number);
const supported = (major === 20 && minor >= 19) || (major === 22 && minor >= 13) || major >= 24;
if (!supported) {
  console.error(`当前 Node ${process.versions.node} 不受支持；请使用 Node 24.21.0（项目包含 .nvmrc）。`);
  process.exit(1);
}
'

python -m pip install -r requirements.txt
npm_config_cache="$PROJECT_DIR/.npm-cache" npm --prefix web install
mkdir -p data/originals data/converted data/downloads data/rag data/search_cache

if command -v lsof >/dev/null 2>&1; then
  for port in 8000 3000; do
    if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "端口 $port 已被其他进程占用。请先停止已有的 SAGE 服务，再重新运行。"
      lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
      exit 1
    fi
  done
fi

(exec python server.py) &
BACKEND_PID=$!
(cd web && exec node node_modules/next/dist/bin/next dev --webpack) &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "SAGE Scholar: http://localhost:3000（后端 http://localhost:8000）"
wait "$FRONTEND_PID"
