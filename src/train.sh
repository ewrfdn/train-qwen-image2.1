#!/usr/bin/env bash
# 一键训练入口 — Qwen-Image-2.1 人物 LoRA (DGX Spark / GB10)
# 用法: ./src/train.sh config/train_lora_qwen_image_21_spark.yaml
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_TOOLKIT="$PROJECT_ROOT/ai-toolkit"
CONFIG="${1:-$PROJECT_ROOT/config/train_lora_qwen_image_21_spark.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "[FATAL] 配置不存在: $CONFIG" >&2; exit 1
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# 训练需要统一内存；sglang 容器占 5.5GB，提醒但不阻断
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^qwen-sglang-40g$'; then
    echo "[WARN] qwen-sglang-40g 正在运行(约5.5GB内存)，建议先停: docker stop qwen-sglang-40g" >&2
fi

# python3.12-dev 头文件本地解包（无 root），triton JIT 编译需要
_PY_HEADERS="$PROJECT_ROOT/.deps/python312-headers/usr/include"
export CPATH="$_PY_HEADERS/python3.12:$_PY_HEADERS${CPATH:+:$CPATH}"

source "$AI_TOOLKIT/venv/bin/activate"
cd "$AI_TOOLKIT"
python run.py "$CONFIG"
