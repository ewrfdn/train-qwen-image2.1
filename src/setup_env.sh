#!/usr/bin/env bash
# 环境初始化 — DGX Spark (GB10) 上的 ai-toolkit 训练环境
# 幂等：已存在的组件跳过，可重复运行
# 用法: ./src/setup_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_TOOLKIT="$PROJECT_ROOT/ai-toolkit"

# ---- 锁定版本（依赖定义）----
AI_TOOLKIT_REPO="https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_COMMIT="ecee894ed2b1f3716d9d7326693061ec1a3105bb"   # 2026-09-27
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_PINS="torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0"

# ComfyUI 权重来源（只读复用，不下载 32GB）
COMFY_MODELS="/home/sakana/workspace/comfyui/ComfyUI/models"
declare -A COMFY_WEIGHTS=(
  [diffusion_models]="qwen_image_2.1_bf16.safetensors"
  [text_encoders]="qwen3vl_8b_bf16.safetensors"
  [vae]="qwen_image_2.1_vae_bf16.safetensors"
)

step() { echo -e "\n=== [setup_env] $1 ==="; }

# ---- 1. clone ai-toolkit（锁 commit）----
step "1/6 ai-toolkit 源码 ($AI_TOOLKIT_COMMIT)"
if [ ! -d "$AI_TOOLKIT/.git" ]; then
    git clone "$AI_TOOLKIT_REPO" "$AI_TOOLKIT"
    git -C "$AI_TOOLKIT" checkout "$AI_TOOLKIT_COMMIT"
elif [ "$(git -C "$AI_TOOLKIT" rev-parse HEAD)" != "$AI_TOOLKIT_COMMIT" ]; then
    echo "[WARN] ai-toolkit 当前 commit 与锁定不一致："
    echo "  锁定: $AI_TOOLKIT_COMMIT"
    echo "  实际: $(git -C "$AI_TOOLKIT" rev-parse HEAD)"
    echo "  如需对齐: git -C $AI_TOOLKIT checkout $AI_TOOLKIT_COMMIT"
fi

# ---- 2. venv + torch(cu130/ARM64) + requirements ----
step "2/6 Python venv 与 PyTorch"
if [ ! -x "$AI_TOOLKIT/venv/bin/python" ]; then
    python3 -m venv "$AI_TOOLKIT/venv"
fi
source "$AI_TOOLKIT/venv/bin/activate"

if python -c "import torch" 2>/dev/null; then
    echo "torch 已安装: $(python -c 'import torch; print(torch.__version__)')"
else
    pip install --no-cache-dir $TORCH_PINS --index-url "$TORCH_INDEX"
fi

if [ -f "$AI_TOOLKIT/requirements.txt" ]; then
    # diffusers 在 requirements 中是 git pin，pip 自动处理；已装则跳过
# pip install 幂等：依赖已满足时为 no-op（几秒）
pip install --no-cache-dir -r "$AI_TOOLKIT/requirements.txt"
echo "requirements 已满足"
fi

# ---- 3. 权重软链（复用 ComfyUI，只读）----
step "3/6 权重软链（复用 ComfyUI 三件套）"
for d in diffusion_models text_encoders vae; do
    mkdir -p "$AI_TOOLKIT/models/$d"
    f="${COMFY_WEIGHTS[$d]}"
    src="$COMFY_MODELS/$d/$f"
    dst="$AI_TOOLKIT/models/$d/$f"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "  已存在: models/$d/$f"
    elif [ -f "$src" ]; then
        ln -s "$src" "$dst" && echo "  链接: models/$d/$f -> $src"
    else
        echo "  [FATAL] ComfyUI 权重缺失: $src" >&2
        echo "  手动下载兜底:" >&2
        echo "    huggingface-cli download Comfy-Org/Qwen-Image-2.1 $d/$f --local-dir $AI_TOOLKIT/models" >&2
        exit 1
    fi
done

# ---- 4. python3.12-dev 头文件（triton JIT 需要，无 root 方案）----
step "4/6 python3.12 头文件 (.deps/)"
HEADERS_DIR="$PROJECT_ROOT/.deps/python312-headers"
if [ -f "$HEADERS_DIR/usr/include/python3.12/Python.h" ]; then
    echo "  已存在: $HEADERS_DIR"
else
    mkdir -p "$HEADERS_DIR" /tmp/opencode-setup-debs
    cd /tmp/opencode-setup-debs
    apt-get download libpython3.12-dev python3.12-dev
    for deb in libpython3.12-dev_*.deb python3.12-dev_*.deb; do
        dpkg-deb -x "$deb" "$HEADERS_DIR"
    done
    cd "$PROJECT_ROOT"
    test -f "$HEADERS_DIR/usr/include/python3.12/Python.h" || { echo "[FATAL] 头文件解包失败" >&2; exit 1; }
    echo "  头文件就位（train.sh 通过 CPATH 注入）"
fi

# ---- 5. 目录骨架 ----
step "5/6 目录骨架"
mkdir -p "$PROJECT_ROOT"/{datasets,output,logs}

# ---- 6. 验证 ----
step "6/6 验证"
python - <<'EOF'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "N/A"
print(f"torch {torch.__version__} | cuda={ok} | device={name}")
assert ok, "CUDA 不可用"
EOF
echo -e "\n[setup_env] 环境就绪。启动训练: ./src/train.sh config/train_lora_qwen_image_21_spark.yaml"
