# Qwen-Image-2.1 人物 LoRA 训练环境（DGX Spark / GB10）

基于 ostris/ai-toolkit，一条命令对任意合规数据集启动 Qwen-Image-2.1 人物 LoRA 训练。
权重复用本机 ComfyUI 已有文件（只读软链，零拷贝）。

## 目录结构

```
train-qwen2.1-image/
├── ai-toolkit/            # 训练引擎（ostris/ai-toolkit + venv，setup_env.sh 生成）
│   └── models/            # 软链 ComfyUI 权重（三件套，勿删）
├── src/
│   ├── setup_env.sh       # 环境初始化（幂等，可重复运行）
│   ├── train.sh           # 一键训练入口
│   └── validate_dataset.py# 训练前数据集校验
├── config/
│   └── train_lora_qwen_image_21_spark.yaml   # 主配置（改 folder_path 后即可训）
├── requirements-frozen.txt# 依赖快照（pip freeze，参考/校验用）
├── datasets/              # 训练数据，每个 LoRA 一个子目录
├── output/                # checkpoint + 采样图输出
└── .deps/                 # python3.12-dev 头文件（triton JIT 需要，无 root 方案）
```

## 环境初始化（首次 / 重建）

```bash
./src/setup_env.sh
```

脚本幂等，做六件事：clone ai-toolkit（锁定 commit `ecee894`）→ venv + torch 2.13.0+cu130
（ARM64）→ requirements → 软链 ComfyUI 三件套权重（零下载）→ 解包 python3.12 头文件到
`.deps/`（无 root，供 triton JIT）→ 验证 CUDA。

依赖版本定义：
- ai-toolkit commit：见 `src/setup_env.sh` 顶部 `AI_TOOLKIT_COMMIT`
- 完整冻结快照：`requirements-frozen.txt`（重建后可 `pip freeze | diff` 校验一致性）
- 关键版本：torch 2.13.0+cu130 / transformers 5.5.3 / diffusers(git pin) / triton 3.7.1

## 如何启动训练

1. 数据集放到 `datasets/<名字>/`：图片（jpg/png）+ 同名 `.txt` caption；
   **触发词直接写进每条 caption**（`cache_text_embeddings: true` 时配置级 trigger_word 不可靠）
2. 校验：`python3 src/validate_dataset.py datasets/<名字> <触发词>` → 必须 ✓
3. 改 `config/train_lora_qwen_image_21_spark.yaml` 的 `folder_path`（以及 name、采样 prompts 里的触发词）
4. 启动：

```bash
./src/train.sh config/train_lora_qwen_image_21_spark.yaml
```

- checkpoint / 采样图在 `output/<name>/`；Ctrl+C 可中断，重跑自动从最近 checkpoint 续训
  （保存中按 Ctrl+C 可能损坏 checkpoint，等保存完再停）
- `steps: 2000`（30 张图参考值）；1000–1500 步时先看采样图决定是否提前停

## 环境要点

- torch 2.13.0+cu130（ARM64），GB10 CUDA 可用；`quantize: false` + bf16 直训
- 大权重（transformer/text_encoder/vae 共 32GB）从 `ai-toolkit/models/` 软链到
  `/home/sakana/workspace/comfyui/ComfyUI/models/`，本地命中即用，绝不下载
- 首次运行自动从 HF 拉 configs/processor/tokenizer（几 MB，HF 直连正常）
- triton JIT 依赖 `.deps/` 下的 python3.12 头文件（train.sh 已自动注入 CPATH）

## 资源状态

- `qwen-sglang-40g` 容器已停止（原 restart policy: unless-stopped），训练完成后恢复：
  `docker start qwen-sglang-40g`
- ComfyUI 服务全程未受影响（训练前后 active / HTTP 200）

## 故障排查

| 症状 | 处理 |
|---|---|
| 首次运行卡在下载 | configs/processor 从 Qwen 官方 repo 拉；`export HF_ENDPOINT=https://hf-mirror.com` 后重跑 |
| 找不到权重 | `ls ai-toolkit/models/{diffusion_models,text_encoders,vae}/` 对照三个文件名 |
| OOM | 不应发生（128GB 统一内存）；检查 yaml 是否被改成量化/大分辨率 |
| 采样图全黑/坏图 | 软链指向的权重被 ComfyUI 更新删除；重新对齐软链 |
