# 训练进度查询（bocchi_lora_v1）

训练在后台运行，不占终端。需要看进度时用下面的命令。

## 常用查询

```bash
cd /home/sakana/workspace/train-qwen2.1-image

# 1. 训练是否还在跑（有输出 = 在跑）
ps aux | grep "[r]un.py"

# 2. 实时进度（当前步数/loss，Ctrl+C 退出查看，不影响训练）
tail -f logs/train_bocchi_v1.log

# 3. 只看最新几行
tail -c 2000 logs/train_bocchi_v1.log | tr '\r' '\n' | tail -10

# 4. 已保存的 checkpoint
ls -lht output/bocchi_lora_v1/*.safetensors

# 5. 采样图（每 250 步更新，按时间排序看最新的）
ls -lt output/bocchi_lora_v1/samples/ | head

# 6. 大致进度百分比（从日志抓当前步数）
grep -o "[0-9]*/2000" logs/train_bocchi_v1.log | tr '\r' '\n' | tail -1
```

## 时间参考

- 共 2000 步，每步约 2-7 秒（分桶大小不同）+ 每 250 步生成 3 张采样图（约 3.5 分钟）
- 预计总时长 2-3 小时；checkpoint 每 250 步保存一次，最多保留 4 个

## 停止 / 恢复

```bash
# 停止（保存进行中别按，等 checkpoint 写完再停）
kill $(pgrep -f "[r]un.py")

# 从最近 checkpoint 续训（直接重跑同一条命令即可）
./src/train.sh config/train_lora_qwen_image_21_spark.yaml
```

## 产物位置

- LoRA 权重：`output/bocchi_lora_v1/bocchi_lora_v1.safetensors`（训练完成）/ `*_000000250.safetensors` 等（中间步）
- 采样图：`output/bocchi_lora_v1/samples/`（文件名含步数，如 `..._000000250_0.jpg`）
- 训练日志：`logs/train_bocchi_v1.log`

## 训练完成后

```bash
docker start qwen-sglang-40g   # 恢复被停掉的 sglang 容器
```
