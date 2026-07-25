# VastAI 统一云端训练入口

一个 VastAI 实例只运行一个任务数据集和一种算法。面向用户的入口同时
要求 `--task` 与 `--algorithm`：

```bash
scripts/vast/launch_job.sh \
  --task task2 \
  --algorithm dp \
  --dry-run
```

Dry-run 只显示模型所需的数据、预训练权重及远端入口，不进行 SSH、下载、
训练、上传、通知或关机。当前支持矩阵：

| task | algorithm | 训练配置 |
|---|---|---|
| `task1` | `openpi` | Pi0.5 Kuavo Task1 |
| `task1` | `lingbot-v1` | Task1 LingBot-VLA full pipeline |
| `task2` | `dp` | `dp_r2_h100.yaml` |
| `task3` | `act` | `act_config.yaml` + `task=task3` |

LingBot-v2 仍按决定暂缓，入口会在联网前拒绝。正式运行前，把示例复制到
仓库外并设置为私有：

```bash
cp scripts/vast/kuavo_vast.env.example /tmp/kuavo-task2-dp.env
chmod 600 /tmp/kuavo-task2-dp.env
editor /tmp/kuavo-task2-dp.env
```

本地启动：

```bash
scripts/vast/launch_job.sh \
  --task task2 \
  --algorithm dp \
  --env-file /tmp/kuavo-task2-dp.env \
  --host "<Vast SSH host>" \
  --port "<Vast SSH port>"
```

启动器通过 `rsync` 同步 unified 仓库及已经初始化的子模块内容，排除
`.git`、虚拟环境、输出、checkpoint、W&B 日志和压缩镜像；私有环境文件
单独上传到远端 `.secrets/` 并设为 `0600`。默认后台启动，日志位于
`/workspace/kuavo_unified_stack/logs/<task>-<backend>-launcher.log`。设置
`DETACH=0` 可前台运行，`SYNC_ONLY=1` 只同步不启动。

## 数据、预训练权重与 resume

私有 env 文件中的 `DATASET_REPO` 指向这一个任务的 LeRobot 数据集，
`MODEL_REPO` 指向本次训练的输出仓库。远端会自动下载数据集；OpenPI 与
LingBot 会按 profile 继续下载各自基模和 tokenizer，DP/ACT 只准备
torchvision ResNet18 公共权重。

从 Hugging Face 完整训练状态续训：

```bash
scripts/vast/launch_job.sh \
  --task task2 \
  --algorithm dp \
  --env-file /tmp/kuavo-task2-dp.env \
  --host "<host>" --port "<port>" \
  --resume-repo owner/task2-dp-full-state \
  --resume-run-id run_20260711_011552
```

resume 仓库必须包含 optimizer、processor、RNG/accelerator 或 DCP 等完整
训练状态，只有推理用 `model.safetensors` 不足以续训。OpenPI 当前要求
resume 仓库就是 `MODEL_REPO`。LingBot-v1 的 resume 仓库必须保存完整
`checkpoints/global_step_*` DCP 目录；流水线会在训练前下载到指定 run。

## 模型与权重

| backend | 云端训练入口 | 需要的预训练资产 |
| --- | --- | --- |
| `dp` | unified `train_policy.py` | torchvision ResNet18 ImageNet 权重 |
| `act` | unified `train_policy.py` | torchvision ResNet18 ImageNet 权重 |
| `openpi` | OpenPI 原生 Vast 流水线 | Pi0.5 Orbax 基础参数、PaliGemma tokenizer |
| `lingbot-v1` | 既有 Task1 完整流水线 | LingBot-VLA-4B、Qwen2.5-VL-3B |
| `lingbot-v2` | unified v2 launcher | LingBot-VLA-v2-6B、Qwen3-VL-4B、MoGe-2；Depth 与 DINO teacher 位于 v2 基础仓库 |

所有 HF 仓库 ID 和目标位置均可通过私有环境文件覆盖。LingBot-v2 的
MoGe、Depth 和 DINO 路径会显式传给上游 trainer，避免继承开发机上的
`/mnt/pqssd` 路径。DP/ACT 不需要 VLA 基础权重；其 ResNet18 权重由对应
训练环境的 torchvision 缓存管理。DP/ACT 在 `GPU_COUNT=1` 时使用普通
训练入口，多卡时自动改用 Accelerate，并保持 batch、梯度累积和最大步数
覆盖参数一致。默认还会用 `requirements_train_cloud.txt` 补齐依赖；
已在镜像中预装并验证依赖时，可设置 `PREPARE_ENV=0`。

LingBot-v2 云端训练适配暂缓，不进入当前流程。

## 进度跟踪

远端作业持续原子更新：

```text
/workspace/kuavo_runs/<backend>/logs/<run-id>/status.json
```

其中包含 task、backend、run id、当前 phase、上传状态、更新时间和最终
success/failed。查看一次状态、有限日志尾部和 GPU：

```bash
TRAINING_TASK=task2 MODEL_BACKEND=dp \
VAST_SSH_HOST="<host>" VAST_SSH_PORT="<port>" \
scripts/vast/status.sh
```

持续刷新可增加 `WATCH_SECONDS=30`。脚本每次只读取默认 30 行日志，
可用 `TAIL_LINES` 调整，不会拉取完整滚动日志。

## 凭据和生命周期

私有环境文件支持：

- `HF_TOKEN`：读取私有数据集并写入目标模型仓库；
- `WANDB_API_KEY` / `WANDB_PROJECT`；
- `SERVERCHAN_SENDKEY` 或 `SERVERCHAN_URL`；
- `VAST_API_KEY` / `VAST_INSTANCE_ID`。

默认仅在训练和上传成功后自动停止实例。失败或上传不完整时会保留实例，
便于恢复；只有为对应场景显式设置 `AUTO_STOP_ON_FAILURE=1` 或
`AUTO_STOP_ON_UPLOAD_FAILURE=1` 才会在失败后请求停机。停机失败
只会通知并保留实例，不会把凭据打印到日志。

远端镜像或环境仍需提供对应 backend 的 Python/CUDA 依赖。正式租用 GPU
前应先在目标镜像内执行 dry-run；首次真实运行建议保持
`AUTO_STOP_ON_FAILURE=0`，确认 checkpoint 与 HF 上传完整后再提高自动化程度。
