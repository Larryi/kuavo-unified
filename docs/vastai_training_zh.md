# VastAI 统一云端训练入口

统一入口覆盖 `dp`、`act`、`openpi`、`lingbot-v1` 和 `lingbot-v2`：

```bash
MODEL_BACKEND=lingbot-v2 DRY_RUN=1 scripts/vast/launch.sh
```

Dry-run 只显示模型所需的数据、预训练权重及远端入口，不进行 SSH、下载、
训练、上传、通知或关机。正式运行前，把示例复制到仓库外并设置为私有：

```bash
cp scripts/vast/kuavo_vast.env.example /tmp/kuavo-lingbot-v2.env
chmod 600 /tmp/kuavo-lingbot-v2.env
editor /tmp/kuavo-lingbot-v2.env
```

本地启动：

```bash
export MODEL_BACKEND=lingbot-v2
export VAST_SSH_HOST="<Vast SSH host>"
export VAST_SSH_PORT="<Vast SSH port>"
export VAST_ENV_FILE=/tmp/kuavo-lingbot-v2.env
scripts/vast/launch.sh
```

启动器通过 `rsync` 同步 unified 仓库及已经初始化的子模块内容，排除
`.git`、虚拟环境、输出、checkpoint、W&B 日志和压缩镜像；私有环境文件
单独上传到远端 `.secrets/` 并设为 `0600`。默认后台启动，日志位于
`/workspace/kuavo_unified_stack/logs/<backend>-launcher.log`。设置
`DETACH=0` 可前台运行，`SYNC_ONLY=1` 只同步不启动。

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

LingBot-v2 不在运行时临时组合环境。统一入口会要求专用环境严格使用
Python 3.12、PyTorch 2.8.0、Transformers 4.57.3、Accelerate 1.7.0，并
验证 FlashAttention 可导入；不满足时会在下载大模型前失败。对应镜像的
构建与容器启动是下一交付工作包。

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
