# VastAI 统一云端训练入口

完整的源码同步、交互恢复、多数据集、训练、Docker 打包和 ROS 交付流程见
`docs/end_to_end_training_delivery_vastai_zh.md`。

推荐从本机解析 VastAI 提供的 SSH 命令，只上传小型 bootstrap；远端再从
Git 拉取主仓库并递归恢复固定 submodule commit：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080'
```

Git 恢复完成后会进入远端 `restore_and_launch.sh` 交互向导。本地工作树
rsync 仅作为显式 `--sync-working-tree` fallback，不是新实例默认方案。

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
| `task2` | `openpi` | Pi0.5 Kuavo Task2 bimanual |
| `task1` | `lingbot-v1` | Task1 LingBot-VLA full pipeline |
| `task2` | `lingbot-v1` | Task2 LingBot-VLA bimanual full pipeline |
| `task1` | `lingbot-v2` | Task1 right-arm LingBot-VLA-v2 |
| `task2` | `lingbot-v2` | Task2 bimanual LingBot-VLA-v2 |
| `task2` | `dp` | `dp_r2_h100.yaml` |
| `task3` | `act` | `act_config.yaml` + `task=task3` |

LingBot-v2 已恢复为 Task2 bimanual 训练入口。正式运行前，把示例复制到
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

旧的非交互 `launch_job.sh` 仍可通过 rsync 同步 unified 工作树；新的推荐
入口 `bootstrap_from_ssh` 默认由远端 Git clone/submodule 恢复源码。
私有环境文件单独保存在远端 `.secrets/` 并设为 `0600`。默认后台启动，
日志位于
`/workspace/kuavo_unified_stack/logs/<task>-<backend>-launcher.log`。设置
`DETACH=0` 可前台运行，`SYNC_ONLY=1` 只同步不启动。

## 数据、预训练权重与 resume

私有 env 文件中的 `DATASET_REPO` 指向所选的第一个 LeRobot 数据集，
`DATASET_MIX_JSON` 保存一个或多个数据集及其权重，
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
训练状态，只有推理用 `model.safetensors` 不足以续训。OpenPI 可以从独立
的 resume 仓库读取完整 run，再上传到本次选择的 `MODEL_REPO`。
交互向导中的 OpenPI 续训不要求手填 `RUN_ID`；本地目录名由 resume 仓库名
生成，W&B 身份从仓库中的 `wandb_id.txt` 自动恢复。旧仓库缺失该文件时，
可输入原 W&B run ID，或留空创建新的 W&B run。恢复后的全局 step
直接进入一段无 warmup 的 LR tail，可选择保持当前 LR 或继续 cosine 衰减。
OpenPI checkpoint 会先进入 `.hf-download` staging 目录；只有完整下载成功
才替换训练目录。中断并重启会继续 staging 下载，避免 Orbax 读取半个
Zstd/OCDBT 分片。
LingBot-v1 的 resume 仓库必须保存完整
`checkpoints/global_step_*` DCP 目录；流水线会在训练前下载到指定 run。

## 模型与权重

| backend | 云端训练入口 | 需要的预训练资产 |
| --- | --- | --- |
| `dp` | unified `train_policy.py` | torchvision ResNet18 ImageNet 权重 |
| `act` | unified `train_policy.py` | torchvision ResNet18 ImageNet 权重 |
| `openpi` | OpenPI 原生 Vast 流水线 | Pi0.5 Orbax 基础参数、PaliGemma tokenizer |
| `lingbot-v1` | `run_lingbot_v1_full_pipeline.sh` | LingBot-VLA-4B、Qwen2.5-VL-3B |
| `lingbot-v2` | unified v2 launcher | LingBot-VLA-v2-6B、Qwen3-VL-4B、MoGe-2；Depth 与 DINO teacher 位于 v2 基础仓库 |

所有 HF 仓库 ID 和目标位置均可通过私有环境文件覆盖。LingBot-v2 的
MoGe、Depth 和 DINO 路径会显式传给上游 trainer，避免继承开发机上的
`/mnt/pqssd` 路径。DP/ACT 不需要 VLA 基础权重；其 ResNet18 权重由对应
训练环境的 torchvision 缓存管理。DP/ACT 在 `GPU_COUNT=1` 时使用普通
训练入口，多卡时自动改用 Accelerate，并保持 batch、梯度累积和最大步数
覆盖参数一致。默认还会用 `requirements_train_cloud.txt` 补齐依赖；
已在镜像中预装并验证依赖时，可设置 `PREPARE_ENV=0`。

OpenPI、DP、ACT 和 LingBot-v1/v2 都支持同构数据集的虚拟加权混合。向导会
逐个选择 HF dataset 并输入比例；远端下载后校验 task 对应 state/action
维度、相机、FPS、robot type 和 LeRobot 版本。DP/ACT 合并 norm stats，
LingBot-v1、LingBot-v2 和 OpenPI 均按训练采样分布重新计算 norm。

LingBot-v2 会自动创建 Python 3.12/PyTorch 2.8 环境，并下载：

- `robbyant/lingbot-vla-v2-6b`（6B 基模、LingBot-Depth、DINO-video）；
- `Qwen/Qwen3-VL-4B-Instruct`（tokenizer/processor）；
- `Ruicheng/moge-2-vitb-normal`（`model.pt`）。

下载后会逐项检查模型 index、Depth、DINO、MoGe 和 Qwen processor 文件，
再按所选 Task2 数据集生成 bimanual norm stats。训练默认开启
`torch.compile`，每 5000 step 保存 checkpoint，只保留最新完整 DCP，并
生成已合并 LoRA 的完整 `hf_ckpt`。

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
