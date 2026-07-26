# Kuavo 五类算法训练、测试与推理镜像交付总指南

本文是 ACT、Diffusion Policy、OpenPI、LingBot-VLA v1/v2 的统一操作入口。
完整生命周期为：

```text
训练/续训
  → 导出部署 checkpoint
  → 无 ROS open-loop
  → Base 镜像只读挂载测试
  → 生成带权重 release 镜像
  → 可选 docker save / Registry push
  → 官方 ROS 环境人工检查并启动
```

数据集已经是 LeRobot 格式，本文不再包含 rosbag 转换。

## 1. Base、release 与 Docker 层

### 1.1 Base 镜像

Base 镜像只包含：

- Ubuntu 20.04、ROS Noetic、KuavoBaseEnv 与部署源码；
- 对应算法的 Python/CUDA/JAX/FlashAttention 环境；
- ACT 所需的 ResNet18 公共预训练权重；
- Open-loop Viewer；
- OpenPI/LingBot-v2 的隔离 Policy Server/Client 运行能力。

Base 镜像不包含任务 checkpoint、数据集、任务 norm stats、凭据。DP 与 ACT
共用 `kuavo-classic:latest`，无需复制两个相同的大环境。

| backend | Base 镜像 |
|---|---|
| `dp`、`act` | `kuavo-classic:latest` |
| `openpi` | `kuavo-openpi:latest` |
| `lingbot-v1` | `kuavo-lingbot-v1:latest` |
| `lingbot-v2` | `kuavo-lingbot-v2:latest` |

### 1.2 release 镜像如何叠加权重

`Dockerfile.release` 使用 `FROM <Base>`，再把资产复制到新的只读层：

```text
/models/checkpoint/                      部署 checkpoint
/assets/qwen/                            Qwen config/tokenizer/processor
/assets/norm_stats/norm_stats.json       任务 norm
/assets/tokenizer/tokenizer.model        OpenPI PaliGemma tokenizer
/assets/runtime/                         其他显式运行资产
/root/.../configs/deploy/kuavo_env.yaml  已渲染部署 YAML
/etc/kuavo/release_manifest.json         构建清单
```

这不是再次复制一份 Base 文件。Docker 使用 content-addressed layer：

- 本机已存在 Base 时，实际新增磁盘主要是 checkpoint 和少量资产层；
- 多个 release 镜像共享相同 Base 层，各自只增加自己的权重/配置层；
- `docker image ls` 显示的是逻辑累计大小，即 Base + release 层；
- `docker system df -v` 才能区分 shared size 与 unique size；
- `docker history <image>` 可以查看各层逻辑大小。

注意，release 不一定只有一个新层：checkpoint、Qwen、norm、tokenizer、
YAML 和 manifest 是多个层，但通常 checkpoint 层占绝大部分。

### 1.3 TAR 和 Registry 不同

- 本机 Docker 存储：Base 层共享，release 的实际增量主要是权重。
- `docker save` 单个 release：为了能在空机器恢复，会包含其引用的 Base
  层和 release 层，TAR 通常接近“Base + 权重”，不是仅权重增量。
- 同一个 `docker save` 命令同时导出多个同源镜像时，共享层在 TAR 中只存
  一份。
- Registry push/pull：按 layer digest 去重。Registry 或目标机器已经有
  相同 Base blob 时，只上传/下载缺失层；否则首次仍需传输完整 Base。

因此日常测试不要生成 TAR。只有正式离线交付才使用 `--save-tar`。

## 2. 已支持任务矩阵与边界

当前支持下面 8 条路由。三种 VLA 均覆盖 Task1/Task2；DP/ACT 仍按当前
交付任务固定：

| task | algorithm | 全新微调 | 完整状态续训 | 同构数据集配比 | 动作/机器人 |
|---|---|---:|---:|---:|---|
| Task1 | OpenPI | 支持 | 支持 | 支持 | 右臂 7D + 单夹爪，共 8D |
| Task2 | OpenPI | 支持 | 支持 | 支持 | 双臂 14D + 双夹爪，共 16D，三相机 |
| Task1 | LingBot-v1 | 支持 | 支持 | 支持 | 右臂 7D + 单夹爪，共 8D |
| Task2 | LingBot-v1 | 支持 | 支持 | 支持 | 双臂 14D + 双夹爪，共 16D，三相机 |
| Task1 | LingBot-v2 | 支持 | 支持 | 支持 | 右臂 7D + 单夹爪，共 8D |
| Task2 | LingBot-v2 | 支持 | 支持 | 支持 | 双臂 14D + 双夹爪，共 16D，三相机 |
| Task2 | DP | 支持 | 支持 | 支持 | 双臂 14D + 双夹爪，共 16D |
| Task3 | ACT | 支持 | 支持 | 支持 | 右臂 + Qiangnao 末端 |

其中 Task2+OpenPI 复用 fork 中已有并验证过结构的 Task2 配置；新接入的
Task2+LingBot-v1 与 Task1+LingBot-v2 已完成配置解析、路由 dry-run 和
部署静态回归，但仍需要各自首次真实 VastAI 训练、open-loop 和 ROS mock
作为最终验收。

这里的“全新微调”是建立一个新的任务训练 run。OpenPI 和 LingBot 系列仍会
加载各自基础模型；它不是从随机参数预训练 VLA 基模。DP/ACT 按各自配置
初始化策略，ACT 仍使用 ResNet18 ImageNet 公共预训练权重。

“任意任务”仍不成立。例如 Task1+ACT、Task1+DP、Task3+任一 VLA 尚无
经过审核的 VastAI profile。要增加组合，至少需要补齐并验证：

- 任务的 state/action 维度、手臂和末端映射；
- 相机键、FPS、LeRobot feature schema 与任务文本；
- 算法 data adapter、robot config 和匹配的 norm 计算；
- 基模/processor 下载清单、训练配置、resume 目录约定；
- open-loop 与 ROS 部署 YAML。

不能只在命令行交换 `--task` 和 `--algorithm`。`job_profile.sh` 与远端向导
会主动拒绝未列入上表的组合，避免训练出维度错误但表面可运行的模型。

多数据集配比也不是把不同任务任意混合。一次 run 中的所有源必须属于同一
任务路由，并具有相同机器人 embodiment、state/action/camera schema、
FPS 和 LeRobot 版本；向导会归一化权重，远端下载后再次校验。

当前数据集：

```text
Task1 /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345
Task2 /mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264
Task3 /mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165
```

## 3. 构建或检查 Base 镜像

先构建 Classic，再构建依赖它的 OpenPI/LingBot-v2：

```bash
scripts/kuavo_base_image \
  --backend dp \
  --build \
  --env-archive /mnt/pqssd/docker_envs/classic/myenv.tar.gz

# ACT 与 DP 共用同一个镜像；这里只检查，不重复构建。
scripts/kuavo_base_image --backend act

scripts/kuavo_base_image \
  --backend lingbot-v1 \
  --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz

scripts/kuavo_base_image \
  --backend lingbot-v2 \
  --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v2/myenv.tar.gz

scripts/kuavo_base_image --backend openpi --build
```

OpenPI 的 `uv.lock` 版本和哈希保持不变，但镜像内副本会把 PyPI wheel URL
映射到 BFSU。本地 Docker build 使用 BFSU；VastAI 远端恢复默认使用官方
PyPI。构建日志保存在 `/tmp/<image>.build.log`，失败时只读末尾：

```bash
tail -n 80 /tmp/kuavo-openpi.build.log
```

不带 `--build` 只检查镜像是否存在。不要为每个 checkpoint 重建 Base。

## 4. 训练

### 4.1 DP Task2

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=dp_r2_h100.yaml \
  task=task2 \
  method=dp \
  root=/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264 \
  repoid=task2_repaired_264 \
  training.device=cuda
```

已有部署候选：

```text
/home/larry/alter/r2/dp/run_20260711_011552/epochbest
```

### 4.2 ACT Task3

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=act_config.yaml \
  task=task3 \
  method=act \
  root=/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165 \
  repoid=task3_repaired_165 \
  training.device=cuda
```

已有部署候选：

```text
/home/larry/alter/r3/act/run_20260629_233939/epochbest
```

DP/ACT 每个 epoch 覆盖 latest 模型和完整 resume 状态，验证更优时更新
`epochbest`。默认不保留大量重复 `epochN`。

续训必须传包含 `learning_state.pth`、`rng_state.pth`、模型与 processors
的运行目录，而不是仅 `epochbest`：

```bash
python kuavo_train/train_policy.py ... \
  training.resume=true \
  training.resume_timestamp=run_YYYYMMDD_HHMMSS
```

### 4.3 OpenPI Task1/Task2

更换数据分布时先重算 norm：

```bash
scripts/kuavo_openpi norm-stats pi05_kuavo \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345 \
  --tokenizer-path /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model
```

训练：

```bash
scripts/kuavo_openpi train pi05_kuavo \
  --exp-name=task1_full \
  --num-train-steps=30000
```

Task2 使用已经配置的双臂 16D/三相机 profile：

```bash
scripts/kuavo_openpi norm-stats pi05_kuavo_task2 \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264 \
  --tokenizer-path /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model

scripts/kuavo_openpi train pi05_kuavo_task2 \
  --exp-name=task2_full \
  --num-train-steps=30000
```

部署 checkpoint 必须是 step 目录：

```text
45000/
  params/
    _METADATA
```

不要只传 `45000/params`。完整 resume 还必须含优化器等 Orbax 训练状态。

### 4.4 LingBot-v1 Task1/Task2

本地先 dry-run：

```bash
HF_DATASETS_CACHE=/tmp/kuavo-lingbot-hf \
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=lingbot_config.yaml \
  policy.dry_run=true
```

完整云端流水线使用：

```bash
TRAINING_TASK=task1 scripts/run_lingbot_v1_full_pipeline.sh

TRAINING_TASK=task2 scripts/run_lingbot_v1_full_pipeline.sh
```

Task2 使用：

```text
configs/policy/lingbot/task2_264_full.yaml
configs/robot_configs/kuavo_v1_bimanual.yaml
```

训练结束必须导出完整 HF checkpoint，并保留：

```text
config.json
model*.safetensors
lingbotvla_cli.yaml
norm_stats.json
```

当前旧权重 `/mnt/pqssd/hf_train_outputs/task1_lingbot` 使用绝对关节动作，
部署时必须选择 `kuavo_v1_right_arm_absolute`。

### 4.5 LingBot-v2 Task1/Task2

Task1 使用：

```text
configs/policy/lingbot_v2/kuavo_lora.yaml
configs/policy/lingbot_v2/kuavo_norm.yaml
configs/robot_configs/kuavo_v2_right_arm.yaml
```

Task2 使用：

```text
configs/policy/lingbot_v2/kuavo_lora_task2_bimanual.yaml
configs/robot_configs/kuavo_v2_bimanual.yaml
assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json
```

本地 dry-run：

```bash
LINGBOT_V2_NORM_STATS=assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json \
python kuavo_train/train_lingbot_v2.py \
  task=task2 \
  root=/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264 \
  repoid=task2_repaired_264 \
  policy.config_path=configs/policy/lingbot_v2/kuavo_lora_task2_bimanual.yaml \
  policy.model_path=/mnt/pqssd/pretrained/lingbot-vla-v2-6b \
  policy.tokenizer_path=/mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct \
  training.output_directory=outputs/train/task2/lingbot_v2_lora \
  policy.dry_run=true
```

确认命令、模型和数据路径后，把最后的 `policy.dry_run=true` 改成
`policy.dry_run=false` 或删除该覆盖开始训练。

训练默认开启 `torch.compile`。训练环境需要 LingBot-v2-6B、Qwen3-VL、
MoGe、LingBot-Depth、DINO teacher；部署只需要合并后的完整 `hf_ckpt`、
Qwen processor/tokenizer、robot config 与 norm stats。

当前已转换部署权重：

```text
/mnt/pqssd/training_outputs/lingbot_v2_task2_bimanual/
  run_20260709_134023/checkpoints/global_step_15000/hf_ckpt
```

## 5. 多数据集混合与 VastAI

### 5.1 VastAI 实例和本地仓库前提

推荐让新实例从 Git 恢复源码，而不是 rsync 整个本地工作树。开始前确认：

```bash
git status --short
git rev-list --left-right --count personal/codex/unified-stack...HEAD
git submodule status --recursive
```

第一条应无输出，第二条应为 `0 0`，submodule 行首不应出现 `-`、`+` 或
`U`。VastAI 基础实例至少需要 NVIDIA 驱动、`git`、`python3`、SSH 服务和
足够的数据集/模型/训练状态磁盘空间。算法环境由远端流水线恢复；远端
Python 包默认使用官方 `https://pypi.org/simple`，不使用 BFSU。

VastAI 页面通常提供类似命令：

```text
ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080
```

本地先做不联网、不上传的命令预览：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080' \
  --repo-url https://github.com/Larryi/kuavo-unified.git \
  --git-ref codex/unified-stack \
  --dry-run
```

确认后去掉 `--dry-run`：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080' \
  --repo-url https://github.com/Larryi/kuavo-unified.git \
  --git-ref codex/unified-stack
```

脚本只 rsync 一个小型 `remote_clone_and_restore.sh` 到 `/tmp`，然后远端：

1. clone `codex/unified-stack` 到 `/workspace/kuavo_unified_stack`；
2. 按主仓库固定 SHA 执行递归 submodule 初始化；
3. 写出 `.vast_git_manifest.json`，记录主仓库和全部 submodule commit；
4. 进入 `scripts/vast/restore_and_launch.sh` 交互向导。

本地入口会先验证当前 HEAD 已推送，并验证每个 submodule SHA 能从配置的
fork 获取。正常新实例不要传 `--sync-working-tree`；该参数只用于远端
无法访问 Git 或明确需要测试尚未提交改动的应急场景。

如果只想恢复代码、暂不配置训练：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4' \
  --repo-url https://github.com/Larryi/kuavo-unified.git \
  --git-ref codex/unified-stack \
  --no-launch
```

之后登录远端手工进入向导：

```bash
cd /workspace/kuavo_unified_stack
bash scripts/vast/restore_and_launch.sh
```

### 5.2 交互向导：新训练、多数据集与凭据

向导依次完成：

1. 选择算法；OpenPI/LingBot-v1/v2 再选择 Task1 或 Task2，DP/ACT 自动
   绑定现有任务；
2. 输入 HF token，终端每个字符回显为 `*`，并通过 `whoami` 验证；
3. 从 HF 用户/组织仓库选择一个或多个同构 LeRobot dataset；
4. 逐个添加数据集并输入正比例，例如 `25,75`；
5. 选择或输入本次训练输出的 HF model repo；
6. 选择是否从完整训练状态 resume；
7. 输入可选的 W&B、ServerChan、VastAI 凭据，秘密同样以 `*` 回显；
   同一实例和工作目录再次运行时可复用已保存的 `0600` 凭据；
8. 选择 GPU ID，例如单卡 `0` 或多卡 `0,1`；
9. 审核不含凭据的任务清单，再确认环境恢复和训练启动。

OpenPI 会明确询问总训练步数、warmup 步数、峰值学习率、cosine decay
步数和最终学习率。当前默认值依次为
`30000 / 1000 / 2.5e-5 / 30000 / 2.5e-6`。续训时总步数按
“已完成 step + 用户输入的追加步数”计算。GPU 预检默认不限定 A100 或
任何具体型号，仍检查 CUDA、GPU 数量和最低显存；仅在用户显式设置
`REQUIRE_GPU_NAME` 时才限制型号。

OpenPI norm stats 默认使用 `NORM_NUM_WORKERS=0`，避免 TorchCodec/FFmpeg
等原生解码器在 DataLoader 多进程 worker 中段错误；正式训练仍使用
`NUM_WORKERS` 并行加载。若手动将 norm workers 调大，失败时流水线会自动
用单进程重试。

数据配比保存在 `DATASET_MIX_JSON`，例如：

```json
[
  {"name":"source_01_task1_sz","repo_id":"owner/task1_sz","weight":0.25},
  {"name":"source_02_task1_bj","repo_id":"owner/task1_bj","weight":0.75}
]
```

远端逐个下载数据集、验证 schema，再生成带本地路径的 resolved manifest。
八条已支持路由都使用该加权采样分布，并按相同分布计算或合并 norm。

HF token 至少要有：读取所选私有 dataset、读取所需基模/resume 仓库，以及
创建或写入输出 model repo 的权限。私密配置写入：

```text
/workspace/kuavo_unified_stack/.secrets/interactive-job.env  # mode 0600
/workspace/kuavo_unified_stack/.secrets/vast-credentials.json # 可复用，mode 0600
```

凭据只在当前 VastAI 实例/持久卷中保存；实例销毁且未挂载持久卷后不会保留。

无凭据审核清单写入：

```text
/workspace/kuavo_unified_stack/logs/interactive-job.json
```

### 5.3 从头开始一个新 run

在“是否从 HF 完整训练状态继续训练？”处选择 `n`。随后输入新的 `RUN_ID`
或留空自动生成。流水线会按算法自动准备：

- DP/ACT：训练环境及 ResNet18 公共权重；
- OpenPI：Pi0.5 base params 与 PaliGemma tokenizer；
- LingBot-v1：LingBot-VLA 基模与 Qwen2.5-VL processor；
- LingBot-v2：6B 基模、Qwen3-VL、MoGe、Depth 与 DINO teacher。

这会创建新的任务微调 run；VLA 算法不会从随机权重训练基础模型。

### 5.4 从完整状态接续训练

在 resume 问题处选择 `y`，再选择 HF model repo 并输入原始 `RUN_ID`。
仓库必须包含对应 trainer 的完整状态：

| algorithm | 至少需要的状态标记 |
|---|---|
| OpenPI | step 目录中的 `params/_METADATA` 及优化器等 Orbax 状态 |
| LingBot-v1 | `checkpoints/global_step_*` 完整 DCP |
| LingBot-v2 | `checkpoints/global_step_*` 完整 DCP/训练状态 |
| DP/ACT | `learning_state.pth`、RNG、processor；Accelerate 还需 `epochlatest/` 和 `training_latest_state.pth` |

只有部署用 `hf_ckpt`、`model.safetensors`、`epochbest` 或 OpenPI 的
`params` 子目录不足以无损续训。向导先检查 HF 文件清单，再把完整 run
恢复到训练器预期目录；续训仍可把结果上传到另一个 `MODEL_REPO`。

### 5.5 启动确认、状态、通知与关机

向导展示 `interactive-job.json` 后询问：

```text
确认恢复算法环境并启动训练？ [y/N]
```

选择 `n` 会保留配置但不训练。之后可在远端启动：

```bash
cd /workspace/kuavo_unified_stack
set -a
source .secrets/interactive-job.env
set +a
bash scripts/vast/run_backend.sh
```

作业状态持续原子写入：

```text
/workspace/kuavo_runs/<backend>/logs/<run-id>/status.json
```

从本地查看一次有限日志尾部和 GPU 状态：

```bash
TRAINING_TASK=task2 MODEL_BACKEND=lingbot-v2 \
VAST_SSH_HOST=1.2.3.4 VAST_SSH_PORT=12345 \
scripts/vast/status.sh
```

持续查看可加 `WATCH_SECONDS=30`；脚本默认只读取末尾 30 行，不下载完整滚动
日志。ServerChan 留空即禁用；配置后成功和失败都会通知。只有同时提供
`VAST_INSTANCE_ID`、`VAST_API_KEY`，并确认自动关机时，成功上传后才停止
实例。失败默认不关机，保留现场和 checkpoint 供排查。

`scripts/vast/launch_job.sh` 是已有私有 env 文件时的非交互/自动化入口，
会同步工作树，正常新实例优先使用上述 Git bootstrap。示例：

```bash
scripts/vast/launch_job.sh \
  --task task2 \
  --algorithm lingbot-v2 \
  --env-file /secure/task2-lingbot-v2.env \
  --host 1.2.3.4 \
  --port 12345
```

## 6. 无 ROS open-loop

命令行统一入口：

```bash
python tools/open_loop_eval.py --help
```

OpenPI 也可使用：

```bash
scripts/kuavo_openpi open-loop \
  --config-name pi05_kuavo \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_pi05_real3.0_45000/45000 \
  --episodes 0 \
  --max-samples 3 \
  --require-cuda
```

LingBot-v2：

```bash
python tools/open_loop_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264 \
  --repo-id task2_repaired_264 \
  --policy-type lingbot_v2 \
  --policy-path /mnt/pqssd/training_outputs/lingbot_v2_task2_bimanual/run_20260709_134023/checkpoints/global_step_15000/hf_ckpt \
  --lingbot-root third_party/lingbot-vla-v2 \
  --qwen25-path /mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct \
  --robot-name kuavo_v2_bimanual \
  --norm-stats-file assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json \
  --device cuda
```

Open-loop 只检查数据集观测下的动作、误差、范围和延迟，不替代 ROS/真机。

## 7. Base 镜像只读挂载测试

测试阶段不复制权重进镜像：

```bash
scripts/kuavo_docker shell \
  --backend dp \
  --task task2-dp \
  --checkpoint /home/larry/alter/r2/dp/run_20260711_011552 \
  --checkpoint-subpath epochbest

scripts/kuavo_docker shell \
  --backend act \
  --task task3-act \
  --checkpoint /home/larry/alter/r3/act/run_20260629_233939 \
  --checkpoint-subpath epochbest

scripts/kuavo_docker shell \
  --backend openpi \
  --task task1-openpi \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_pi05_real3.0_45000/45000 \
  --tokenizer /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model

scripts/kuavo_docker shell \
  --backend lingbot-v1 \
  --task task1-lingbot-v1 \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_lingbot \
  --qwen /home/larry/Qwen2.5_VL \
  --norm-stats /mnt/pqssd/hf_train_outputs/task1_lingbot/norm_stats.json

scripts/kuavo_docker shell \
  --backend lingbot-v2 \
  --task task2-lingbot-v2 \
  --checkpoint /mnt/pqssd/training_outputs/lingbot_v2_task2_bimanual/run_20260709_134023/checkpoints/global_step_15000/hf_ckpt \
  --qwen /mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct \
  --norm-stats assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json
```

同一套 Base 镜像也支持新增 VLA 路由；训练完成后只需替换对应 checkpoint、
Qwen processor 和 norm：

```text
task2-openpi
task2-lingbot-v1
task1-lingbot-v2
```

CLI 会显示本次渲染的 YAML，操作者确认后进入容器。检查 ROS master、
topics、相机、动作维度和末端配置，然后统一运行：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

OpenPI 与 LingBot-v2 会由该命令自动启动、等待和清理容器内 Policy Server；
无需先打开第二个 shell。DP、ACT、LingBot-v1 在当前 ROS Python 进程加载。

## 8. 容器 Open-loop Viewer

### 8.1 Viewer 如何进入镜像

Viewer 源码是 `tools/open_loop_viewer.py`。Classic、LingBot-v1 和
LingBot-v2 Dockerfile 都安装并校验 Streamlit；OpenPI 继承 Classic，再
加入隔离 JAX Server。release 镜像继承对应 Base，因此也自动继承 Viewer，
不需要把 Viewer 再复制进权重层。

修改 Viewer 或其依赖后必须重新构建对应 Base，旧镜像不会自动获得源码：

```bash
# DP/ACT Viewer
scripts/kuavo_base_image --backend dp --build \
  --env-archive /mnt/pqssd/docker_envs/classic/myenv.tar.gz

# OpenPI 必须在新 Classic 之后构建
scripts/kuavo_base_image --backend openpi --build

scripts/kuavo_base_image --backend lingbot-v1 --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz

scripts/kuavo_base_image --backend lingbot-v2 --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v2/myenv.tar.gz
```

Open-loop 从 LeRobot dataset 读取历史观测和 GT action，不启动 ROS、不发布
真机动作。模型和数据集均以只读 volume 挂载，退出 Viewer 不修改 Base。

### 8.2 五种算法的启动命令

DP Task2：

```bash
scripts/kuavo_docker viewer \
  --backend dp \
  --checkpoint /home/larry/alter/r2/dp/run_20260711_011552 \
  --checkpoint-subpath epochbest \
  --dataset /mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264
```

ACT Task3：

```bash
scripts/kuavo_docker viewer \
  --backend act \
  --checkpoint /home/larry/alter/r3/act/run_20260629_233939 \
  --checkpoint-subpath epochbest \
  --dataset /mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165
```

OpenPI Task1：

```bash
scripts/kuavo_docker viewer \
  --backend openpi \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_pi05_real3.0_45000/45000 \
  --tokenizer /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model \
  --dataset /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345
```

OpenPI 命令会在同一个容器中先启动 JAX Policy Server，确认
`127.0.0.1:8000` 可用后再启动 Streamlit；不需要第二个终端。

LingBot-v1 Task1：

```bash
scripts/kuavo_docker viewer \
  --backend lingbot-v1 \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_lingbot \
  --qwen /home/larry/Qwen2.5_VL \
  --norm-stats /mnt/pqssd/hf_train_outputs/task1_lingbot/norm_stats.json \
  --dataset /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345
```

LingBot-v2 Task2：

```bash
scripts/kuavo_docker viewer \
  --backend lingbot-v2 \
  --checkpoint /mnt/pqssd/training_outputs/lingbot_v2_task2_bimanual/run_20260709_134023/checkpoints/global_step_15000/hf_ckpt \
  --qwen /mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct \
  --norm-stats assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json \
  --dataset /mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264 \
  --viewer-port 8502
```

浏览器打开 CLI 打印的地址，默认是：

```text
http://127.0.0.1:8501
http://127.0.0.1:8502
```

需要同时开多个 Viewer 时，用不同 `--viewer-port` 和
`--container-name`。LingBot-v1/v2 必须使用各自容器，不能在同一个
Python 进程轮流加载同名 `deploy` 包。

### 8.3 页面字段

CLI 自动预填以下容器路径：

```text
Dataset root       /data/dataset
Policy path        /models/checkpoint[/checkpoint-subpath]
Qwen path          /assets/qwen
Norm stats         /assets/norm_stats/norm_stats.json
OpenPI endpoint    127.0.0.1:8000
```

CLI 还会按任务 profile 预填 Repo ID、训练 task 文本和 state/action 维度：

| 算法 | Repo ID | Policy type | 额外检查 |
|---|---|---|---|
| DP Task2 | `task2_repaired_264` | `diffusion` | action 16D |
| ACT Task3 | `task3_repaired_165` | `act` | Qiangnao/action schema |
| OpenPI Task1 | `kuavo/task1_sz` | `openpi` | endpoint `127.0.0.1:8000` |
| LingBot-v1 | `task1_repaired_345` | `lingbot` | `kuavo_v1_right_arm_absolute` |
| LingBot-v2 | `task2_repaired_264` | `lingbot_v2` | preset `Task2 bimanual`、`kuavo_v2_bimanual` |

OpenPI 页面额外显示 state/action dimension。Task1 默认 8/8，Task2 默认
16/16；即使原生 OpenPI Server metadata 没有声明 `action_dim`，Viewer
也会使用该显式 profile，并对首个响应形状进行校验。

其他推荐初始值：

```text
Device          cuda
Video backend   pyav
Episode         0
Predict mode    chunk
Brightness      1.0
Contrast        1.0
Color           1.0
Center crop     1.0
```

`Task` 必须填写训练时使用的自然语言指令。OpenPI/LingBot 对 prompt
敏感，不应随意使用默认文本评估其他任务。

### 8.4 操作顺序

1. 检查 dataset、checkpoint、算法、robot config、Norm 和 Task prompt。
2. 点击 `Load / run inference`，等待模型和 dataset 加载。
3. 选择 `Frame in selected episode` 查看某一观测。
4. 使用 `Compare horizon` 检查该帧开始的预测 action chunk。
5. 在 `Action dimension` 中逐维比较 GT、Pred、Raw Pred/Postprocessed。
6. 设置 `Timeline frames` 和 `Chunk inference stride`。
7. 点击 `Run episode timeline` 运行所选 episode 的时间线评测。

第一次应关闭所有 causal postprocessing：

```text
Dataset-derived joint rate limit  off
Blend new chunk prefix            off
Latch gripper intent              off
```

这样看到的是原始模型结果。确认原始结果后，再单独开启 rate limit、chunk
blend 或 gripper latch，比较部署后处理带来的变化。

### 8.5 结果怎么看

- `First MAE`：当前帧第一步 action 的平均绝对误差。
- `Horizon MAE`：action chunk 每个未来步的误差。
- `Dimension MAE`：每个关节/夹爪维度的误差。
- `Pred vs GT by Action Dimension`：指定动作维的完整预测曲线。
- `Covered`：时间线中有模型预测覆盖的比例。
- `Timeline MAE/RMSE`：被覆盖时间线上的整体误差。
- `Raw MAE`：启用后处理时，处理前预测误差。
- `Mean/P95 inference`：平均与 P95 推理延迟。

对结果至少检查：

- 图像方向、颜色、裁剪和相机对应关系；
- state/action 维度以及左右臂顺序；
- 夹爪维度是否为第 7/15 维；
- chunk 边界是否跳变；
- action 是否出现 NaN/Inf 或超出训练数据范围；
- OpenPI/LingBot 的 Task prompt、robot config 与 Norm 是否匹配。

Viewer 指标是离线诊断，不等价于任务成功率。曲线合理后仍需执行 Base
镜像 ROS mock，最终再在官方部署环境做真机安全验收。

## 9. 生成带权重 release 镜像

默认只构建本地 Docker image，不生成 TAR。

DP Task2：

```bash
scripts/package_inference_image \
  --task task2 \
  --algorithm dp \
  --checkpoint /home/larry/alter/r2/dp/run_20260711_011552 \
  --checkpoint-subpath epochbest \
  --tag kuavo-task2-dp:epochbest
```

ACT Task3：

```bash
scripts/package_inference_image \
  --task task3 \
  --algorithm act \
  --checkpoint /home/larry/alter/r3/act/run_20260629_233939 \
  --checkpoint-subpath epochbest \
  --tag kuavo-task3-act:epochbest
```

OpenPI Task1：

```bash
scripts/package_inference_image \
  --task task1 \
  --algorithm openpi \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_pi05_real3.0_45000/45000 \
  --tokenizer /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model \
  --tag kuavo-task1-openpi:45000
```

LingBot-v1 Task1：

```bash
scripts/package_inference_image \
  --task task1 \
  --algorithm lingbot-v1 \
  --checkpoint /mnt/pqssd/hf_train_outputs/task1_lingbot \
  --qwen /home/larry/Qwen2.5_VL \
  --norm-stats /mnt/pqssd/hf_train_outputs/task1_lingbot/norm_stats.json \
  --tag kuavo-task1-lingbot-v1:current
```

LingBot-v2 Task2：

```bash
scripts/package_inference_image \
  --task task2 \
  --algorithm lingbot-v2 \
  --checkpoint /mnt/pqssd/training_outputs/lingbot_v2_task2_bimanual/run_20260709_134023/checkpoints/global_step_15000/hf_ckpt \
  --qwen /mnt/pqssd/pretrained/Qwen3-VL-4B-Instruct \
  --norm-stats assets/norm_stats/kuavo_v2_bimanual_task2_meanstd.json \
  --tag kuavo-task2-lingbot-v2:step15000
```

`package_inference_image` 同时接受 `task2+openpi`、
`task2+lingbot-v1` 和 `task1+lingbot-v2`；参数结构与上面的同算法示例
相同，checkpoint/norm 必须来自对应任务，不能跨任务复用。

LingBot release 只从 Qwen 目录提取 config、tokenizer 和 processor 文件，
排除 `model*.safetensors` 基模权重。LingBot 的完整部署模型权重来自
`hf_ckpt`。

构建前预览：

```bash
scripts/package_inference_image ... --dry-run
```

离线交付才增加：

```bash
--save-tar /mnt/pqssd/releases/kuavo-task-image.tar
```

脚本拒绝覆盖已有 TAR，也拒绝将 `.env`、私钥和常见凭据文件放入 release。

## 10. release 镜像部署

```bash
docker run --rm -it --init \
  --network host \
  --gpus all \
  --name kuavo-release-test \
  kuavo-task1-openpi:45000
```

容器内先检查：

```bash
cat /etc/kuavo/release_manifest.json
sed -n '1,180p' configs/deploy/kuavo_env.yaml
```

确认 ROS master、topics、相机、动作维度、绝对/相对动作和急停后手工执行：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

镜像默认进入 shell，不自动发布动作。OpenPI/LingBot-v2 Server 会由标准
入口自动管理。最终真机验收仍必须在宿主机提供 ROS 消息的官方环境完成。

## 11. 检查镜像实际增量

```bash
docker image ls
docker system df -v
docker history kuavo-task1-openpi:45000
```

判断磁盘压力时以 `docker system df -v` 的 shared/unique 为准，不要把
`docker image ls` 中每个 release 的累计 SIZE 简单相加。
