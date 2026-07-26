# Kuavo 训练、带权重 Docker 交付与 VastAI 全流程

本文是当前交付主入口。支持矩阵：

| 任务 | 算法 | 本地/云端训练 | ROS Noetic 推理镜像 | 多数据集比例混合 |
|---|---|---|---|---|
| Task1 | OpenPI Pi0.5 | 支持 | 支持 | 支持 |
| Task1 | LingBot-VLA v1 | 支持 | 支持 | 支持 |
| Task2 | DP | 支持 | 支持 | 支持 |
| Task3 | ACT | 支持 | 支持 | 支持 |
| Task2 | LingBot-VLA v2 | 支持（待真实云端训练验收） | 支持（待真实权重/ROS 验收） | 支持 |

“多数据集混合”使用虚拟加权采样，不复制或合并源数据。
各源必须具有相同机器人 embodiment、FPS、state/action schema 和相机键。
比例不必合计为 1，向导会自动归一化。DP/ACT 会按混合权重合并
mean/std/min/max，并让学习率调度器使用虚拟 epoch 长度；LingBot-v1 的
norm stats 与训练 sampler 使用同一个虚拟混合分布。

## 一、源码同步策略

新的 VastAI 实例默认应从 Git 恢复，不应从本机复制整个源码树。前提是先
把主仓库提交和每个 submodule 固定 commit 推送到对应仓库。主仓库的
`.gitmodules` 已记录 LeRobot、LingBot 和个人 OpenPI/LingBot-v2 fork；
远端 clone 后严格执行递归 submodule 初始化，因此取得的是主仓库 pin 的
版本，而不是各上游当时的最新 main。

本地执行：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4 -L 8080:localhost:8080'
```

脚本会解析 SSH user、host、port 和端口转发参数，只把小型
`remote_clone_and_restore.sh` 上传到远端 `/tmp`。该脚本在远端 clone：

```text
/workspace/kuavo_unified_stack
```

并执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

解析后的主仓库及 submodule commit 记录在 `.vast_git_manifest.json`。
正式连接前，本地入口还会检查工作区干净、主分支 HEAD 已发布，并确认每个
submodule commit 可以从其配置仓库按 SHA 获取。任一提交未推送都会在复制
bootstrap 前失败并指出具体仓库；`--dry-run` 不联网，因此跳过发布检查。
需要其他目录时增加：

```bash
--remote-root /workspace/my_kuavo
```

只 clone/更新 Git，不进入远端向导：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4' \
  --no-launch
```

正式执行前可使用 `--dry-run`。脚本不使用 `eval`，也不接受 SSH 命令中的
远端 shell 命令，避免把复制来的字符串当作本地命令执行。

只有本地确实有本次实验必须使用、但不能先推送的改动时，才显式使用：

```bash
scripts/vast/bootstrap_from_ssh \
  --ssh-command 'ssh -p 12345 root@1.2.3.4' \
  --sync-working-tree
```

这个 fallback 才会 rsync 主仓库和已检出的 submodule 内容，并排除 `.git`、
凭据、输出、checkpoint、虚拟环境和 TAR。正常新实例不要使用该选项。

## 二、VastAI 远端恢复向导

同步结束后默认进入：

```bash
scripts/vast/restore_and_launch.sh
```

它执行以下步骤：

1. 在 `/workspace/kuavo_bootstrap` 创建小型 bootstrap venv；
2. 默认使用官方 `https://pypi.org/simple` 安装 `huggingface_hub`；
3. 输入 HF token，每个字符以 `*` 回显，并调用 `whoami` 验证身份；
4. 列出当前 HF 用户及组织可见的 dataset/model 仓库；
5. 选择算法，自动绑定任务；
6. 选择数据集；OpenPI、DP、ACT、LingBot-v1 均使用“逐个添加、确认是否
   继续”的流程选择多个，然后输入对应比例；
7. 选择训练输出模型仓库；
8. 可选选择 resume 模型仓库并检查完整训练状态标记；
9. 输入 W&B、ServerChan、VastAI 凭据，均以 `*` 回显；
10. 生成权限 `0600` 的私有 env 和不含凭据的 JSON 任务清单；
11. 用户再次确认后才恢复算法环境并启动训练。

默认文件：

```text
.secrets/interactive-job.env       # 包含凭据，0600，不可提交
logs/interactive-job.json          # 不含凭据，可用于审核
```

HF token 至少需要：

- 读取所选私有 dataset；
- 读取所选预训练/resume model；
- 创建或写入输出 model 仓库。

ServerChan 可留空；配置后成功或失败都会通知。只有同时提供
`VAST_INSTANCE_ID`、`VAST_API_KEY` 并确认自动关机，成功结束后才关闭实例。
默认失败不关机，以便保留现场排障。

若某个特定远端网络必须使用其他 PyPI 源，可在启动恢复脚本前显式设置
`PIP_INDEX_URL`；默认不使用 BFSU。

## 三、数据集选择和下载

向导把选择保存为 `DATASET_MIX_JSON`：

```json
[
  {"name": "source_01_task1_sz", "repo_id": "owner/task1_sz", "weight": 0.25},
  {"name": "source_02_task1_bj", "repo_id": "owner/task1_bj", "weight": 0.75}
]
```

各算法云端流水线逐个下载到各自工作目录下的：

```text
/workspace/kuavo_runs/openpi/datasets/mixture/
```

下载完成后再次检查 LeRobot v3、机器人类型、FPS、action/state 维度和相机
schema，再把本地 root 写入 resolved manifest。norm stats 使用相同的
weighted sampler 重新计算，并以 mixture hash 隔离缓存，避免误用另一种
数据比例的统计。

多数据集选择示意：

```text
选择或输入训练数据集: 1
已添加 1 个数据集：owner/task1_suzhou
继续添加另一个数据集？ [y/N]: y
选择或输入训练数据集: 2
已添加 2 个数据集：owner/task1_beijing
继续添加另一个数据集？ [y/N]: n
依次输入 2 个正数配比（逗号分隔，如 55,20,25）: 25,75
```

resolved manifest 会保留每个 HF repo、本地 root 和归一化权重。训练时
均从该清单读取，不支持用逗号拼接路径。LingBot-v2 使用 Task2 bimanual
schema，并按同一虚拟混合分布重新计算 norm stats。

## 四、预训练权重和 resume

无需 resume 时，各算法自动准备：

- DP/ACT：训练环境中的 torchvision ResNet18 ImageNet 权重；
- OpenPI：Pi0.5 Orbax base params 和 PaliGemma tokenizer；
- LingBot-v1：LingBot-VLA 基模及 Qwen2.5-VL tokenizer/processor。

resume 仓库必须是完整训练状态，不是部署权重：

| 算法 | 必要标记 |
|---|---|
| OpenPI | 数字 step 目录及 `params/_METADATA`、优化器状态等 |
| LingBot-v1 | `checkpoints/global_step_*` DCP 目录 |
| DP/ACT 单卡 | `learning_state.pth`、`rng_state.pth`、模型和 processors |
| DP/ACT Accelerate | `epochlatest/`、`training_latest_state.pth` 等 |

向导会先读取 HF repo 文件清单并检查算法对应标记。随后还要求输入原始
`RUN_ID`，下载到训练器期望的原目录，再传入正确的 resume 参数。若只有
`model.safetensors` 或 LingBot `hf_ckpt`，只能推理，不能无损续训。

### checkpoint 保存与磁盘保留

| 算法 | 保存触发 | 默认保留 |
|---|---|---|
| OpenPI | 每 `SAVE_INTERVAL=1000` step | Orbax `max_to_keep=1`；云端 `KEEP_PERIOD` 设为极大值，正常只保留最新完整 step |
| LingBot-v1 | 每 `SAVE_STEPS=500` step | `KEEP_LAST_CHECKPOINTS=1`，只保留最新完整 DCP；结束后另导出部署 HF checkpoint，并把两者上传到同一模型仓库 |
| LingBot-v2 | Task2 默认每 5000 step | `keep_last_checkpoints=1`，只保留最新完整 DCP；同时生成已合并 LoRA 的完整 `hf_ckpt` |
| DP/ACT | 每个 epoch 覆盖 latest resume；验证更优时更新 `epochbest` | 默认 `keep_last_epoch_checkpoints=0`，不生成重复 `epochN`；latest 模型/训练状态只留一份，`epochbest` 作为部署候选分开保留 |

LingBot-v1 会把最新 DCP 上传到模型仓库的 `checkpoints/global_step_*`，
因此 resume 与部署产物同时可用。上传及远端结构校验成功后，如设置
`DELETE_LOCAL_DCP_AFTER_UPLOAD=1`，会删除本地 DCP；默认值仍为 `0`，以免
在首次云端验收前误删可恢复现场。

## 五、训练入口

云端向导最终统一调用：

```bash
scripts/vast/run_backend.sh
```

阶段状态写入：

```text
/workspace/kuavo_runs/<backend>/logs/<run-id>/status.json
```

本地查看有限日志和 GPU：

```bash
TRAINING_TASK=task1 MODEL_BACKEND=openpi \
VAST_SSH_HOST=1.2.3.4 VAST_SSH_PORT=12345 \
scripts/vast/status.sh
```

默认只读取 30 行日志。持续刷新：

```bash
WATCH_SECONDS=30 TAIL_LINES=40 \
TRAINING_TASK=task1 MODEL_BACKEND=openpi \
VAST_SSH_HOST=1.2.3.4 VAST_SSH_PORT=12345 \
scripts/vast/status.sh
```

本地直接训练的算法命令、checkpoint 结构和 open-loop 验证分别见：

- `docs/dp_act_workflow_zh.md`
- `docs/openpi_workflow_zh.md`
- `docs/lingbot_workflow_zh.md`

## 六、训练后验证

训练成功并上传 HF 后，先把 checkpoint 下载到本地数据盘。至少完成：

1. 检查 checkpoint 结构；
2. 运行对应算法的无 ROS smoke/open-loop；
3. 使用 Viewer 检查动作曲线和相机输入；
4. 再进行只读挂载的 Docker mock；
5. 最后才生成带权重推理镜像。

不要把 DCP/optimizer 状态直接当部署 checkpoint：

- DP/ACT 选择 `epochbest` 或明确 epoch，并携带 processors；
- OpenPI 选择包含 `params/_METADATA` 的 step 目录；
- LingBot-v1 先导出完整 HF checkpoint，并携带 norm stats 与 Qwen
  processor/tokenizer。

## 七、基础镜像

基础镜像只有算法环境、ROS Noetic、KuavoBaseEnv 和源码，不含任务权重：

```bash
scripts/kuavo_base_image --backend dp --build \
  --env-archive /path/to/classic/myenv.tar.gz

scripts/kuavo_base_image --backend openpi --build

scripts/kuavo_base_image --backend lingbot-v1 --build \
  --env-archive /path/to/lingbot/myenv.tar.gz
```

DP 与 ACT 共用 `kuavo-classic:latest`。基础镜像不需要为每个 checkpoint
重复构建。只有离线传输时才增加：

```bash
--save-tar /path/to/base-image.tar
```

## 八、带权重推理镜像

训练验证通过后，按任务、算法和 checkpoint 生成派生镜像：

```bash
scripts/package_inference_image \
  --task task2 \
  --algorithm dp \
  --checkpoint /path/to/task2_dp_run \
  --checkpoint-subpath epochbest \
  --tag kuavo-task2-dp:epochbest
```

OpenPI：

```bash
scripts/package_inference_image \
  --task task1 \
  --algorithm openpi \
  --checkpoint /path/to/openpi/45000 \
  --tokenizer /path/to/paligemma/tokenizer.model \
  --tag kuavo-task1-openpi:45000
```

LingBot-v1：

```bash
scripts/package_inference_image \
  --task task1 \
  --algorithm lingbot-v1 \
  --checkpoint /path/to/hf_ckpt \
  --qwen /path/to/Qwen2.5_VL \
  --norm-stats /path/to/norm_stats.json \
  --tag kuavo-task1-lingbot-v1:run-id
```

Qwen 只提取 config/tokenizer/processor，不复制 Qwen 基模权重。默认仅生成
本地 Docker image，不执行 `docker save`。确实需要交付 TAR 时增加：

```bash
--save-tar /path/to/kuavo-task-image.tar
```

## 九、ROS 部署

带权重镜像仍以 `bash` 为默认入口，不自动发布机器人动作。使用
`scripts/kuavo_docker shell` 或等价 `docker run --network host --gpus all`
启动后，人工检查：

- `configs/deploy/kuavo_env.yaml` 的 policy type 和 checkpoint；
- ROS master、topic、相机键和图像维度；
- Task2 双臂/双夹爪与 Task3 Qiangnao 末端配置；
- OpenPI/LingBot server 是否使用 GPU；
- 动作维度、绝对/相对动作语义和安全限幅。

确认后手工执行官方入口：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

真机宿主机必须发布 ROS 消息，因此镜像构建成功和 mock 成功都不能替代
最终真机验收。
