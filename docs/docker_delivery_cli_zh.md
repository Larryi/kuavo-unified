# Kuavo Docker 交互构建与交付

统一入口：

```bash
scripts/kuavo_docker
```

不带参数时交互选择 `build`、`shell`、`release` 或 `export`，再选择
ACT、DP、LingBot-v1、LingBot-v2 或 OpenPI。所有操作也支持完整命令行
参数，便于复现和云端脚本调用。

最终交付使用显式任务矩阵，CLI 会拒绝模型与任务错配：

| `--task` | backend | 机器人配置 |
|---|---|---|
| `task1-openpi` | `openpi` | Task1 右臂 + Leju 夹爪 |
| `task1-lingbot-v1` | `lingbot-v1` | Task1 右臂 + Leju 夹爪、旧权重绝对动作 |
| `task2-dp` | `dp` | Task2 双臂 + 双 Leju 夹爪 + 三相机 |
| `task3-act` | `act` | Task3 右臂 + Qiangnao 末端 |

每个 backend 都有上述唯一默认任务，也建议在自动化命令中显式写出
`--task`。自定义 `--config` 只能覆盖同一模型/任务的细节，不能绕过任务
绑定。LingBot-v2 当前暂缓，CLI 会阻止其 `shell` 和 `release`。

## 安全边界

- 容器不会自动执行 `script_auto_test.py`。进入容器后必须先检查 YAML、
  ROS master、topic、相机、动作维度和末端类型，再由操作者手工启动。
- `shell` 只读挂载 checkpoint 和运行资产，不把它们复制进环境镜像。
- `release` 将选定资产固化进新的派生镜像，但不生成 TAR。
- 只有显式执行 `export --yes` 才调用 `docker save`。
- HF/W&B/ServerChan/Vast token 和其他凭据不得作为 checkpoint 或 runtime
  asset 输入。
- `release` 会扫描 `.env`、私钥、credentials/secrets 文件并拒绝把它们
  固化进镜像。

## 构建环境镜像

推荐通过基础镜像入口操作。默认只检查镜像存在，不生成 TAR：

```bash
scripts/kuavo_base_image --backend dp
scripts/kuavo_base_image --backend act       # 与 DP 共用 Classic
scripts/kuavo_base_image --backend openpi
scripts/kuavo_base_image --backend lingbot-v1
```

需要重建时显式加入 `--build`。Classic 与 LingBot 还需要环境归档：

```bash
scripts/kuavo_base_image \
  --backend dp \
  --build \
  --env-archive /mnt/pqssd/docker_envs/classic/myenv.tar.gz

scripts/kuavo_base_image \
  --backend lingbot-v1 \
  --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz

scripts/kuavo_base_image --backend openpi --build
```

仅在需要把基础环境搬到离线机器时导出：

```bash
scripts/kuavo_base_image \
  --backend openpi \
  --save-tar /mnt/pqssd/releases/kuavo-openpi-base.tar
```

输出文件存在时拒绝覆盖。基础镜像不包含训练 checkpoint、任务 norm stats、
数据集或凭据。

Classic ACT/DP：

```bash
scripts/kuavo_docker build \
  --backend act \
  --env-archive /mnt/pqssd/docker_envs/classic/myenv.tar.gz
```

构建脚本会校验并预置官方 `resnet18-f37072fd.pth` 到镜像的 Torch cache，
避免 ACT 首次推理在线下载。默认复用宿主机 Torch cache；若文件不存在，
构建阶段会断点重试下载。可用 `RESNET18_CHECKPOINT` 指定已有权重文件。

LingBot-v1：

```bash
scripts/kuavo_docker build \
  --backend lingbot-v1 \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz
```

OpenPI 复用已经构建的 `kuavo-classic:latest`：

```bash
scripts/kuavo_docker build --backend openpi
```

LingBot-v2 环境与最终镜像适配已按操作者决定暂缓，不属于本轮交付。

## 测试阶段：只读挂载并进入 shell

ACT 示例：

```bash
scripts/kuavo_docker shell \
  --backend act \
  --checkpoint /data/act_deploy_bundle \
  --checkpoint-subpath epochlast
```

`--checkpoint` 是挂载到 `/models/checkpoint` 的完整部署 bundle；
`--checkpoint-subpath` 是其中实际传给 policy loader 的相对目录。
ACT/DP bundle 应同时包含 `policy_preprocessor.json` 和
`policy_postprocessor.json`，否则 CLI 会警告。

LingBot-v1：

```bash
scripts/kuavo_docker shell \
  --backend lingbot-v1 \
  --checkpoint /data/lingbot/hf_ckpt \
  --qwen /data/runtime/Qwen2.5-VL-3B-Instruct-processor \
  --norm-stats /data/lingbot/norm_stats.json
```

LingBot checkpoint 必须是可由 `strict=True` 完整加载的 HF checkpoint。
LoRA、adapter 或 DCP 必须先合并导出。部署用 Qwen 目录只需
`AutoConfig`/`AutoProcessor` 所需的 config、tokenizer 和图像 processor
文件；如果目录还包含大模型权重，CLI 会报告体积但不会擅自删除。

OpenPI：

```bash
scripts/kuavo_docker shell \
  --backend openpi \
  --checkpoint /data/openpi/45000 \
  --tokenizer /data/paligemma/tokenizer.model \
  --openpi-config pi05_kuavo
```

`--checkpoint` 应指向包含 `params/_METADATA` 的训练 step 目录（例如
`45000`），因为 OpenPI 会在 `--policy.dir` 后追加 `params`。如果误传
`45000/params`，CLI 会自动提升到父级 step 目录，避免容器内形成错误的
`/models/checkpoint/params/params` 路径。

当前固定 OpenPI 配置仍使用历史 PaliGemma tokenizer 绝对路径，CLI 会把
指定文件只读挂载到该路径。进入容器后先启动 Server 和检查 YAML：

```bash
OPENPI_POLICY_CONFIG=pi05_kuavo \
OPENPI_POLICY_DIR=/models/checkpoint \
docker/start_openpi_ros.sh bash
```

推荐使用上述结构化变量，避免多行 `SERVER_ARGS` 中的引号或反斜杠被当作
参数传给 Tyro。旧的单行 `SERVER_ARGS` 仍兼容，但端口是顶层选项，必须
写在子命令之前：

```bash
SERVER_ARGS='--port=8000 policy:checkpoint --policy.config=pi05_kuavo --policy.dir=/models/checkpoint'
```

CLI 会为本次测试生成并显示一个待检查 YAML，挂载到：

```text
/run/kuavo/kuavo_env.yaml
/root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml
```

两处是同一个只读文件。第二处覆盖镜像内可能属于其他模型的示例配置，
确保官方标准入口不会因读取旧 `kuavo_env.yaml` 而路由到错误后端。
交互模式会把完整 YAML 打印出来并等待确认。非交互脚本必须显式传入
`--yes` 才会真正创建容器；`--dry-run` 不需要确认。

检查完成后由操作者手动运行：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

使用 `--dry-run` 只打印 Docker 命令，不创建容器：

```bash
scripts/kuavo_docker shell ... --dry-run
```

## 最终 release 镜像

面向操作者的推理打包入口是：

```bash
scripts/package_inference_image \
  --task task2 \
  --algorithm dp \
  --checkpoint /path/to/dp/run \
  --checkpoint-subpath epochbest
```

它根据任务与算法路由到同一个 release 构建器。默认只生成 Docker 镜像；
只有显式传 `--save-tar /path/image.tar` 才随后执行 `docker save`。
自动化调用应增加 `--yes`，人工交互时会在固化资产前确认。

将同样的测试资产固化进派生镜像：

```bash
scripts/kuavo_docker release \
  --backend lingbot-v1 \
  --task task1-lingbot-v1 \
  --checkpoint /data/lingbot/hf_ckpt \
  --qwen /data/runtime/Qwen2.5-VL-3B-Instruct-processor \
  --norm-stats /data/lingbot/norm_stats.json \
  --tag kuavo-lingbot-v1-release:task1
```

等价的新入口：

```bash
scripts/package_inference_image \
  --task task1 \
  --algorithm lingbot-v1 \
  --checkpoint /data/lingbot/hf_ckpt \
  --qwen /data/runtime/Qwen2.5_VL \
  --norm-stats /data/lingbot/norm_stats.json
```

统一镜像内路径：

| 资产 | 容器路径 |
|---|---|
| checkpoint bundle | `/models/checkpoint` |
| Qwen processor | `/assets/qwen` |
| norm stats | `/assets/norm_stats/norm_stats.json` |
| 其他运行资产 | `/assets/runtime` |
| OpenPI tokenizer | `/assets/tokenizer/tokenizer.model` |
| 已渲染 deploy YAML | `configs/deploy/kuavo_env.yaml`（另存 `kuavo_env.release.yaml`） |

release 镜像仍以 `bash` 为默认入口，不自动发布动作。
每个镜像还包含 `/etc/kuavo/release_manifest.json`，记录 backend、任务、
policy 类型、容器内 checkpoint 路径和基础镜像。默认标签为
`kuavo-<task>-release:latest`。

ACT/DP 使用 run 根目录作为 `--checkpoint`，再通过
`--checkpoint-subpath epochbest` 选择权重。release 只复制选中的 epoch、
processor 与必要 config，不会把其他 epochs、optimizer、event log 一起
固化进镜像。

LingBot 的 `--qwen` 可以指向完整本地 Qwen 目录；release 会只提取
config、tokenizer 和 processor 文件，明确排除 `model*.safetensors`，
不会修改源目录。

## 明确导出 TAR

只在最终需要交付文件时执行：

```bash
scripts/kuavo_docker export \
  --image kuavo-lingbot-v1-release:task1 \
  --output /data/releases/kuavo-lingbot-v1-task1.tar \
  --yes
```

输出文件已存在时 CLI 拒绝覆盖。测试阶段不要执行此命令。
