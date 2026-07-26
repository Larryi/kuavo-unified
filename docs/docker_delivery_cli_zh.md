# Kuavo Docker 交互构建与交付

统一入口：

```bash
scripts/kuavo_docker
```

不带参数时交互选择 `build`、`shell`、`viewer`、`release` 或 `export`，再选择
ACT、DP、LingBot-v1、LingBot-v2 或 OpenPI。所有操作也支持完整命令行
参数，便于复现和云端脚本调用。

最终交付使用显式任务矩阵，CLI 会拒绝模型与任务错配：

| `--task` | backend | 机器人配置 |
|---|---|---|
| `task1-openpi` | `openpi` | Task1 右臂 + Leju 夹爪 |
| `task2-openpi` | `openpi` | Task2 双臂 + 双 Leju 夹爪 + 三相机 |
| `task1-lingbot-v1` | `lingbot-v1` | Task1 右臂 + Leju 夹爪、旧权重绝对动作 |
| `task2-lingbot-v1` | `lingbot-v1` | Task2 双臂 + 双 Leju 夹爪 + 三相机 |
| `task1-lingbot-v2` | `lingbot-v2` | Task1 右臂 + Leju 夹爪 |
| `task2-lingbot-v2` | `lingbot-v2` | Task2 双臂 + 双 Leju 夹爪 + 三相机 |
| `task2-dp` | `dp` | Task2 双臂 + 双 Leju 夹爪 + 三相机 |
| `task3-act` | `act` | Task3 右臂 + Qiangnao 末端 |

三种 VLA backend 默认选择已有交付权重对应的任务，但在自动化命令中必须
显式写出 `--task`。自定义 `--config` 只能覆盖同一模型/任务的细节，不能
绕过任务绑定。新增 VLA 路由的真实权重、open-loop 和真机 ROS 仍是发布
门禁。

## 安全边界

- 容器不会自动执行 `script_auto_test.py`。进入容器后必须先检查 YAML、
  ROS master、topic、相机、动作维度和末端类型，再由操作者手工启动。
- OpenPI 和 LingBot-v2 的 YAML 会让这个标准入口自动启动、等待并清理
  容器内模型 Server；操作者不需要再开第二个 shell 手工启动 Server。
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

OpenPI 使用 `uv sync --frozen` 保持子模块锁定的版本和哈希。镜像构建会在
容器内副本中把锁文件的 PyPI registry 与 `files.pythonhosted.org` wheel
URL 映射到 BFSU，不修改子模块工作树。下载继续使用原
`/root/.cache/uv` BuildKit cache；只要没有执行 BuildKit prune，后续因
统一仓库源码变化而重建时仍可复用已下载包。

LingBot-v2 基础镜像：

```bash
scripts/kuavo_base_image \
  --backend lingbot-v2 \
  --build \
  --env-archive /mnt/pqssd/docker_envs/lingbot-v2/myenv.tar.gz
```

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
指定文件只读挂载到该路径。生成的 YAML 带
`client_autostart_backend: openpi`；标准推理入口会直接启动 JAX Server，
等待端口就绪后再创建 ROS PolicyClient，退出时自动清理 Server。

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

## Open-loop Viewer：匹配的 Base 容器

Viewer 也通过统一入口运行，不要求宿主机安装模型包。checkpoint、数据集
和 processor 仍然只读挂载，不写入 Base 镜像：

```bash
scripts/kuavo_docker viewer \
  --backend openpi \
  --checkpoint /data/openpi/45000 \
  --tokenizer /data/paligemma/tokenizer.model \
  --dataset /data/lerobot_task1_345

scripts/kuavo_docker viewer \
  --backend lingbot-v1 \
  --checkpoint /data/lingbot-v1/hf_ckpt \
  --qwen /data/Qwen2.5-VL-processor \
  --norm-stats /data/lingbot-v1/norm_stats.json \
  --dataset /data/lerobot_task1_345

scripts/kuavo_docker viewer \
  --backend lingbot-v2 \
  --checkpoint /data/lingbot-v2/hf_ckpt \
  --qwen /data/Qwen3-VL-processor \
  --norm-stats /data/lingbot-v2/norm_stats.json \
  --dataset /data/lerobot_task2_264 \
  --viewer-port 8502
```

浏览器访问打印的 `http://127.0.0.1:<port>`。界面内统一使用容器路径
`/data/dataset`、`/models/checkpoint`、`/assets/qwen` 和
`/assets/norm_stats/norm_stats.json`。OpenPI Viewer 命令会在同一容器
自动管理 JAX Server；LingBot-v1/v2 则直接在各自模型环境中加载策略。
任务 profile 还会预填 Repo ID、真实 dataset task 文本、state/action
维度和 LingBot robot preset。OpenPI 不依赖原生 Server metadata 提供
`action_dim`。

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
