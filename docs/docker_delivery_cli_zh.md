# Kuavo Docker 交互构建与交付

统一入口：

```bash
scripts/kuavo_docker
```

不带参数时交互选择 `build`、`shell`、`release` 或 `export`，再选择
ACT、DP、LingBot-v1、LingBot-v2 或 OpenPI。所有操作也支持完整命令行
参数，便于复现和云端脚本调用。

仓库内新增的 ACT、DP、LingBot-v1 模板以及现有 OpenPI/LingBot-v2
模板均以 Task1 右臂 + Leju 夹爪为安全起点。Task2 双臂/双腕相机和
Task3 Qiangnao 末端必须传入对应的 `--config`，不能只修改 checkpoint。

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

Classic ACT/DP：

```bash
scripts/kuavo_docker build \
  --backend act \
  --env-archive /mnt/pqssd/docker_envs/classic/myenv.tar.gz
```

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

LingBot-v2 当前仍被标记为非 ROS-ready。CLI 允许检查其 build 路由，但会
阻止 `shell` 和 `release`，直到镜像包含 ROS Noetic 和 KuavoBaseEnv。

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
  --checkpoint /data/openpi/checkpoint \
  --tokenizer /data/paligemma/tokenizer.model \
  --openpi-config pi05_kuavo
```

当前固定 OpenPI 配置仍使用历史 PaliGemma tokenizer 绝对路径，CLI 会把
指定文件只读挂载到该路径。进入容器后先启动 Server 和检查 YAML：

```bash
export SERVER_ARGS='policy:checkpoint --policy.config=pi05_kuavo \
--policy.dir=/models/checkpoint --port=8000'
docker/start_openpi_ros.sh bash
```

CLI 会为本次测试生成并显示一个待检查 YAML，挂载到：

```text
/run/kuavo/kuavo_env.yaml
```

交互模式会把完整 YAML 打印出来并等待确认。非交互脚本必须显式传入
`--yes` 才会真正创建容器；`--dry-run` 不需要确认。

检查完成后由操作者手动运行：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config /run/kuavo/kuavo_env.yaml
```

使用 `--dry-run` 只打印 Docker 命令，不创建容器：

```bash
scripts/kuavo_docker shell ... --dry-run
```

## 最终 release 镜像

将同样的测试资产固化进派生镜像：

```bash
scripts/kuavo_docker release \
  --backend lingbot-v1 \
  --checkpoint /data/lingbot/hf_ckpt \
  --qwen /data/runtime/Qwen2.5-VL-3B-Instruct-processor \
  --norm-stats /data/lingbot/norm_stats.json \
  --tag kuavo-lingbot-v1-release:task1
```

统一镜像内路径：

| 资产 | 容器路径 |
|---|---|
| checkpoint bundle | `/models/checkpoint` |
| Qwen processor | `/assets/qwen` |
| norm stats | `/assets/norm_stats/norm_stats.json` |
| 其他运行资产 | `/assets/runtime` |
| OpenPI tokenizer | `/assets/tokenizer/tokenizer.model` |
| 已渲染 deploy YAML | `configs/deploy/kuavo_env.release.yaml` |

release 镜像仍以 `bash` 为默认入口，不自动发布动作。

## 明确导出 TAR

只在最终需要交付文件时执行：

```bash
scripts/kuavo_docker export \
  --image kuavo-lingbot-v1-release:task1 \
  --output /data/releases/kuavo-lingbot-v1-task1.tar \
  --yes
```

输出文件已存在时 CLI 拒绝覆盖。测试阶段不要执行此命令。
