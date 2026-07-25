# Policy worker Docker 构建与运行

## 架构边界

ROS 节点留在宿主机，模型在独立 GPU 容器中运行。两者只通过统一的
MessagePack WebSocket 协议通信。这样 OpenPI（JAX/Python 3.11）、
LingBot-v1、LingBot-v2（Python 3.12/PyTorch 2.8）以及 classic
ACT/DP 不需要共享同一个 Python/CUDA 环境。

镜像只包含代码和运行环境。checkpoint、Qwen、norm stats 及 API key
必须在运行时只读挂载或通过权限为 600/400 的 env 文件注入。

## 构建

Classic、LingBot-v1 和 LingBot-v2 使用已经验收过的 conda-pack 环境：

```bash
CLASSIC_ENV_ARCHIVE=/secure/classic/myenv.tar.gz docker/build_classic.sh
LINGBOT_ENV_ARCHIVE=/secure/lingbot-v1/myenv.tar.gz docker/build_lingbot.sh
LINGBOT_V2_ENV_ARCHIVE=/secure/lingbot-v2/myenv.tar.gz docker/build_lingbot_v2.sh
docker/build_openpi.sh
```

LingBot-v2 的归档必须由其上游 `tools/create_train_env.sh` 创建的
Python 3.12 / PyTorch 2.8 环境生成。构建脚本通过 BuildKit named
context 读取归档，归档不进入源码 context。可先用 `DRY_RUN=1` 检查
完整命令。

OpenPI 直接复用固定子模块内的官方 `serve_policy.Dockerfile` 和
`uv.lock`，避免在 unified 仓库重复维护一套 JAX 依赖。

## 启动模型 worker

ACT 示例：

```bash
BACKEND=act \
MODEL_DIR=/data/checkpoints/act \
POLICY_PATH=/models/epochlast \
docker/run_policy_worker.sh
```

LingBot-v2 示例：

```bash
BACKEND=lingbot_v2 \
MODEL_DIR=/data/checkpoints/lingbot-v2 \
ASSET_DIR=/data/pretrained \
POLICY_PATH=/models/exported \
QWEN_PATH=/assets/Qwen3-VL-4B-Instruct \
NORM_STATS_FILE=/models/norm_stats.json \
ROBOT_NAME=kuavo_v2_right_arm \
docker/run_policy_worker.sh
```

OpenPI 使用其原生 server 参数：

```bash
BACKEND=openpi \
MODEL_DIR=/data/openpi \
SERVER_ARGS='policy:checkpoint --policy.config=pi05_kuavo --policy.dir=/models/checkpoint --port=8000' \
docker/run_policy_worker.sh
```

如需协议鉴权，创建仅含 `KUAVO_POLICY_API_KEY=...` 的 env 文件并执行
`chmod 600`，再设置 `API_ENV_FILE=/secure/policy.env`。runner 不会删除
已有容器或镜像；同名容器存在时会安全失败，要求操作者显式处理。

先用 `DRY_RUN=1` 检查挂载、镜像和参数。实际 GPU/ROS 验收时，再在宿主
机启动 `real_single_test.py` 或 open-loop viewer，并把 deploy YAML 的
policy host/port 指向 `127.0.0.1:8000`。
