# Policy worker Docker 构建与运行

## 架构边界

每种最终算法镜像都必须包含 ROS Noetic 和 Kuavo 部署端，能够通过
host network 与宿主机 ROS master、相机和机器人通讯。模型服务与 ROS
Client 仍通过统一 MessagePack WebSocket 协议隔离进程和 Python 环境，
但二者位于同一个最终镜像内，默认通过 `127.0.0.1` 通讯。

OpenPI 镜像继承已验证的 Classic ROS 镜像：ROS Client 使用 Classic
Python 3.10 环境，OpenPI Server 使用独立 Python 3.11/JAX uv 环境。
依赖不混装，同时满足单镜像交付。LingBot-v1/v2 镜像也必须保留 ROS，
不能交付为纯模型 Worker 镜像。

镜像只包含代码和运行环境。checkpoint、Qwen、norm stats 及 API key
必须在运行时只读挂载或通过权限为 600/400 的 env 文件注入。

## 构建

Classic、LingBot-v1 和 LingBot-v2 使用已经验收过的 conda-pack 环境：

```bash
CLASSIC_ENV_ARCHIVE=/secure/classic/myenv.tar.gz docker/build_classic.sh
LINGBOT_ENV_ARCHIVE=/secure/lingbot-v1/myenv.tar.gz docker/build_lingbot.sh
LINGBOT_V2_ENV_ARCHIVE=/secure/lingbot-v2/myenv.tar.gz docker/build_lingbot_v2.sh
docker/build_openpi.sh  # 要求本地已有 kuavo-classic:latest
```

LingBot-v2 的归档必须由其上游 `tools/create_train_env.sh` 创建的
Python 3.12 / PyTorch 2.8 环境生成。构建脚本通过 BuildKit named
context 读取归档，归档不进入源码 context。可先用 `DRY_RUN=1` 检查
完整命令。

OpenPI 使用固定子模块的 `uv.lock`，通过独立 BuildKit context 导入
源码；最终镜像以 `kuavo-classic:latest` 为基础，增加 OpenPI uv 环境
和 `docker/start_openpi_ros.sh`。因此最终只有一个 ROS-capable 镜像，
而不是 Ubuntu 22.04 的纯推理 Worker。

## 启动模型服务

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
SERVER_ARGS='--port=8000 policy:checkpoint --policy.config=pi05_kuavo --policy.dir=/models/checkpoint' \
docker/run_policy_worker.sh
```

该命令启动单一 `kuavo-openpi:latest` 容器。默认只启动容器内 Policy
Server，操作者随后可在同一容器中执行 ROS 推理命令；也可以直接调用：

```bash
docker/start_openpi_ros.sh \
  python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_env.openpi_client.yaml
```

启动器先等待 localhost Policy Server 监听，再启动 ROS Client，并在
任一进程退出时清理另一进程。真机动作命令仍必须由操作者在急停可用时
显式给出，镜像默认不会自动发布动作。

如需协议鉴权，创建仅含 `KUAVO_POLICY_API_KEY=...` 的 env 文件并执行
`chmod 600`，再设置 `API_ENV_FILE=/secure/policy.env`。runner 不会删除
已有容器或镜像；同名容器存在时会安全失败，要求操作者显式处理。

先用 `DRY_RUN=1` 检查挂载、镜像和参数。实际 GPU/ROS 验收时使用
`--network host`，容器内 deploy YAML 的 policy host/port 保持
`127.0.0.1:8000`，ROS master 则指向宿主机地址。
