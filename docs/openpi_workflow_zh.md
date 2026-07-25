# OpenPI Pi0.5 统一入口

OpenPI 使用独立的 Python 3.11/JAX 环境，不与 ROS、classic LeRobot 或
LingBot 的依赖混装。源码固定为子模块 `third_party/openpi-kuavo`，统一入口：

```bash
scripts/kuavo_openpi --help
```

首次使用先初始化子模块并在 OpenPI 目录创建原生 `.venv`。如果复用已有环境：

```bash
export OPENPI_PYTHON=/path/to/openpi/.venv/bin/python
scripts/kuavo_openpi check
```

OpenPI 的 Hugging Face 缓存默认写入 `/tmp`；需要持久缓存时使用
`OPENPI_HF_HOME` 和 `OPENPI_HF_DATASETS_CACHE`，JAX 编译缓存可用
`OPENPI_JAX_CACHE_DIR` 指定，避免继承到只读数据盘路径。

## 归一化统计

Task1 已把当前 345-episode repaired 数据集的统计纳入
`assets/openpi/pi05_kuavo/kuavo_task1/norm_stats.json`。更换数据集后必须重算：

```bash
scripts/kuavo_openpi norm-stats pi05_kuavo \
  --dataset-root /path/to/lerobot_dataset \
  --tokenizer-path /path/to/tokenizer.model
```

统计会写入统一仓库的 `assets/openpi/<config>/<asset-id>/`，不会遗留在
OpenPI 子模块的 ignored `assets/` 中。

## 训练

```bash
scripts/kuavo_openpi train pi05_kuavo \
  --exp-name=task1_full \
  --num-train-steps=30000
```

默认 Task1/Task2 配置使用 Pi0.5 JAX、50-step action horizon。Task1 是 8D
右臂加夹爪；Task2 是 16D 双臂加双夹爪。配置中的基础权重默认位置为
`/mnt/pqssd/pretrained/pi05_local_jax/params`。云端启动器必须在训练前确保：

- tokenizer 位于配置指定位置；
- 基础 Orbax 权重包含 `params/_METADATA`；
- 对应 norm stats 已同步；
- 训练检查点目录持久化。

## 推理服务

```bash
scripts/kuavo_openpi serve \
  policy:checkpoint \
  --policy.config=pi05_kuavo \
  --policy.dir=/path/to/checkpoint \
  --port=8000
```

训练检查点应携带自己的 `assets/<asset-id>/norm_stats.json`；服务端优先使用
检查点资产，避免误用后来重算的统计。

离线单样本冒烟：

```bash
scripts/kuavo_openpi smoke \
  --config-name pi05_kuavo \
  --checkpoint /path/to/checkpoint
```

## Open-loop 与 Viewer

```bash
scripts/kuavo_openpi open-loop \
  --config-name pi05_kuavo \
  --checkpoint /path/to/checkpoint \
  --episodes 0 \
  --max-samples 3 \
  --require-cuda
```

```bash
scripts/kuavo_openpi viewer
```

命令行评测和 Viewer 会先尝试统一资产目录；若其中没有统计，则从训练
检查点的 `assets/` 回退加载。Open-loop 输出是无 ROS 的动作诊断，不替代
宿主机 ROS 消息、动作执行频率和真机安全验收。

## ROS 侧隔离调用

OpenPI server 与 ROS client 不需要共享 Python 环境。ROS 侧使用
`configs/deploy/kuavo_env.openpi_client.yaml` 和
`requirements_policy_client.txt`，通过兼容 OpenPI 的 msgpack WebSocket
协议取得动作块。配置、reset/health/metadata 语义和安全默认值见
`docs/policy_protocol_zh.md`。
