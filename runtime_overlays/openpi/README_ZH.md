# OpenPI 运行时覆盖模板

这个目录用于把 OpenPI 的运行环境镜像与任务资产解耦。同一个
`kuavo-openpi` 或 OpenPI release 镜像可以在容器启动时替换：

- Orbax 推理 checkpoint；
- checkpoint 对应的 `norm_stats.json`；
- PaliGemma tokenizer；
- ROS/任务部署 YAML；
- 每次重新推理前执行的动作步数。

这些覆盖不会修改镜像和 TAR。容器退出后覆盖消失。

## 1. 创建私有运行配置

Task1：

```bash
cp runtime_overlays/openpi/task1.env.example \
  /mnt/pqssd/runtime_configs/openpi-task1.env
```

Task2：

```bash
cp runtime_overlays/openpi/task2.env.example \
  /mnt/pqssd/runtime_configs/openpi-task2.env
```

编辑复制后的文件，至少确认：

- `IMAGE_NAME`
- `CHECKPOINT_DIR`
- `TOKENIZER_FILE`
- `NORM_STATS_FILE`
- `OPENPI_ASSET_ID`
- `DEPLOY_CONFIG`
- `EXECUTE_STEPS`
- `ROS_MASTER_URI`
- `ROS_IP`

`OPENPI_ASSET_ID` 必须与训练时的动态 asset ID 一致。例如：

```text
kuavo_task2_mix_2c74a207c5
```

OpenPI 会从下面的容器路径加载归一化统计：

```text
/models/checkpoint/assets/$OPENPI_ASSET_ID/norm_stats.json
```

启动脚本会创建一个很小的临时 assets overlay，只复制 norm JSON，不复制
模型参数。

容器启动时分别把 checkpoint 的 `params/` 与临时 `assets/` 挂到
`/models/checkpoint` 下的两个平级子目录。它不会先只读挂载整个
`/models/checkpoint` 再做嵌套挂载，因此不会触发 Docker 的
`read-only file system` 挂载错误。

## 2. 启动

```bash
runtime_overlays/openpi/start_openpi_overlay.sh \
  /mnt/pqssd/runtime_configs/openpi-task2.env
```

脚本在进入容器前验证所有宿主机资产，并在容器内再次检查：

- `/models/checkpoint/params/_METADATA`
- 动态 norm asset 路径
- `/assets/tokenizer/tokenizer.model`
- OpenPI 历史 tokenizer 兼容路径
- 被覆盖的 `kuavo_env.yaml`

验证完成后进入交互式 Bash。人工检查 YAML 后执行：

```bash
python kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_env.yaml
```

## 3. 动作执行长度

编辑 env 文件：

```bash
EXECUTE_STEPS=5
```

启动脚本会从任务模板生成临时 YAML，并覆盖其中的
`client_execute_steps`。OpenPI 模型动作块长度仍为 50；这里仅决定客户端
消费多少步后重新请求模型。

建议从 `1`、`3`、`5`、`8` 逐级测试，不要直接执行完整 50 步。

## 4. 快速切换权重

只需修改：

```bash
CHECKPOINT_DIR=/path/to/another/openpi/checkpoint
NORM_STATS_FILE=/path/to/its/norm_stats.json
OPENPI_ASSET_ID=the_matching_asset_id
```

如果任务类型不变，无需重建 Docker 镜像或重新导出 TAR。checkpoint 可以
包含额外训练状态；它只作为只读 bind mount 使用，OpenPI 推理只读取
`params/` 和选定的 `assets/<asset_id>/norm_stats.json`。

启动脚本会把 tokenizer 所在目录同时挂载到：

```text
/assets/tokenizer
/mnt/pqssd/pretrained/google/paligemma-3b-pt-224
```

因此它既适用于已经带兼容软链接的旧 release 镜像，也适用于完全不含任务
资产的 `kuavo-openpi` Runtime 基础镜像；不要求修改镜像内部文件。
