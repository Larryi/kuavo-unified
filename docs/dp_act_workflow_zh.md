# DP / ACT 训练与 Open-loop 验证

本文覆盖当前统一仓库中最基础的 DP、ACT 本地闭环：训练、断点续训、
无 ROS 推理评测，以及可视化检查。数据转换已经完成，不在本文重复说明。

## 环境与检查点约定

训练运行目录必须同时保存：

- 根目录模型（最新模型）；
- `epochN/` 和 `epochbest/` 模型快照；
- `policy_preprocessor.json`、`policy_postprocessor.json` 及其权重；
- `learning_state.pth` 和 `rng_state.pth`。

推理既可传运行根目录，也可传 `epochN/`。加载器会从快照目录向上查找
配套的 pre/post processor，并把 processor 的设备改为本次指定的
`--device`，因此 GPU 训练得到的检查点也能用于 CPU 冒烟。

如数据盘上的 Hugging Face cache 不可写，先指定本机可写缓存：

```bash
export HF_DATASETS_CACHE=/tmp/kuavo-hf-datasets
```

## 单卡训练

Task1 DP 示例：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=dp_r1.yaml \
  task=task1 \
  method=dp \
  root=/mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345 \
  repoid=task1_repaired_345 \
  training.device=cuda \
  training.batch_size=32
```

Task3 ACT 示例：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=act_config.yaml \
  task=task3 \
  method=act \
  root=/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165 \
  repoid=task3_repaired_165 \
  training.device=cuda \
  training.batch_size=32
```

`training.max_training_step` 以 optimizer update 为单位；设置后到达该步数
即保存最新模型和训练状态并退出。梯度累积会在完整窗口后更新，epoch 末尾
不足一个窗口的 batch 也会被正确缩放并提交。

## 断点续训

续训会原地复用指定的 `run_...` 目录，不再把模型状态加载到另一个新目录：

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=act_config.yaml \
  task=task3 \
  method=act \
  root=/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165 \
  repoid=task3_repaired_165 \
  training.resume=true \
  training.resume_timestamp=run_YYYYMMDD_HHMMSS
```

## 命令行 Open-loop

DP queue（按部署时的动作队列逐步取动作）：

```bash
python tools/open_loop_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345 \
  --repo-id task1_repaired_345 \
  --policy-type diffusion \
  --policy-path /path/to/dp/run_or_epoch \
  --mode queue \
  --device cuda
```

ACT chunk（直接比较动作块）：

```bash
python tools/open_loop_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165 \
  --repo-id task3_repaired_165 \
  --policy-type act \
  --policy-path /path/to/act/run_or_epoch \
  --mode chunk \
  --max-horizon 50 \
  --device cuda
```

工具不会把观测历史预先堆成额外的时间维；历史由策略自身的 causal queue
管理。DP chunk 的真实动作窗口会跳过观测历史偏移，只与实际可执行的
`n_action_steps` 对齐。输出目录包含 `summary.json`、首动作误差 CSV，以及
可选的 horizon/dimension 图。

## Viewer

```bash
streamlit run tools/open_loop_viewer.py
```

在侧栏选择数据目录、模型类型和检查点后运行。Viewer 同样由策略管理历史
队列，真实动作时间线直接读取 parquet。ROS 部署效果和宿主机相机/关节消息
仍需在目标机器上手工确认。

## 已验证基线

2026-07-25 在 CPU 上完成以下最小真实检查点验证：

- Task1 DP：根目录 queue 和 `epoch100` chunk；
- Task3 ACT：`epoch500` queue 和 chunk；
- Task3 ACT：从零训练 1 个 optimizer step，保存后原地续训到第 2 step。

这些是接口和存档契约冒烟，不替代 GPU 吞吐、长程精度或 ROS 真机验收。
