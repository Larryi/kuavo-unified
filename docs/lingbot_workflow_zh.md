# LingBot-VLA v1 / v2 训练与推理

LingBot v1、v2 分别使用固定子模块和隔离环境：

- v1：`third_party/lingbot-vla@4eb34b7`；
- v2：`third_party/lingbot-vla-v2@d34898d`，远端为
  `Larryi/lingbot-vla-v2`。

2026-07-25 核对官方 `Robbyant/lingbot-vla-v2` 后，官方 main 最新为
`69729b4`；Larryi fork 以该提交为祖先，并增加三个数据映射修复。官方当前
没有覆盖这三个修复，因此继续保留：

- 删除旧 dataset 限制；
- 正确使用 `target_start/target_end` 写入 55D canonical slot；
- 同时保留没有显式 target slot 的紧凑映射，并修复 float image 重复缩放。

## v1 训练

先用 dry-run 核对 torchrun、数据维度和路径：

```bash
HF_DATASETS_CACHE=/tmp/kuavo-lingbot-hf \
python kuavo_train/train_policy.py \
  --config-path=../configs/policy \
  --config-name=lingbot_config.yaml \
  policy.dry_run=true
```

确认 `/workspace/models/LingBotVLA` 和 `/workspace/models/Qwen2.5_VL`
存在后去掉 `policy.dry_run=true`。VastAI 完整流程继续使用
`scripts/run_task1_lingbot_full_pipeline.sh`；它会计算 Task1 norm stats，
训练并把 `norm_stats.json` 与 `lingbotvla_cli.yaml` 放进最终 `hf_ckpt`。

## v2 训练

```bash
python kuavo_train/train_lingbot_v2.py policy.dry_run=true
python kuavo_train/train_lingbot_v2.py
```

默认是 Task1 repaired 345 episodes、`kuavo_v2_right_arm`、LoRA 和单 GPU。
训练需要基础 LingBot-VLA-v2、Qwen3-VL，以及配置启用的
MoGe/LingBot-Depth/DINO-video teacher；部署只需要合并后的 `hf_ckpt`、
Qwen3-VL processor/tokenizer、robot config 和匹配 norm stats。

## 推理和 Open-loop

v1 适配器使用当前官方入口 `deploy.lingbot_vla_policy.LingbotVLAServer`，
不再依赖已删除的 `deploy.lingbot_robotwin_policy`。v1 与 v2 的
`FeatureTransform` schema 不完全相同，因此 v1 使用独立的
`kuavo_v1_right_arm`，不能直接复用带 55D target slot 的 v2 YAML。
模型目录中应携带 `norm_stats.json`；也可显式传
`lingbot_norm_stats_file`。

```bash
python tools/open_loop_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345 \
  --repo-id task1_repaired_345 \
  --policy-type lingbot \
  --policy-path /path/to/v1/hf_ckpt \
  --lingbot-root third_party/lingbot-vla \
  --qwen25-path /path/to/Qwen2.5_VL \
  --robot-name kuavo_v1_right_arm \
  --norm-stats-file /path/to/norm_stats.json \
  --device cuda
```

把 `--policy-type` 改为 `lingbot_v2`，并把 processor 路径改为 Qwen3-VL，
即可使用 v2。交互 Viewer：

```bash
streamlit run tools/open_loop_viewer.py
```

v1/v2 的 Python 包名和 `deploy` 包名冲突，不能在同一进程轮流加载；应在
各自容器或 worker 进程中运行。当前无 GPU 沙箱只验证了命令组合、payload、
slot 映射与导入，完整权重加载和 ROS 消息推理需在 CUDA 宿主机手工验收。
