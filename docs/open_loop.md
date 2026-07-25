# 统一 Open-loop 评测

Open-loop 工具从 LeRobot 数据集中读取图像、状态和真实动作，在不启动
ROS、仿真器或真机的情况下运行策略，并输出动作误差、推理延迟和动作范围
诊断。

当前入口：

```bash
python tools/open_loop_eval.py --help
streamlit run tools/open_loop_viewer.py
```

本地评测支持：

- classic ACT；
- classic Diffusion Policy；
- LingBot-VLA v1；
- LingBot-VLA v2；
- OpenPI（通过隔离的 WebSocket Policy Server）。

示例：

```bash
python tools/open_loop_eval.py \
  --dataset-root /path/to/lerobot \
  --repo-id kuavo/task1 \
  --policy-type act \
  --policy-path /path/to/checkpoint \
  --episodes 0 1 2 \
  --mode chunk
```

LingBot-v2 还需要传入匹配的源码、processor、robot config 和归一化文件：

```bash
python tools/open_loop_eval.py \
  --dataset-root /path/to/lerobot \
  --repo-id kuavo/task1 \
  --policy-type lingbot_v2 \
  --policy-path /path/to/hf_ckpt \
  --lingbot-root third_party/lingbot-vla-v2 \
  --qwen25-path /path/to/Qwen3-VL \
  --robot-name kuavo_v2_right_arm \
  --norm-stats-file assets/norm_stats/kuavo_v2_right_arm_meanstd.json
```

OpenPI 示例（先在 OpenPI 容器或隔离环境启动 Server）：

```bash
python tools/open_loop_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345 \
  --repo-id kuavo/task1_sz \
  --policy-type openpi \
  --policy-endpoint 127.0.0.1:8000 \
  --episodes 0 \
  --max-frames-per-episode 3
```

Viewer 中选择 `openpi` 并填写 `host:port`。JAX 仍留在独立 Server 环境，
viewer 只使用轻量 MessagePack WebSocket Client。

SmolVLA、GR00T 和 EEF diffusion 不属于最终训练或交付范围，不再提供加载
选项。
