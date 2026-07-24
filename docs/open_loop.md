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
- LingBot-VLA v2。

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

不同模型依赖不应被强行安装进同一个 Python 环境。当前工具仍可在对应模型
环境内本地加载策略；建立 `kuavo_policy_protocol` 后，统一入口应优先通过
隔离的 policy worker 调用 ACT、DP、LingBot 和 OpenPI。OpenPI 支持将在该
协议建立后接入，而不是在此工具中直接导入 JAX。

SmolVLA、GR00T 和 EEF diffusion 不属于最终训练或交付范围，不再提供加载
选项。
