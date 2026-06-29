# VLA Open-loop 测试

这里的 open-loop 测试不需要 ROS、仿真器或真机。它从 LeRobot 数据集中读取某一帧的图像和 state，调用本地策略预测动作，再和数据集里的 GT action 做对比。目前支持 SmolVLA、LingBot、ACT 和 Diffusion Policy。

## LingBot 本地测试

LingBot 使用 `kdc_vla` 环境。先用很少的帧完成模型、数据和归一化链路检查：

```bash
conda activate kdc_vla
cd /home/larry/kuavo_data_challenge

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/open_loop_smolvla_eval.py \
  --policy-type lingbot \
  --policy-path /mnt/pqssd/lingbot_weights/clean_meanstd_fm_L2V2_mb16_gb16_8k_20260623_175250/checkpoints/global_step_8000/hf_ckpt \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ/lerobot_trimmed \
  --repo-id kuavo/task1_sz \
  --lingbot-root /home/larry/lingbot-vla \
  --qwen25-path /home/larry/Qwen2.5_VL \
  --norm-stats-file assets/norm_stats/lerobot_trimmed.json \
  --lingbot-data-type customized \
  --task-description "Pick and Place the safety belt, cable and pin connector" \
  --episodes 0 \
  --max-frames-per-episode 3 \
  --stride 50 \
  --max-horizon 50 \
  --mode chunk \
  --video-backend pyav \
  --device cuda \
  --output-dir outputs/open_loop/r1/lingbot/smoke
```

`chunk` 模式对每个采样观测独立预测动作块。通过后可增加 episode 和采样帧数。模拟部署的五步缓存执行使用：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python tools/open_loop_smolvla_eval.py \
  --policy-type lingbot \
  --policy-path /mnt/pqssd/lingbot_weights/clean_meanstd_fm_L2V2_mb16_gb16_8k_20260623_175250/checkpoints/global_step_8000/hf_ckpt \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ/lerobot_trimmed \
  --lingbot-root /home/larry/lingbot-vla \
  --qwen25-path /home/larry/Qwen2.5_VL \
  --norm-stats-file assets/norm_stats/lerobot_trimmed.json \
  --task-description "Pick and Place the safety belt, cable and pin connector" \
  --episodes 0 \
  --max-frames-per-episode 50 \
  --stride 1 \
  --max-horizon 1 \
  --lingbot-use-length 5 \
  --mode queue \
  --video-backend pyav \
  --device cuda \
  --output-dir outputs/open_loop/r1/lingbot/queue
```

交互 Viewer：

```bash
conda activate kdc_vla
streamlit run tools/open_loop_smolvla_viewer.py
```

在侧栏选择 `lingbot`。Viewer 的 `chunk` 模式展示完整预测动作块；`single` 模式只展示第一步。Viewer 不模拟跨帧缓存，部署式缓存请使用批量 evaluator 的 `queue` 模式。

## 推荐先跑的批量诊断

```bash
conda run -n kdc_dev python tools/open_loop_smolvla_eval.py \
  --dataset-root /mnt/pqssd/Real_PQ_3.0/TASK1_SZ/lerobot_trimmed \
  --policy-type smolvla \
  --policy-path outputs/train/r1/smolvla/run_20260625_212157_from10k_lrfix/checkpoints/040000/pretrained_model \
  --episodes 0 1 2 3 4 \
  --max-frames-per-episode 80 \
  --stride 5 \
  --max-horizon 50 \
  --device cuda \
  --output-dir outputs/open_loop/r1/smolvla/chunk_eval
```

输出文件：

- `summary.json`：整体 MAE/RMSE、每个 horizon 的 MAE、每个动作维度的 MAE、动作范围越界率。
- `first_action_errors.csv`：每个采样帧第一个动作的 pred/gt/error，适合后续排序找坏例子。
- `horizon_mae.png` 和 `dim_mae.png`：如果环境里有 matplotlib，会自动保存。

默认 `mode=chunk`，即每个采样帧都独立预测一次 50 步动作块，并和 LeRobot 根据 policy config 取出的未来 action chunk 对齐比较。这最适合检查模型本身是否学到数据分布。

批量脚本也支持 `--policy-type act` 和 `--policy-type diffusion`，需要同时把 `--policy-path` 指到对应 ACT/DP checkpoint。

如果想更接近部署时 `select_action` 的缓存行为，可以跑：

```bash
conda run -n kdc_dev python tools/open_loop_smolvla_eval.py \
  --episodes 0 1 2 \
  --max-frames-per-episode 200 \
  --stride 1 \
  --max-horizon 1 \
  --mode queue \
  --output-dir outputs/open_loop/r1/smolvla/queue_eval
```

## 交互式逐帧观察器

```bash
conda run -n kdc_dev streamlit run tools/open_loop_smolvla_viewer.py
```

打开页面后可以选择 `policy_type`、episode 和 frame。frame 是 episode 内滑条，页面顶部会显示当前 episode 进度，不会固定在首帧。

页面会展示：

- 当前视觉观测，包括 RGB 和 depth，可调整亮度、对比度、颜色。
- Crop 支持 `Center` 和 `Manual` 两种模式。`Manual` 使用归一化的 left/top/right/bottom 边界，可以手动框定任意区域。
- 第一帧动作的 pred/gt/error 表格。
- horizon MAE 和每个动作维度 MAE。
- 单个动作维度的 pred/gt 两条曲线，以及 error 曲线。

观察器现在支持 `smolvla`、`act`、`diffusion` 三种 `policy_type`。SmolVLA 默认使用动作块预测；ACT/DP 也会优先尝试 `predict_action_chunk`，如果选择 `single` 模式则走部署式 `select_action` 单步输出。

如果 policy 需要某个观测 key，但当前数据集样本没有这个 key，例如 depth policy 需要 `observation.depth_h`，页面会提前列出缺失 key。此时需要切到和 checkpoint 匹配的 LeRobot 数据集，或者使用未启用该模态的 checkpoint。

## 如何理解结果

SmolVLA 当前配置是：

- `n_obs_steps = 1`
- `chunk_size = 50`
- `n_action_steps = 10`
- `action_dim = 8`

训练数据是 10 FPS，所以 50 步动作块对应约 5 秒未来动作；实际部署时 `select_action` 每次预测 50 步，但默认只缓存/执行前 10 步，然后再重新预测。

训练集 open-loop 的误差通常会偏乐观，因为输入来自数据集真实轨迹，没有闭环误差累积，也没有动作执行后的状态偏移。它最适合做这些事情：

- 检查 checkpoint、tokenizer、pre/postprocessor 是否能在离线环境完整加载。
- 找出某些 episode、图像条件或动作维度是否明显异常。
- 对比图像增强或 crop 后动作是否剧烈漂移，估计视觉鲁棒性。
- 确认动作输出是否超出训练集动作范围。

它不能直接证明真机闭环成功；如果 open-loop 已经很差，闭环通常风险更高。如果训练集 open-loop 很好，但真机失败，重点应转向观测预处理、相机裁剪/颜色、动作坐标语义、控制频率和异步/RTC 调度。
