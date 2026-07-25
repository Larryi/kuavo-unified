# 训练、推理与容器交付验收顺序

本文只列最终交付门禁。先做无动作的离线检查，再连接 ROS；任何会发布
机器人动作的命令都由操作者在急停可用、工作空间清空后手工启动。

## 1. 共同预检

```bash
git submodule status --recursive
nvidia-smi
python -m py_compile \
  kuavo_deploy/src/eval/real_single_test.py \
  kuavo_deploy/src/eval/sim_auto_test.py
```

要求所有子模块提交与 `.gitmodules`/迁移计划一致，GPU 驱动可用，且
checkpoint、norm stats、数据集来自同一次训练/转换。

当前确认的数据集是：

- Task1：`/mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345`
- Task2：`/mnt/pqssd/Real_PQ_3.0/TASK2_SZ_Repaired/lerobot_task2_264`
- Task3：`/mnt/pqssd/Real_PQ_3.0/TASK3_SZ_Repaired/lerobot_task3_165`

三者均为 10 Hz，分别包含 345/264/165 episodes 和
81,142/50,042/26,987 frames。Task1、Task3 的 state/action 为 8 维，
使用头部加右腕相机；Task2 为 16 维并使用头部、左腕、右腕三路相机。
Task1 是右 Leju 夹爪，Task2 是左右 Leju 夹爪，Task3 是右 Qiangnao
末端。模型配置必须同时对应维度、相机和末端类型，不能只替换数据路径。

## 2. 无 ROS open-loop

DP/ACT 使用 [dp_act_workflow_zh.md](dp_act_workflow_zh.md) 的命令；
LingBot-v1 使用 [lingbot_workflow_zh.md](lingbot_workflow_zh.md)；
OpenPI 使用 [openpi_workflow_zh.md](openpi_workflow_zh.md)。每种模型先只跑
1 个 episode、3 个 sample，确认：

- observation key、state/action 维度与训练配置一致；
- 输出全部有限，关节/夹爪不越训练数据范围；
- chunk horizon、首动作与 queue 模式时间对齐；
- viewer 中原始动作和限速/边界平滑后的动作均可解释。

DP/ACT 是基础门禁，但优先级低于最终交付的 OpenPI 和 LingBot-v1。
LingBot-v2 自 2026-07-25 起由操作者明确暂缓，不进入当前验收序列。

## 3. 隔离 worker 与协议

按 [docker_policy_workers_zh.md](docker_policy_workers_zh.md) 启动对应 worker，
先不要启动机器人控制：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

如启用 `KUAVO_POLICY_API_KEY`：

```bash
curl --fail \
  -H "Authorization: Api-Key ${KUAVO_POLICY_API_KEY}" \
  http://127.0.0.1:8000/healthz
```

然后在 ROS 环境中使用
`configs/deploy/kuavo_env.openpi_client.yaml` 作为所有远程 worker 的共同
起点。首次验收保持 `client_execute_steps: 1`。

## 4. ROS 只读观测

启动 ROS master、相机和关节状态后，先确认输入，不发布动作：

```bash
rostopic hz /cam_h/color/image_raw/compressed
rostopic hz /cam_r/color/image_raw/compressed
rostopic hz /sensors_data_raw
rostopic hz /leju_claw_state
```

频率、图像尺寸和 topic 类型必须匹配 deploy YAML。确认 policy worker
日志已收到 observation 且返回合法 action chunk 后，才进入动作测试。

## 5. 真机动作测试

在宿主机 ROS 环境中，由操作者手工运行：

```bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_env.openpi_client.yaml
```

第一轮只执行一个 episode，并保持急停。依次验收 OpenPI、LingBot-v1；
DP/ACT 可复用同一 client YAML。每次只启动一个 worker，
核对 10 Hz 控制频率、动作维度、左右臂/夹爪 slot、暂停和停止信号。

## 6. Docker 与 VastAI

完整镜像 build 后先执行容器内 import、`nvidia-smi` 和单样本 open-loop，
再连接宿主机 ROS。VastAI 流程先执行：

当前已完成的无权重容器门禁：

- `kuavo-classic:latest`：Ubuntu 20.04/ROS Noetic，Torch
  2.7.1+cu126 识别 RTX 3090；
- `kdc_real_task1_lingbot:latest`：flash-attn 2.7.0.post2 和
  LingBot-v1 adapter 导入通过，Torch 识别 RTX 3090；
- `kuavo-openpi:latest`：单一 Ubuntu 20.04.6/ROS Noetic 镜像，
  Classic Python 3.10 ROS Client 与 OpenPI Python 3.11/JAX Server
  同时导入通过，JAX 识别 `cuda:0`。

上述结果不替代真实 checkpoint、单样本 open-loop 和 ROS 动作验收。

当前真实权重结果：

- OpenPI Pi0.5：ROS mock 与 Task1 3-sample open-loop 已通过；
- LingBot-v1：checkpoint/GPU/ROS mock 接口可运行，但 Task1 3-sample
  open-loop 的关节 MAE 为 0.757 rad，动作范围越界率为 60.75%，当前
  判定动作质量门禁失败，禁止进入真机动作验收；
- LingBot-v2：按操作者决定暂缓，不进入当前交付完成度。

```bash
MODEL_BACKEND=openpi DRY_RUN=1 scripts/vast/launch.sh
MODEL_BACKEND=lingbot-v1 DRY_RUN=1 scripts/vast/launch.sh
```

LingBot-v2 云端、镜像和推理验收均暂缓；统一入口保留其已有路由，但当前
不应将其运行结果计入交付完成度。

真实云端运行前逐项确认代码同步目标、数据集仓库、预训练权重清单和挂载
位置，以及 HF/W&B/ServerChan/Vast token 仅存在于权限 600 的私有 env
文件。验收完成后检查 ServerChan 通知和 Vast 实例自动关机；失败路径也
必须通知，但默认不自动销毁实例，保留现场供排障。
