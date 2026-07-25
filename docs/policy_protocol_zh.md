# Kuavo 隔离策略协议与 OpenPI ROS 接入

统一策略客户端使用 MessagePack + WebSocket，不反序列化 pickle、Torch
对象或任意 Python 类。ROS 进程只需要：

```bash
python -m pip install -r requirements_policy_client.txt
```

模型和框架依赖继续留在独立 worker 环境。当前协议客户端兼容 OpenPI
`WebsocketPolicyServer`：

1. 建立 WebSocket 连接；
2. 接收第一帧 metadata；
3. 发送 observation mapping；
4. 接收包含 `actions[T,D]` 的结果。

OpenPI 的健康检查是 `GET /healthz`。其当前 policy reset 是 no-op，因此
ROS client 的 `reset()` 会清空本地动作队列、关闭连接、重新连接并刷新
metadata。后续 DP/ACT/LingBot worker 也必须实现相同可观察语义。

## 观测和动作契约

请求：

```text
observation.images.head_cam_h  CHW 或 HWC RGB
observation.images.wrist_cam_l / wrist_cam_r 至少一个
observation.state              float[D]
prompt                         非空字符串
```

响应：

```text
actions                        finite float[T,D]
```

客户端拒绝 object、structured 和 complex NumPy dtype，检查图像、状态和
动作维度及非有限值。`client_execute_steps` 决定一个动作块最多执行多少步
才重新推理；安全默认值为 1。

## 启动 OpenPI server

在 OpenPI 的 Python 3.11/JAX 环境：

```bash
scripts/kuavo_openpi serve \
  policy:checkpoint \
  --policy.config=pi05_kuavo \
  --policy.dir=/path/to/checkpoint \
  --port=8000
```

在 ROS/部署环境安装轻量客户端依赖，然后以
`configs/deploy/kuavo_env.openpi_client.yaml` 为起点：

```yaml
inference:
  policy_type: client
  client_protocol: msgpack_websocket
  client_host: 127.0.0.1
  client_port: 8000
  client_action_dim: 8
  client_state_dim: 8
  client_execute_steps: 1
  task_prompt: "任务描述"
```

API key 不写入 YAML。需要时只设置环境变量：

```bash
export KUAVO_POLICY_API_KEY="<private-key>"
```

容器运行时使用 host network 或显式端口映射，并把 `client_host` 指向模型
worker。模型权重、HF/W&B/ServerChan token 和 ROS 网络参数均应在运行时
挂载或注入，不能写入镜像层。

## 当前验证边界

已验证：

- Kuavo 与 OpenPI NumPy MessagePack 双向兼容；
- metadata 握手、infer、reset/reconnect 和 `/healthz`；
- 连接/请求超时和服务端文本错误；
- ROS adapter 的动作块队列与 prompt fallback；
- `policy_type=client` 配置加载和字段校验。

完整 JAX checkpoint server、ROS topic 和真机动作仍需在 CUDA/ROS 宿主机
手工验收。该验收前保持 `client_execute_steps=1`，确认动作语义和频率后再
增加执行步数。
