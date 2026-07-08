#!/usr/bin/env python3
"""
mock_env.py — 本地 bag 回放测试用的全量 Mock 节点

作用:模拟赛事方真机环境中由机器人控制栈提供的:
  1. ROS Parameter Server 参数 (robot_version, *RealDof 等)
  2. 裁判服务 (/simulator/reset, /simulator/start)
  3. /simulator/init 的周期性触发 (auto_test 等待它来开始每个 episode)
  4. 占位节点 /humanoid_gait_switch_by_name (KuavoRealEnv reset 时检查它是否存活)

启动顺序:
  1. roscore                        (宿主机, ROS_MASTER_URI=http://127.0.0.1:11311)
  2. rosbag play --loop your.bag    (宿主机, 只放观测话题)
  3. python mock_env.py             (宿主机, 本脚本)         ← 先于 auto_test 起
  4. 容器内: policy server
  5. 容器内: script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml

注意: 本脚本与容器必须共享同一个 ROS master (--net=host + 同一个 ROS_MASTER_URI)
"""

import rospy
import threading
import time
import multiprocessing
import json
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, JointState
from std_srvs.srv import Trigger, TriggerResponse
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawState, sensorsData
from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (
    changeArmCtrlMode,
    changeArmCtrlModeResponse,
    getCurrentGaitName,
    getCurrentGaitNameResponse,
)


# ─────────────────────────────────────────────
# 1. 配置区 — 按需修改
# ─────────────────────────────────────────────

# Kuavo 4 Pro 机器人参数 (来自官方文档默认值)
ROBOT_PARAMS = {
    "/robot_version": 47,
    "/armRealDof":     14,
    "/legRealDof":    12,
    "/headRealDof":    2,
    "/waistRealDof":   0,
    "/robot_type":      1,  # Wheel-arm mode skips irrelevant MPC waits in the SDK.
    "/end_effector_type": "lejuclaw",
    "/com_height":      0.8328437523948975,
    "/enable_manipulation_mpc": False,
    "/use_shm_communication": False,
}

# mock /simulator/init 的触发间隔 (秒)
# auto_test 在每个 episode 开始前等待这个触发,1s 已足够
INIT_POKE_INTERVAL = 1.0

OBS_STATE_HZ = 250.0
OBS_IMAGE_HZ = 30.0

RGB_TOPICS = (
    "/cam_h/color/image_raw/compressed",
    "/cam_l/color/image_raw/compressed",
    "/cam_r/color/image_raw/compressed",
)
DEPTH_TOPICS = (
    "/cam_h/depth/image_raw/compressedDepth",
    "/cam_l/depth/image_rect_raw/compressedDepth",
    "/cam_r/depth/image_rect_raw/compressedDepth",
)
COMMAND_TOPICS = (
    "/kuavo_arm_traj",
    "/robot_head_motion_data",
    "/cmd_vel",
    "/kuavo_arm_target_poses",
    "/cmd_pose",
    "/humanoid_mpc_foot_pose_target_trajectories",
    "/humanoid_switch_gait_by_name",
    "/cmd_pose_world",
    "/leju_claw_command",
)


def make_mock_urdf():
    link_names = [
        *(f"leg_l{i}_link" for i in range(1, 7)),
        *(f"leg_r{i}_link" for i in range(1, 7)),
        *(f"zarm_l{i}_link" for i in range(1, 8)),
        *(f"zarm_r{i}_link" for i in range(1, 8)),
        "zhead_1_link",
        "zhead_2_link",
    ]
    parts = ['<robot name="kuavo_mock"><link name="base_link"/>']
    parent = "base_link"
    for link_name in link_names:
        joint_name = link_name.replace("_link", "_joint")
        parts.append(f'<link name="{link_name}"/>')
        parts.append(
            f'<joint name="{joint_name}" type="revolute">'
            f'<parent link="{parent}"/><child link="{link_name}"/>'
            '<limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>'
            '</joint>'
        )
        parent = link_name
    parts.append("</robot>")
    return "".join(parts)


ROBOT_PARAMS.update(
    {
        "/humanoid_description": make_mock_urdf(),
        "/kuavo_configuration": json.dumps(
            {
                "end_frames_names": [
                    "torso",
                    "zarm_l7_link",
                    "zarm_r7_link",
                    "zarm_l4_link",
                    "zarm_r4_link",
                ]
            }
        ),
    }
)


# ─────────────────────────────────────────────
# 2. 占位节点 — 在子进程中运行
#    必须独立进程: 一个进程只能 init_node 一次
# ─────────────────────────────────────────────

def _gait_switch_placeholder():
    """
    子进程: 起一个名为 /humanoid_gait_switch_by_name 的空节点.
    KuavoRealEnv.reset() 用 /rosnode/list 检查它是否存在,
    名字必须精确匹配、anonymous=False.
    """
    import rospy
    rospy.init_node("humanoid_gait_switch_by_name", anonymous=False)
    rospy.loginfo("[mock] /humanoid_gait_switch_by_name placeholder alive")
    rospy.spin()


# ─────────────────────────────────────────────
# 3. 主节点 — 裁判服务 + 参数 + init 触发
# ─────────────────────────────────────────────

def reset_handler(req):
    rospy.loginfo("[mock] /simulator/reset called")
    return TriggerResponse(success=True, message="reset ok")


def start_handler(req):
    rospy.loginfo("[mock] /simulator/start called")
    return TriggerResponse(success=True, message="start ok")


_arm_ctrl_mode = 2


def arm_ctrl_mode_handler(req):
    global _arm_ctrl_mode
    if getattr(req, "control_mode", 0) in (0, 1, 2):
        _arm_ctrl_mode = req.control_mode
    return changeArmCtrlModeResponse(result=True, mode=_arm_ctrl_mode, message="mock arm mode")


def gait_name_handler(_req):
    return getCurrentGaitNameResponse(success=True, gait_name="stance")


class SyntheticObservationPublisher:
    def __init__(self):
        self.sensor_pub = rospy.Publisher("/sensors_data_raw", sensorsData, queue_size=10)
        self.gripper_pub = rospy.Publisher("/leju_claw_state", lejuClawState, queue_size=10)
        self.rgb_pubs = [rospy.Publisher(topic, CompressedImage, queue_size=2) for topic in RGB_TOPICS]
        self.depth_pubs = [rospy.Publisher(topic, CompressedImage, queue_size=2) for topic in DEPTH_TOPICS]

        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
        rgb[..., 1] = np.arange(48, dtype=np.uint8)[:, None] * 5
        rgb[..., 2] = 96
        ok, encoded_rgb = cv2.imencode(".jpg", rgb)
        if not ok:
            raise RuntimeError("Failed to encode synthetic RGB image")
        self.rgb_data = encoded_rgb.tobytes()

        depth = np.full((48, 64), 800, dtype=np.uint16)
        ok, encoded_depth = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError("Failed to encode synthetic depth image")
        self.depth_data = encoded_depth.tobytes()

        self.state_timer = rospy.Timer(rospy.Duration(1.0 / OBS_STATE_HZ), self.publish_state)
        self.image_timer = rospy.Timer(rospy.Duration(1.0 / OBS_IMAGE_HZ), self.publish_images)

    def publish_state(self, _event):
        stamp = rospy.Time.now()
        sensor = sensorsData()
        sensor.header.stamp = stamp
        sensor.sensor_time = stamp
        sensor.joint_data.joint_q = [0.0] * 28
        sensor.joint_data.joint_v = [0.0] * 28
        sensor.joint_data.joint_vd = [0.0] * 28
        sensor.joint_data.joint_torque = [0.0] * 28
        self.sensor_pub.publish(sensor)

        gripper = lejuClawState()
        gripper.header.stamp = stamp
        gripper.state = [0, 0]
        gripper.data.name = ["left_claw", "right_claw"]
        gripper.data.position = [0.0, 0.0]
        gripper.data.velocity = [0.0, 0.0]
        gripper.data.effort = [0.0, 0.0]
        self.gripper_pub.publish(gripper)

    def publish_images(self, _event):
        stamp = rospy.Time.now()
        for pub in self.rgb_pubs:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = self.rgb_data
            pub.publish(msg)
        for pub in self.depth_pubs:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "16UC1; compressedDepth png"
            msg.data = self.depth_data
            pub.publish(msg)


def install_command_sinks():
    return [rospy.Subscriber(topic, rospy.AnyMsg, lambda _msg: None) for topic in COMMAND_TOPICS]


def poke_init_loop():
    """
    周期性调用 /simulator/init (auto_test 自己 advertise 的服务).
    auto_test 在 while not init_evt.is_set() 里等待这个调用来触发每个 episode.
    """
    rospy.wait_for_service("/simulator/init")
    rospy.loginfo("[mock] /simulator/init service found, will poke it periodically")
    call_init = rospy.ServiceProxy("/simulator/init", Trigger)
    while not rospy.is_shutdown():
        try:
            call_init()
            rospy.logdebug("[mock] /simulator/init poked")
        except Exception as e:
            rospy.logwarn(f"[mock] /simulator/init poke failed: {e}")
        time.sleep(INIT_POKE_INTERVAL)


def set_robot_params():
    for name, value in ROBOT_PARAMS.items():
        rospy.set_param(name, value)
        display = f"<{len(value)} chars>" if isinstance(value, str) and len(value) > 120 else value
        rospy.loginfo(f"[mock] set_param {name} = {display}")


def main():
    # ── 先起占位节点子进程 ──────────────────────
    gait_proc = multiprocessing.Process(
        target=_gait_switch_placeholder, daemon=True)
    gait_proc.start()
    rospy.loginfo("[mock] gait_switch placeholder process started")

    # ── 主节点 (裁判 + 参数) ────────────────────
    rospy.init_node("mock_referee_env", anonymous=False)

    # 写入机器人参数到 Parameter Server
    set_robot_params()

    # 裁判服务 (auto_test 作为 client 来调用)
    rospy.Service("/simulator/reset", Trigger, reset_handler)
    rospy.Service("/simulator/start", Trigger, start_handler)
    rospy.Service("/humanoid_get_arm_ctrl_mode", changeArmCtrlMode, arm_ctrl_mode_handler)
    rospy.Service("/wheel_arm_change_arm_ctrl_mode", changeArmCtrlMode, arm_ctrl_mode_handler)
    rospy.Service("/change_arm_ctrl_mode", changeArmCtrlMode, arm_ctrl_mode_handler)
    rospy.Service("/humanoid_get_current_gait_name", getCurrentGaitName, gait_name_handler)
    rospy.loginfo("[mock] /simulator/reset and /simulator/start advertised")

    synthetic_observations = SyntheticObservationPublisher()
    command_sinks = install_command_sinks()
    rospy.loginfo(
        "[mock] synthetic RGB/depth/joint/gripper observations and control sinks enabled"
    )

    # 后台线程: 周期触发 /simulator/init
    poke_thread = threading.Thread(target=poke_init_loop, daemon=True)
    poke_thread.start()

    rospy.loginfo("[mock] mock_env ready — waiting for auto_test to connect")
    rospy.spin()

    del synthetic_observations, command_sinks

    gait_proc.terminate()


if __name__ == "__main__":
    # multiprocessing 在某些 Linux 环境下需要 spawn/fork 声明
    multiprocessing.set_start_method("fork")
    main()
