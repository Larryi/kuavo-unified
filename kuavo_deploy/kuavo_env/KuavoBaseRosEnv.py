#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import gymnasium as gym
import time
import sys
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, DexterousHand
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawCommand
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.ros_manager import ROSManager
import traceback
import torch
from torchvision.transforms.functional import to_tensor
from kuavo_deploy.utils.obs_buffer import ObsBuffer
from kuavo_deploy.utils.eef_commands import normalized_leju_claw_positions
from kuavo_deploy.utils.signal_controller import ControlSignalManager


log_robot = setup_logger("robot")


class KuavoBaseRosEnv(gym.Env):
    """Kuavo robot ROS environment base class"""

    def __init__(self, config: KuavoConfig):
        self._set_config(config.env)
        
        #Initialize ROS manager Initialize ROS manager
        self.ros_manager = ROSManager()
        self.control_signal_manager = ControlSignalManager()
        
        #Initialize other components Initialize other components
        self.bridge = CvBridge()
        self._set_observation_space()
        self._set_action_space()
        self._init_kuavo_sdk()
        self._set_ros_topics()
        
        #Wait for ROS topics to initialize Wait for ROS topics to initialise
        log_robot.info(f"Inializing done!")
        print(f"Inializing done!")

    def _set_config(self, config_kuavo_env):
        """Set configuration parameters Set configuration parameters"""
        self.ros_rate = config_kuavo_env.ros_rate
        self.control_mode = config_kuavo_env.control_mode
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.only_arm = config_kuavo_env.only_arm
        self.eef_type = config_kuavo_env.eef_type
        self.which_arm = config_kuavo_env.which_arm
        self.qiangnao_dof_needed = config_kuavo_env.qiangnao_dof_needed

        self.is_binary = config_kuavo_env.is_binary
        self.head_init = config_kuavo_env.head_init
        self.arm_init = np.array([0]*14)


        #Obtain limit values from the configuration
        self.limits = config_kuavo_env.limits
        self.obs_key_map = config_kuavo_env.obs_key_map
        self.obs_buffer = ObsBuffer(
            config=config_kuavo_env, 
            obs_key_map=self.obs_key_map,
        )
        self.arm_state_keys = config_kuavo_env.arm_state_keys # observation.state key ordering
        self.ratio = config_kuavo_env.ratio
        self.frame_alignment = config_kuavo_env.frame_alignment

    def _set_observation_space(self):
        limits = self.limits
        obs_low, obs_high = [], []

        # -------- State space (joint_q + gripper) --------
        if 'joint_q' in self.obs_key_map:
            joint_min, joint_max = limits['joint_q']['min'], limits['joint_q']['max']
        else:
            joint_min, joint_max = [], []
        if 'gripper' in self.obs_key_map:
            grip_min, grip_max = limits['gripper']['min'], limits['gripper']['max']
        else:
            grip_min, grip_max = [], []
        if self.which_arm == 'both':
            obs_low.extend(joint_min[:7]+grip_min[:1]+joint_min[7:14]+grip_min[1:2])
            obs_high.extend(joint_max[:7]+grip_max[:1]+joint_max[7:14]+grip_max[1:2])
        if self.which_arm == 'left':
            obs_low.extend(joint_min[:7]+grip_min[:1])
            obs_high.extend(joint_max[:7]+grip_max[:1])
        if self.which_arm == 'right':
            obs_low.extend(joint_min[7:14]+grip_min[1:2])
            obs_high.extend(joint_max[7:14]+grip_max[1:2])

        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)

        # -------- Image space --------
        obs_spaces = {}
        for key, obs_name in self.obs_key_map.items():
            if any(tag in key for tag in ['cam', 'depth']):
                h, w = obs_name["handle"]["params"]["resize_wh"]
                if 'depth' in key:
                    low, high = obs_name['handle']['params']['depth_range']
                    obs_spaces[f"observation.{key}"] = gym.spaces.Box(
                        low=low, high=high, shape=(1, h, w), dtype=np.uint16
                    )
                else:  # cam keys
                    obs_spaces[f"observation.images.{key}"] = gym.spaces.Box(
                        low=0, high=255, shape=(3, h, w), dtype=np.uint8
                    )

        # -------- Adding state space --------
        obs_spaces["observation.state"] = gym.spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32,
            shape=(len(self.obs_low),)
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _set_action_space(self):
        limits = self.limits

        # ===============================
        #Aux function: Constructing single arm operating range Aux function: Constructing single arm operating range
        # ===============================
        def get_arm_action_range(arm: str):
            """(low, high)"""
            if self.control_mode == 'joint':
                if arm == 'left':
                    return (
                        limits['joint_q']['min'][:7] + limits['gripper']['min'][:1],
                        limits['joint_q']['max'][:7] + limits['gripper']['max'][:1],
                    )
                elif arm == 'right':
                    return (
                        limits['joint_q']['min'][7:14] + limits['gripper']['min'][1:2],
                        limits['joint_q']['max'][7:14] + limits['gripper']['max'][1:2],
                    )
                elif arm == 'both':
                    return (
                        limits['joint_q']['min'][:7] + limits['gripper']['min'][:1]+limits['joint_q']['min'][7:14] + limits['gripper']['min'][1:2],
                        limits['joint_q']['max'][:7] + limits['gripper']['max'][:1]+limits['joint_q']['max'][7:14] + limits['gripper']['max'][1:2],
                    )

            elif self.control_mode == 'eef':
                # key = 'eef_relative' if self.use_delta else 'eef'
                key = 'eef'
                inf_pad = [-np.inf] * 6
                inf_pad_pos = [np.inf] * 6

                def eef_block(start, end, grip_idx):
                    low = limits[key]['min'][start:end] + inf_pad + limits['gripper']['min'][grip_idx:grip_idx + 1]
                    high = limits[key]['max'][start:end] + inf_pad_pos + limits['gripper']['max'][grip_idx:grip_idx + 1]
                    return low, high

                if arm == 'left':
                    return eef_block(0, 3, 0)
                elif arm == 'right':
                    return eef_block(6, 9, 1)
                elif arm == 'both':
                    low1, high1 = eef_block(0, 3, 0)
                    low2, high2 = eef_block(6, 9, 1)
                    return low1 + low2, high1 + high2

            raise ValueError(f"Unsupported arm mode: {arm}")

        # ===============================
        #Get arm motion range
        # ===============================
        arm_low, arm_high = get_arm_action_range(self.which_arm)

        # ===============================
        #If a base control is included, splice the base range
        # ===============================
        if not self.only_arm:
            base_low, base_high = limits['base']['min'], limits['base']['max']
            arm_low += base_low
            arm_high += base_high

        # ===============================
        #Create a Gym Box space
        # ===============================
        self.action_space = gym.spaces.Box(
            low=np.array(arm_low, dtype=np.float64),
            high=np.array(arm_high, dtype=np.float64),
            dtype=np.float64,
        )

    def _init_kuavo_sdk(self):
        """Initialise Kuavo SDK"""
        if not KuavoSDK().Init():
            log_robot.error("Init KuavoSDK failed, exit!")
            sys.exit(1)
        self.robot = KuavoRobot()
        self.robot_state = KuavoRobotState()

    def _set_ros_topics(self):
        """Setup ROS Topics Setup ROS Topics"""
        self.rate = rospy.Rate(self.ros_rate)
        
        if self.eef_type == 'rq2f85':
            self.pub_eef_joint = self.ros_manager.register_publisher('/gripper/command', JointState, queue_size=10)
        elif self.eef_type == 'leju_claw':
            self.lejuclaw = LejuClaw()
        elif self.eef_type == 'qiangnao':
            self.qiangnao = DexterousHand()
        #obs buffer initialization
        self.obs_buffer.wait_buffer_ready()


    def reset(self, **kwargs):
        """Reset Robot state Reset Robot state"""
        self._enter_external_control_mode()
        self._reset_head()
        self._reset_eef()

        #=== Average current observation and pose ===
        avg_data = self._compute_average_state(average_num=10)

        #=== Update status ===
        self.cur_state = avg_data["state"]
        self.cur_joint_angles_action = avg_data["joint_action"]

        obs = self.get_obs()
        self.sleep_time = 0
        self.average_sleep_time = 0
        return obs, {}

    # ==========================================================
    #Subfunction 1. External control mode setting
    # ==========================================================
    def _enter_external_control_mode(self):
        self.robot.set_external_control_arm_mode()
        print("set_external_control_arm_mode", self.robot_state.arm_control_mode())

    # ==========================================================
    #Subfunction 2. Head reset
    # ==========================================================
    def _reset_head(self):
        if self.head_init is not None:
            self.robot.control_head(self.head_init[0], self.head_init[1])

    # ==========================================================
    #Subfunction 3. End effector (gripper) reset
    # ==========================================================
    def _reset_eef(self):
        if self.which_arm == 'both':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control(target_positions=[0, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control(target_positions=[0, 0], target_velocities=None, target_torques=None)
        elif self.which_arm == 'left':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_left(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_left(target_positions=[0], target_velocities=None, target_torques=None)
        elif self.which_arm == 'right':
            if self.eef_type == 'qiangnao':
                self.qiangnao.control_right(target_positions=[0, 100, 0, 0, 0, 0], target_velocities=None, target_torques=None)
            elif self.eef_type == 'leju_claw':
                self.lejuclaw.control_right(target_positions=[0], target_velocities=None, target_torques=None)
        else:
            raise KeyError(f"Unsupported arm type: {self.which_arm}")

    # ==========================================================
    #Subfunction 4. Find the average state and terminal attitude
    # ==========================================================
    def _compute_average_state(self, average_num=10):
        state_sum, joint_sum = None, None

        for i in range(average_num):
            state = self.get_obs()
            fk_joint_angles = self._get_init_joint_angles(self.arm_state["joint_q"])

            if i == 0:
                state_sum = np.array(state["observation.state"], dtype=float)
                joint_sum = np.array(fk_joint_angles, dtype=float)
            else:
                state_sum += np.array(state["observation.state"], dtype=float)
                joint_sum += fk_joint_angles
            time.sleep(0.001)

        #average calculation
        avg_state = state_sum / average_num
        avg_joint = joint_sum / average_num

        return {
            "state": avg_state,
            "joint_action": avg_joint,
        }

    # ==========================================================
    #Subfunction 5. Construct the complete FK input
    # ==========================================================
    def _get_init_joint_angles(self, joint_q):
        # if self.which_arm == 'both':
        #     if self.fk_joint_angles_for_reset is not None:
        #         fk_joint_angles = np.array(self.fk_joint_angles_for_reset) / 180 * np.pi
        #     else:
        #         fk_joint_angles = np.array([-10, 15, 25, -85, -90, 15, -20,   50, 0, 0, -140, 90, 0, 0])/180*np.pi
        #     return fk_joint_angles
        if self.which_arm == 'both':
            return np.array(joint_q)
        elif self.which_arm == 'left':
            return np.concatenate((joint_q, self.arm_init[7:14]))
        elif self.which_arm == 'right':
            return np.concatenate((self.arm_init[:7], joint_q))
        else:
            raise ValueError(f"Invalid which_arm: {self.which_arm}")
    
    def check_action(self, action, mode='default'):
        if mode == 'default':  #compare action_space
            expected_dim = len(self.action_space.low)
            if len(action) != expected_dim:
                action = self._coerce_action_dim(action, expected_dim)
            if len(action) != len(self.action_space.low):
                raise ValueError(f"action shape must be {len(self.action_space.low)}")
            if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
                log_robot.warning(
                    f"action out of range, action: {action}, "
                    f"low: {self.action_space.low}, high: {self.action_space.high}"
                )
                action = np.clip(action, self.action_space.low, self.action_space.high)
            return action

        raise ValueError(f"Unsupported mode: {mode}")

    def _coerce_action_dim(self, action, expected_dim):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        current_gripper = np.asarray(self.arm_state.get("gripper", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if current_gripper.size < 2:
            current_gripper = np.pad(current_gripper, (0, 2 - current_gripper.size), constant_values=0.0)

        if self.control_mode == "joint" and expected_dim == 16 and action.size == 14:
            coerced = np.concatenate((action[:7], current_gripper[:1], action[7:14], current_gripper[1:2]), axis=0)
            log_robot.warning(
                f"Coerced 14-dim joint action to 16-dim by inserting current gripper states: {current_gripper.tolist()}"
            )
            return coerced

        if self.control_mode == "joint" and expected_dim == 8 and action.size == 14:
            if self.which_arm == "left":
                coerced = np.concatenate((action[:7], current_gripper[:1]), axis=0)
            elif self.which_arm == "right":
                coerced = np.concatenate((action[7:14], current_gripper[1:2]), axis=0)
            else:
                coerced = action
            if coerced.size == expected_dim:
                log_robot.warning(
                    f"Coerced 14-dim joint action to {expected_dim}-dim for {self.which_arm} arm by selecting arm joints and current gripper state."
                )
                return coerced

        return action



    def step(self, action):
        t0 = time.time()
        log_robot.info(f"action: {action}")
        # check clip action in action space
        action = self.check_action(action, mode='default')
        t1 = time.time()
        log_robot.info(f"clip action: {action}, check time: {t1 - t0:.3f}s")

        if not self.only_arm:
            #Get the base movement-related part of the action (the last 4 values). The last bit is used to determine whether it is movement or hand action.
            base_action = action[-4:]
            move_flag = base_action[-1]  #A value between 0-1, used to determine whether to perform base movement
            if move_flag > 0.5:  #If greater than 0.5, perform base movement
                #To perform base movement, you need to call the relevant base movement interface.
                log_robot.info(f"➡️ Execute [Chassis Movement] (mode_flag > 0.5)")
                log_robot.info(f"cmd_pos_world = [x:{base_action[0]:.4f}, y:{base_action[1]:.4f}, yaw:{base_action[2]:.4f}], move_flag={move_flag:.4f}")
                self.robot.control_command_pose_world(base_action[0], base_action[1], 0, base_action[2])
                self.rate.sleep()
                self._record_sleep_time(t1)
                return self.get_obs(), 0, False, False, {}
            else:
                log_robot.info(f"Whether the robot is standing: {self.robot._kuavo_core.state}, (mode_flag < 0.5)")
                if self.robot._kuavo_core.state != 'stance':
                    self.robot.stance()
                    self.robot_state.wait_for_stance()
                    log_robot.info(f"➡️ Successfully executed [Robot Stand]! (mode_flag < 0.5)")
            #If the base movement is not performed, the hand movement is performed, and the front part of the action is used at this time.
            action = action[:-4]


        #=== 4. Execute action ===
        t2 = time.time()
        self.cur_joint_angles_action = np.concatenate((action[:7], action[8:15]), axis=0)
        self.exec_action(action)

        #=== 5. Delay and observation ===
        
        self.rate.sleep()
        self._record_sleep_time(t2)
        t3 = time.time()
        obs = self.get_obs()
        t4 = time.time()
        log_robot.info(f"get obs time: {t4 - t3:.3f}s")

        #=== 6. Rewards and Returns ===
        reward = self.compute_reward()
        return obs, reward, False, False, {}
    
    
    def _record_sleep_time(self, t_start):
        self.sleep_time = time.time() - t_start
        self.average_sleep_time += self.sleep_time
        log_robot.info(f"rate.sleep time: {self.sleep_time:.3f}s")

    def exec_action(self, action):
        """Execute arm and end-effector motion Execute arm and end-effector motion"""
        # if not self.only_arm:
        #     return

        def safe_control_arm(target_position):
            try:
                self.robot.control_arm_joint_positions(target_position)
            except RuntimeError as e:
                #Unable to control arms when robot is in command_pose_world state (chassis moving)
                if "must be in stance state" in str(e):
                    log_robot.warning(f"⚠️  Cannot send arm commands: Robot's current state does not allow such operation (possibly robot is not in stance state)")
                    log_robot.debug(f"   Details: {e}")
                else:
                    raise


        if self.which_arm == 'both':
            left_joints, left_eef = action[:7], action[7]
            right_joints, right_eef = action[8:15], action[15]
            target_position = np.concatenate((left_joints, right_joints), axis=0)
            safe_control_arm(target_position)
            self._control_eef(left_eef, right_eef)

        elif self.which_arm == 'left':
            left_joints, left_eef = action[:7], action[7]
            target_position = np.concatenate((left_joints, self.arm_init[7:14]), axis=0)
            safe_control_arm(target_position)
            self._control_eef(left_eef, 0)

        elif self.which_arm == 'right':
            right_joints, right_eef = action[:7], action[7]
            target_position = np.concatenate((self.arm_init[:7], right_joints), axis=0)
            safe_control_arm(target_position)
            self._control_eef(0, right_eef)
        else:
            raise KeyError(f"Unsupported which_arm: {self.which_arm}")



    def _control_eef(self, left_eef, right_eef):
        """Choose end-effector based on eef_type"""
        if self.eef_type == 'rq2f85':
            eef_msg = JointState()
            try:
                eef_msg.name = ['left_gripper_joint', 'right_gripper_joint']
            except Exception as e:
                log_robot.info(f"_control_eef error! {e}")
            eef_msg.position = np.array([left_eef * 255, right_eef * 255])
            self.pub_eef_joint.publish(eef_msg)

        elif self.eef_type == 'leju_claw':
            eef_msg = JointState()
            eef_msg.position = normalized_leju_claw_positions(left_eef, right_eef)
            self.lejuclaw.control(target_positions=eef_msg.position)

        elif self.eef_type == 'qiangnao':
            if self.qiangnao_dof_needed != 1:
                raise KeyError("qiangnao_dof_needed != 1 is not supported!")

            tem_left, tem_right = left_eef * 100, right_eef * 100
            target_positions = np.array([
                tem_left, 100, *([tem_left] * 4),
                tem_right, 100, *([tem_right] * 4)
            ])
            self.qiangnao.control(target_positions=target_positions)

        else:
            raise KeyError(f"Unsupported eef_type: {self.eef_type}")

    def compute_reward(self):
        """Compute reward Compute reward"""
        return 0

    def get_obs(self):
        """Obtain observation image and state Obtain observation image and state"""
        obs = {}
        self.arm_state = {}

        if self.frame_alignment:
            obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=1/self.ros_rate,ratio=self.ratio)
            if obs_from_buffer is None or not all(v is not None for v in obs_from_buffer.values()):
                obs_from_buffer = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=float('inf'),ratio=self.ratio)
        else:
            obs_from_buffer = self.obs_buffer.get_latest_obs()
        
        for k,v in obs_from_buffer.items():
            # remap key
            if 'depth' in k:
                obs[f"observation.{k}"] = v
            elif 'cam' in k:
                obs[f"observation.images.{k}"] = v
            else:
                self.arm_state[f"{k}"] = v

        if self.is_binary:
            self.arm_state['gripper'] = np.where(self.arm_state['gripper']>0.5, 1, 0)

        assert len(self.arm_state.keys()) >= 2, f"arm_state must have exactly 2 elements, but got {len(self.arm_state.keys())}"

        state_keys = [k for k in self.arm_state_keys if k in self.arm_state]

        arm_data = { "left": [], "right": [] }

        for key in state_keys:
            data = self.arm_state[key]
            if len(data) == 0:
                continue
            mid = len(data) // 2
            if self.which_arm == "both":
                arm_data["left"].append(data[:mid])
                arm_data["right"].append(data[mid:])
            elif self.which_arm == "left":
                arm_data["left"].append(data)
            elif self.which_arm == "right":
                arm_data["right"].append(data)
            else:
                raise KeyError(f"Unsupported which_arm: {self.which_arm}")

        #Splicing result
        obs["observation.state"] = np.concatenate(
            arm_data["left"] + arm_data["right"], axis=0
        )
        log_robot.info(f"STATE: contained {state_keys}, concated value: {obs['observation.state']}")

        obs["observation.state"] = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0)
        return obs    

    def close(self):
        """Close the environment and release resources Closing environment"""
        log_robot.info("Closing KuavoBaseRosEnv...")
        try:
            if hasattr(self, 'obs_buffer'):
                self.obs_buffer.stop_subscribers()
                if hasattr(self.obs_buffer, 'obs_buffer_data'):
                    for k in self.obs_buffer.obs_buffer_data:
                        self.obs_buffer.obs_buffer_data[k]["data"].clear()
                        self.obs_buffer.obs_buffer_data[k]["timestamp"].clear()
                if hasattr(self.obs_buffer, 'ros_manager'):
                    self.obs_buffer.ros_manager = None
                if hasattr(self.obs_buffer, 'control_signal_manager'):
                    self.obs_buffer.control_signal_manager = None
                del self.obs_buffer
            
            if hasattr(self, 'ros_manager'):
                self.ros_manager.close()
            if hasattr(self, 'control_signal_manager'):
                self.control_signal_manager.close()
            log_robot.info("KuavoBaseRosEnv closed successfully.")
        except Exception as e:
            log_robot.error(f"Error closing KuavoBaseRosEnv: {e}")
            traceback.print_exc()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

class LejuClaw:
    """leju claw controller leju claw controller"""
    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager or ROSManager()
        self._pub_leju_claw_cmd = self.ros_manager.register_publisher('/leju_claw_command', lejuClawCommand, queue_size=10)

    def control(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """Hand control"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 2)
        
        cmd = lejuClawCommand()
        cmd.data.name = ['left_claw', 'right_claw']
        
        target_positions = [max(0.0, min(100.0, pos)) for pos in target_positions]
        target_velocities = self._get_default_velocities(target_velocities, 2)
        target_torques = self._get_default_torques(target_torques, 2)
        
        cmd.data.position = target_positions
        cmd.data.velocity = target_velocities
        cmd.data.effort = target_torques
        self._pub_leju_claw_cmd.publish(cmd)

    def control_left(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """Left hand control Left hand control"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [target_positions[0], 0],
            [target_velocities[0] if target_velocities else 90, 0],
            [target_torques[0] if target_torques else 1.0, 0]
        )

    def control_right(self, target_positions: list, target_velocities: list = None, target_torques: list = None):
        """Right hand control"""
        self._validate_inputs(target_positions, target_velocities, target_torques, 1)
        self.control(
            [0, target_positions[0]],
            [0, target_velocities[0] if target_velocities else 90],
            [0, target_torques[0] if target_torques else 1.0]
        )

    def _validate_inputs(self, positions, velocities, torques, expected_len):
        """Validate input parameters Validate input parameters"""
        assert len(positions) == expected_len, f"target_positions must be a list of length {expected_len}"
        if velocities is not None:
            assert len(velocities) == expected_len, f"target_velocities must be a list of length {expected_len}"
        if torques is not None:
            assert len(torques) == expected_len, f"target_torques must be a list of length {expected_len}"

    def _get_default_velocities(self, velocities, length):
        """Obtain default velocities"""
        if velocities is None:
            return [90] * length
        return [max(0.0, min(100.0, vel)) for vel in velocities]

    def _get_default_torques(self, torques, length):
        """Obtain default torques Obtain default torques"""
        if torques is None:
            return [1.0] * length
        return [max(0.0, min(10.0, torque)) for torque in torques]

    def close(self):
        """Release resources Release resources"""
        if hasattr(self, 'ros_manager'):
            self.ros_manager.close()

#Usage example
if __name__ == "__main__":
    from kuavo_deploy.config import load_kuavo_config
    
    #Use context manager to ensure resources are properly released Use context manager to ensure resources are properly released
    with KuavoBaseRosEnv(load_kuavo_config()) as env:
        obs, info = env.reset()

        for _ in range(1):
            obs = env.get_obs()
            env.rate.sleep()
            print(obs.keys())
            for k, v in obs.items():
                print(k, v.shape)
                print(v.max(), v.min())
