"""
Robot control sample program
Provides functions such as robotic arm motion control, trajectory playback, etc.

Usage example:
  python scripts.py --task go --config /path/to/custom_config.yaml"                   #First interpolate to the position of the first frame of the bag, then play back the bag and return to the working position.
  python scripts.py --task run --config /path/to/custom_config.yaml"                  #Run the model directly from the current location
  python scripts.py --task go_run --config /path/to/custom_config.yaml"               #Arrive at work and run the model directly
  python scripts.py --task here_run --config /path/to/custom_config.yaml"             #Interpolation to the last frame state of the bag starts running
  python scripts.py --task back_to_zero --config /path/to/custom_config.yaml"         #After interrupting model inference, turn the bag upside down and return it to position 0.
"""

import rospy
import rosbag
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
from kuavo_deploy.config import load_kuavo_config, KuavoConfig
import gymnasium as gym

import numpy as np
import signal
import sys,os
import threading
import subprocess
import traceback

from std_msgs.msg import Bool

#Configuration log
log_model = setup_logger("model", "DEBUG")  #Web logs Web logs
log_robot = setup_logger("robot", "DEBUG")  #Robot logs Robot logs

#control variables
class ArmMoveController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        
    def pause(self):
        with self.lock:
            self.paused = True
            log_robot.info("🔄 Robot arm motion stopped")
    
    def resume(self):
        with self.lock:
            self.paused = False
            log_robot.info("▶️ Robot arm motion resumed")
    
    def stop(self):
        with self.lock:
            self.should_stop = True
            log_robot.info("⏹️ Robot arm motion stopped")
    
    def is_paused(self):
        with self.lock:
            return self.paused
    
    def should_exit(self):
        with self.lock:
            return self.should_stop

#Controller instance
arm_controller = ArmMoveController()

#ROS pause and stop publishers ROS pause and stop publishers
pause_pub = rospy.Publisher('/kuavo/pause_state', Bool, queue_size=1)
stop_pub = rospy.Publisher('/kuavo/stop_state', Bool, queue_size=1)

def signal_handler(signum, frame):
    """Signal handler Signal handler"""
    log_robot.info(f"🔔 Received signal: {signum}")
    if signum == signal.SIGUSR1:  #pause/resume
        if arm_controller.is_paused():
            log_robot.info("🔔 Current status: Paused. Resuming")
            arm_controller.resume()
            pause_pub.publish(False)
        else:
            log_robot.info("🔔 Current status: Operating. Pausing")
            arm_controller.pause()
            pause_pub.publish(True)
    elif signum == signal.SIGUSR2:  #stop
        log_robot.info("�� Stopping")
        arm_controller.stop()
        stop_pub.publish(True)
    log_robot.info(f"🔔 Signal successfully processed. Current state - Pause: {arm_controller.is_paused()}, Stop: {arm_controller.should_exit()}")

def setup_signal_handlers():
    """Setting up signal handler Setting up signal handler"""
    signal.signal(signal.SIGUSR1, signal_handler)  #pause/resume
    signal.signal(signal.SIGUSR2, signal_handler)  #stop
    log_robot.info("📡 Signal handler successfully set up:")
    log_robot.info("  SIGUSR1 (kill -USR1): Pause/resume arm motion")
    log_robot.info("  SIGUSR2 (kill -USR2): Stop arm motion")

def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

class ArmMove:
    """Robot arm motion class Robot arm motion class"""
    
    def __init__(self, config: KuavoConfig):
        """
        Initialize arm motion control Initialize arm motion control
        
        Args:
            bag_path: Track file path
        """
        self.config = config
        #Set up signal handler
        self.shutdown_requested = False
        #Set up signal handler
        setup_signal_handlers()
        
        #Output the current process ID to facilitate external control
        pid = os.getpid()
        log_robot.info(f"🆔 Current process ID: {pid}")
        log_robot.info(f"💡 Use the following commands to control arm motion:")
        log_robot.info(f"   Pause/Resume: kill -USR1 {pid}")
        log_robot.info(f"   Stop: kill -USR2 {pid}")

        self.inference_config = config.inference
        self.bag_path = self.inference_config.go_bag_path
        self.msg_dict_of_list = {}

        rospy.init_node('kuavo_deploy', anonymous=True)
        self.env = gym.make(
            self.config.env.env_name,
            max_episode_steps=self.inference_config.max_episode_steps,
            config=self.config,
        )
        self.env = unwrap_env(self.env)


    def _check_control_signals(self):
        """Check control signals"""
        #Check pause status
        while arm_controller.is_paused():
            log_robot.info("🔄 Robot arm motion paused")
            time.sleep(0.1)
            if arm_controller.should_exit():
                log_robot.info("🛑 Robot arm motion stopped")
                return False
        
        #Check if it needs to be stopped
        if arm_controller.should_exit():
            log_robot.info("🛑 Stop signal detected, exiting arm motion")
            return False
            
        return True  #Continue normally
    
    def _read_topic_messages(self, bag_path, topic_names: list = None) -> dict:
        """
        Read the messages of the specified topic in the bag and convert them into a dictionary
        :param bag_path: bagfile path
        :param topic_names: Topic name list
        :return: Message dictionary, key is the topic name, value is the message list of the topic
        """
        messages_dict = {}
        try:
            bag = rosbag.Bag(bag_path)
            for topic, msg, t in bag.read_messages(topics=topic_names):
                if topic not in messages_dict:
                    messages_dict[topic] = []
                messages_dict[topic].append(msg)
            bag.close()
            return messages_dict
        except Exception as e:
            rospy.logerr(f"Failed to read messages from bag: {e}")
            return {}

    def _ensure_bag_loaded(self) -> None:
        """Load bags on demand, only called when trajectory playback tasks are required."""
        if self.msg_dict_of_list:
            return
        if not self.bag_path or not Path(self.bag_path).exists():
            raise FileNotFoundError(f"Bag file not found: {self.bag_path}")
        self.msg_dict_of_list = self._read_topic_messages(
            bag_path=self.bag_path,
            topic_names=["/control_robot_hand_position", "/leju_claw_command", "/kuavo_arm_traj"],
        )
        if "/kuavo_arm_traj" not in self.msg_dict_of_list or not self.msg_dict_of_list["/kuavo_arm_traj"]:
            raise ValueError(f"No '/kuavo_arm_traj' messages found in bag: {self.bag_path}")

    def _pub_arm_traj(self, msg) -> None:
        """Publish robot arm trajectory"""
        #If msg is a list, publish it directly
        if isinstance(msg, list):
            position = msg
        else:
            position = np.array(msg.position)/180*np.pi
        if self.env.which_arm=="both":
            target_positions = position
        elif self.env.which_arm=="left":
            target_positions = np.concatenate([position[:7],self.env.arm_init[7:]],axis=0)
        elif self.env.which_arm=="right":
            target_positions = np.concatenate([self.env.arm_init[:7],position[7:]],axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.robot.control_arm_joint_positions(target_positions)
    
    def _pub_leju_claw(self, msg) -> None:
        """Release Gripper"""
        if self.env.which_arm=="both":
            target_positions = msg.data.position
        elif self.env.which_arm=="left":
            target_positions = np.concatenate([msg.data.position[:1],[0]],axis=0)
        elif self.env.which_arm=="right":
            target_positions = np.concatenate([[0],msg.data.position[1:]],axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.lejuclaw.control(target_positions)
    
    def _pub_qiangnao(self, msg) -> None:
        """post dexterity"""
        left_hand_position = np.frombuffer(msg.left_hand_position, dtype=np.uint8)
        right_hand_position = np.frombuffer(msg.right_hand_position, dtype=np.uint8)
        if self.env.which_arm=="both":
            target_positions = np.concatenate([left_hand_position,right_hand_position],axis=0)
        elif self.env.which_arm=="left":
            target_positions = np.concatenate([left_hand_position,[0,0,0,0,0,0]],axis=0)
        elif self.env.which_arm=="right":
            target_positions = np.concatenate([[0,0,0,0,0,0],right_hand_position],axis=0)
        else:
            raise ValueError(f"Invalid which_arm: {self.env.which_arm}, must be 'left', 'right', or 'both'")
        self.env.qiangnao.control(target_positions)

    def _pub_rq2f85(self,msg) -> None:
        self.env.pub_eef_joint.publish(msg)

    def play_bag(self, go_bag, reverse=False):
        """
        Move the robotic arm to the working posture. Issue arm, hand position, and gripper commands evenly.
        
        Args:
            reverse (bool): If True, play the command sequence in reverse order
        """

        # topic_names = ["/joint_cmd", "/control_robot_hand_position", "/leju_claw_command"],
        if self.env.eef_type == 'leju_claw':
            topics = ["/kuavo_arm_traj", "/leju_claw_command"]
        elif self.env.eef_type == 'qiangnao':
            topics = ["/kuavo_arm_traj", "/control_robot_hand_position"]
        elif self.env.eef_type == 'rq2f85':
            topics = ["/kuavo_arm_traj", "/gripper_command"]
        else:
            raise ValueError(f"Invalid eef_type: {self.env.eef_type}, must be 'leju_claw' or 'qiangnao' or 'rq2f85' ")
        
        msg_dict_of_list = self._read_topic_messages(
            bag_path = go_bag, 
            topic_names = topics
        )
        if reverse:
            msg_dict_of_list = {topic: msg_dict_of_list[topic][::-1] for topic in msg_dict_of_list}
        log_robot.info(f"Messages for the topic {[topic for topic in msg_dict_of_list.keys()]} in {go_bag} will be played back")
        
        #Initialize the message dictionary, check whether the key value exists and get the message list
        msg_lists = {}
        for topic in msg_dict_of_list:
            msg_lists[topic] = {
                "msgs": msg_dict_of_list[topic],
                "total": len(msg_dict_of_list[topic]),
                "index": 0,
            }
                
        if not msg_lists:
            log_robot.warning("No valid messages playable")
            return
        
        #Calculate the total number of steps as the length of the longest message list
        max_steps = max(info["total"] for info in msg_lists.values())
        log_robot.info(f"Now evenly playing {max_steps} steps of message data")
        
        #Publish remaining data evenly
        rate = rospy.Rate(100)  #100Hz, can be adjusted as needed
        for step in range(1, max_steps):
            #Check control signals
            if not self._check_control_signals():
                log_robot.info("🛑 Track playback stopped")
                return

            for topic, info in msg_lists.items():
                #Calculate the current index that should be published
                #Use floating point calculations to ensure uniform distribution, then round
                target_index = min(int(step * info["total"] / max_steps), info["total"] - 1)
                
                #Only publish new messages when the index changes
                if target_index > info["index"]:
                    if topic=="/kuavo_arm_traj":
                        self._pub_arm_traj(info["msgs"][target_index])
                    elif topic=="/leju_claw_command":
                        self._pub_leju_claw(info["msgs"][target_index])
                    elif topic=="/control_robot_hand_position":
                        self._pub_qiangnao(info["msgs"][target_index])
                    elif topic=="/gripper_command":
                        self._pub_rq2f85(info["msgs"][target_index])
                log_robot.info(f"Publishing {topic} message: {target_index+1}/{info['total']}")
            #Control publishing frequency
            rate.sleep()
        
        #Ensure the last frame of data is published
        for topic, info in msg_lists.items():
            if info["index"] < info["total"] - 1:
                target_index = info["total"] - 1
                if topic=="/kuavo_arm_traj":
                    self._pub_arm_traj(info["msgs"][target_index])
                elif topic=="/leju_claw_command":
                    self._pub_leju_claw(info["msgs"][target_index])
                elif topic=="/control_robot_hand_position":
                    self._pub_qiangnao(info["msgs"][target_index])
                elif topic=="/gripper_command":
                    self._pub_rq2f85(info["msgs"][target_index])
                log_robot.info(f"Publishing {topic}'s last message")
        
        log_robot.info("Sequential message playback completed")

    def _get_current_joint_angles(self) -> List[float]:
        """Get the current joint angle (rad)"""
        return self.env.robot_state.arm_joint_state().position

    def _arm_interpolate_joint(self, q0: List[float], q1: List[float], steps: int = 100) -> List[List[float]]:
        """
        Generates a smooth interpolated trajectory from q0 to q1.
        
        Args:
            q0: Initial joint position list
            q1: Target joint position list
            steps: Number of interpolation steps, default is INTERPOLATION_STEPS
            
        Returns:
            List containing interpolation positions, each element is a list of length NUM_JOINTS
            
        Raises:
            ValueError: If you enter an incorrect number of joint positions
        """
        NUM_JOINTS = 14  #Assume there are 14 joints
        if len(q0) != NUM_JOINTS or len(q1) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} joint positions")
        
        return [
            [
                q0[j] + i / float(steps) * (q1[j] - q0[j])
                for j in range(NUM_JOINTS)
            ]
            for i in range(steps)
        ]

    def _move_to_joint_angles(self, target_angles: List[float], steps: int = 100) -> None:
        """
        Move to target joint angle
        
        Args:
            target_angles: Target joint angle list
            steps: Interpolation steps
        """
        current_angles = self._get_current_joint_angles()
        log_robot.info(f"Current joint angle: {current_angles}")
        arm_inter = self._arm_interpolate_joint(
            current_angles, target_angles, steps=steps
        )
        
        for joint_angles in arm_inter:
            if not self._check_control_signals():
                log_robot.info("🛑 Joint angle movement stopped")
                return
            log_robot.info(f"Robot joint angle: {joint_angles}")
            self._pub_arm_traj(joint_angles)
            time.sleep(0.1)

    def go(self) -> None:
        """First interpolate to the position of the first frame of the bag, then play back the bag and return to the working position."""
        time.sleep(1)
        self._ensure_bag_loaded()
        #Move to the starting position of the track
        start_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[0].position]
        start_angles = np.array(start_angles)/180*np.pi
        self._move_to_joint_angles(start_angles)
        #Play track
        self.play_bag(go_bag=self.bag_path)

    def here_run(self) -> None:
        """Interpolate directly to the last frame position of the bag and run"""
        time.sleep(1)
        self._ensure_bag_loaded()
        #Move to the end of the track
        end_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[-1].position]
        end_angles = np.array(end_angles)/180*np.pi
        self._move_to_joint_angles(end_angles)
        #Perform assessment
        self.run()

    def back_to_zero(self) -> None:
        """Return to zero position"""
        time.sleep(1)
        self._ensure_bag_loaded()
        #Move to the end of the track
        end_angles = [float(j) for j in self.msg_dict_of_list.get("/kuavo_arm_traj", [])[-1].position]
        end_angles = np.array(end_angles)/180*np.pi
        self._move_to_joint_angles(end_angles)
        #Play track in reverse
        self.play_bag(go_bag=self.bag_path,reverse=True)
        #move to zero position
        zero_angles = [0.0] * 14
        self._move_to_joint_angles(zero_angles)

    def go_run(self) -> None:
        """Execute go and run"""
        self.go()
        self.run()

    def run(self) -> None:
        """Execute run"""
        from kuavo_deploy.src.eval.real_single_test import kuavo_eval
        from kuavo_deploy.utils.policy_server_manager import managed_policy_server

        with managed_policy_server(self.inference_config, log_model):
            kuavo_eval(config=self.config, env=self.env)

def parse_args():
    """Parse command line parameters"""
    parser = argparse.ArgumentParser(
        description="Kuavo robot control sample program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python scripts.py --task go --config /path/to/custom_config.yaml"                   #First interpolate to the position of the first frame of the bag, then play back the bag and return to the working position.
  python scripts.py --task run --config /path/to/custom_config.yaml"                  #Run the model directly from the current location
  python scripts.py --task go_run --config /path/to/custom_config.yaml"               #Arrive at work and run the model directly
  python scripts.py --task here_run --config /path/to/custom_config.yaml"             #Interpolation to the last frame state of the bag starts running
  python scripts.py --task back_to_zero --config /path/to/custom_config.yaml"         #After interrupting model inference, turn the bag upside down and return it to position 0.

Mission description:
  go          - First interpolate to the position of the first frame of the bag, then play back the bag and return to the working position.
  run         - Run the model directly from the current location
  go_run      - Arrive at work and run the model directly
  here_run    - Interpolation to the last frame state of the bag starts running
  back_to_zero - After interrupting model inference, turn the bag upside down and return it to position 0.
  auto_test   - Automatically test the model in simulation and execute eval_episodes times
        """
    )
    
    #Required parameters
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["go", "run", "go_run", "here_run", "back_to_zero"],
        help="type of task to perform"
    )
    
    #Optional parameters
    parser.add_argument(
        "--config", 
        type=str,
        required=True,
        help="Configuration file path (must be specified)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode, which only displays the operations to be performed but does not actually execute them"
    )
    
    return parser.parse_args()

def main():
    """main function"""
    #Parse command line parameters
    args = parse_args()
    
    #Set log level
    if args.verbose:
        log_model.setLevel("DEBUG")
        log_robot.setLevel("DEBUG")
    
    #Determine configuration file path
    config_path = Path(args.config)
    
    log_robot.info(f"Use configuration file: {config_path}")
    log_robot.info(f"Executing task: {args.task}")
    
    config = load_kuavo_config(config_path)
    #Initialize the robot arm
    try:
        arm = ArmMove(config)
        log_robot.info("Arm initialisation successful")
    except Exception as e:
        log_robot.error(f"Arm initialisation failed: {e}")
        return
    
    #Dry running mode
    if args.dry_run:
        log_robot.info("=== Dry Run Mode ===")
        log_robot.info(f"Task to be executed: {args.task}")
        log_robot.info("Dry run successfully completed. No actual tasks executed")
        return
    
    #task mapping
    task_map = {
        "go": arm.go,                    #Arrive at work location
        "run": arm.run,                  #Run the model directly from the current location
        "go_run": arm.go_run,           #Arrive at work and run the model directly
        "here_run": arm.here_run,       #Start running from the last frame state of go_bag
        "back_to_zero": arm.back_to_zero, #After interrupting model inference, turn the bag upside down and return it to position 0.
    }
    
    #perform tasks
    try:
        log_robot.info(f"Now running task: {args.task}")
        task_map[args.task]()
        log_robot.info(f"Task {args.task} successfully completed")
    except KeyboardInterrupt:
        log_robot.info("User interrupt detected!")
    except Exception as e:
        traceback.print_exc()
        log_robot.error(f"Task {args.task} encountered error: {e}")

if __name__ == "__main__":
    main()
