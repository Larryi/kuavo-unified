"""
Robot control sample program
Provides functions such as robotic arm motion control, trajectory playback, etc.

Usage example:
  python scripts_auto_test.py --task auto_test --config /path/to/custom_config.yaml
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
log_model = setup_logger("model", "DEBUG")  #weblog
log_robot = setup_logger("robot", "DEBUG")  #Robot log

#control variables
class ArmMoveController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        
    def pause(self):
        with self.lock:
            self.paused = True
            log_robot.info("🔄 Robot arm motion paused")
    
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

#Ros issues pause/stop signal
pause_pub = rospy.Publisher('/kuavo/pause_state', Bool, queue_size=1)
stop_pub = rospy.Publisher('/kuavo/stop_state', Bool, queue_size=1)

def signal_handler(signum, frame):
    """signal processor"""
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
    """Set up signal handler"""
    signal.signal(signal.SIGUSR1, signal_handler)  #pause/resume
    signal.signal(signal.SIGUSR2, signal_handler)  #stop
    log_robot.info("📡 Signal handler successfully set up:")
    log_robot.info("  SIGUSR1 (kill -USR1): Pause/resume arm motion")
    log_robot.info("  SIGUSR2 (kill -USR2): Stop arm motion")

class ArmMove:
    """Robotic arm motion control class"""
    
    def __init__(self, config: KuavoConfig):
        """
        Initialize robot arm control
        
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

        rospy.init_node('kuavo_deploy', anonymous=True)

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
    

    def auto_test(self) -> None:
        """Execute automated tests"""
        from kuavo_deploy.src.eval.sim_auto_test import kuavo_eval_autotest
        from kuavo_deploy.utils.policy_server_manager import managed_policy_server

        with managed_policy_server(self.inference_config, log_model):
            kuavo_eval_autotest(config=self.config)
    
def parse_args():
    """Parse command line parameters"""
    parser = argparse.ArgumentParser(
        description="Kuavo robot control sample program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage example:
  python scripts_auto_test.py --task auto_test --config /path/to/custom_config.yaml"           #Automatically test the model in simulation and execute eval_episodes times


Mission description:
  auto_test   - Automatically test the model in simulation and execute eval_episodes times
        """
    )
    
    #Required parameters
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        choices=["auto_test"],
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
        "auto_test": arm.auto_test,      #Automatically test the model in simulation and execute eval_episodes times
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
