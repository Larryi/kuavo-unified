# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from lerobot_patches import custom_patches

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

from kuavo_deploy.config import KuavoConfig
from kuavo_deploy.utils.diagnostics import DiagnosticsManager
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.gripper_latch import GripperIntentLatch, GripperLatchConfig
from kuavo_deploy.utils.policy_loader import (
    load_policy_and_processors,
    resolve_eval_output_dir,
    resolve_policy_path,
)
from kuavo_deploy.kuavo_service.client import PolicyClient
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors

log_model = setup_logger("model")
log_robot = setup_logger("robot")


def _create_diagnostics(output_directory: Path, config: KuavoConfig, cfg):
    """Create diagnostics without ever making telemetry a deployment dependency."""
    try:
        return DiagnosticsManager.from_env(
            output_directory=output_directory,
            run_meta={
                "entrypoint": "real_single_test",
                "task": cfg.task,
                "method": cfg.method,
                "timestamp": cfg.timestamp,
                "epoch": str(cfg.epoch),
                "env_name": config.env.env_name,
                "eval_episodes": cfg.eval_episodes,
                "policy_type": cfg.policy_type,
            },
        )
    except Exception as exc:
        log_robot.warning(f"Diagnostics disabled: failed to initialize diagnostics manager: {exc}")
        return None


def _finalize_diagnostics_episode(
    diagnostics,
    recorder,
    episode: int,
    success: bool,
    steps: int,
    fps,
    video_paths=None,
    extra=None,
):
    """Finalize/upload a real-robot episode, swallowing all probe failures."""
    if diagnostics is None or recorder is None:
        return
    try:
        bundle_path = recorder.finalize(
            video_paths=video_paths or [],
            success=success,
            steps=steps,
            fps=fps,
            extra=extra or {},
        )
        diagnostics.upload_async(bundle_path, episode=episode, success=success)
    except Exception:
        return

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()


def setup_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """
    Set up and load the policy model.
    
    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion' or 'act')
        
    Returns:
        Loaded policy model and device
    """
    
    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)  
    
    if policy_type == 'client':
        policy = PolicyClient()
        return policy
    else:
        policy, _, _ = load_policy_and_processors(pretrained_path, policy_type, device)
    return policy


def _policy_kwargs_for_cfg(cfg):
    if cfg.policy_type == "lingbot":
        return {
            "lingbot_root": getattr(cfg, "lingbot_root", ""),
            "qwen25_path": getattr(cfg, "qwen25_path", ""),
            "task_prompt": getattr(cfg, "task_prompt", "") or getattr(cfg, "task", ""),
            "use_length": getattr(cfg, "lingbot_use_length", 1),
            "chunk_ret": getattr(cfg, "lingbot_chunk_ret", True),
            "norm_stats_file": getattr(cfg, "lingbot_norm_stats_file", ""),
            "data_type": getattr(cfg, "lingbot_data_type", "robotwin"),
            "execute_raw_action": getattr(cfg, "lingbot_execute_raw_action", False),
        }
    if cfg.policy_type == "lingbot_v2":
        return {
            "lingbot_v2_root": getattr(cfg, "lingbot_v2_root", ""),
            "qwen3vl_path": getattr(cfg, "qwen3vl_path", ""),
            "robot_name": getattr(cfg, "lingbot_v2_robot_name", "kuavo_v2"),
            "task_prompt": getattr(cfg, "task_prompt", "") or getattr(cfg, "task", ""),
            "use_length": getattr(cfg, "lingbot_use_length", 5),
            "chunk_ret": getattr(cfg, "lingbot_chunk_ret", True),
            "norm_stats_file": getattr(cfg, "lingbot_norm_stats_file", ""),
            "use_compile": getattr(cfg, "lingbot_v2_use_compile", False),
        }
    return {}

def main(config: KuavoConfig, env: gym.Env):
    # load config
    cfg = config.inference

    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed
    policy_type = cfg.policy_type
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    env_name = cfg.env_name

    pretrained_path = resolve_policy_path(cfg)
    output_directory = resolve_eval_output_dir(cfg, pretrained_path)
    # Create a directory to store the video of the evaluation
    output_directory.mkdir(parents=True, exist_ok=True)

    diagnostics = _create_diagnostics(output_directory, config, cfg)
    if diagnostics is not None:
        try:
            diagnostics.notify_eval_started()
            diagnostics.smoke_test_async()
        except Exception:
            pass

    active_recorder = None
    active_episode = 0
    active_step = 0
    active_finalized = True
    active_fps = getattr(env, "ros_rate", 0)
    active_video_recorder = None
    active_snapshot_uploader = None

    try:
        # set seed
        set_seed(seed=seed)

        # Select your device
        device = torch.device(cfg.device)

        if policy_type == "client":
            policy = setup_policy(pretrained_path, policy_type, device)
            preprocessor, postprocessor = make_pre_post_processors(
                None, Path(str(pretrained_path).split("/epoch", 1)[0])
            )
        else:
            policy, preprocessor, postprocessor = load_policy_and_processors(
                pretrained_path,
                policy_type,
                device,
                policy_kwargs=_policy_kwargs_for_cfg(cfg),
            )

        # Initialize evaluation environment to render two observation types:
        # an image of the scene and state/position of the agent.
        max_episode_steps = cfg.max_episode_steps

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
        if policy_type != 'client':
            log_model.info(f"policy.config.input_features: {policy.config.input_features}")
            log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
        if policy_type != 'client':
            log_model.info(f"policy.config.output_features: {policy.config.output_features}")
            log_robot.info(f"env.action_space: {env.action_space}")

    # Log evaluation results
        log_file_path = output_directory / "evaluation.log"
        with log_file_path.open("w") as log_file:
            log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
            log_file.write(f"Total Episodes: {eval_episodes}\n")

        success_count = 0
        for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
            active_episode = episode
            active_step = 0
            active_finalized = False
            active_fps = getattr(env, "ros_rate", 0)
            active_recorder = diagnostics.start_episode(episode, config) if diagnostics is not None else None
            gripper_latch = None
            if cfg.gripper_latch_enabled:
                control_hz = float(getattr(env, "ros_rate", config.env.ros_rate))
                gripper_latch = GripperIntentLatch(
                    GripperLatchConfig(
                        action_indices=tuple(cfg.gripper_latch_action_indices),
                        intent_steps=cfg.gripper_latch_intent_steps,
                        close_threshold=cfg.gripper_latch_close_threshold,
                        open_threshold=cfg.gripper_latch_open_threshold,
                        close_ratio=cfg.gripper_latch_close_ratio,
                        open_ratio=cfg.gripper_latch_open_ratio,
                        min_close_steps=round(cfg.gripper_latch_min_close_seconds * control_hz),
                        min_open_steps=round(cfg.gripper_latch_min_open_seconds * control_hz),
                    )
                )
                log_robot.info(
                    "Gripper latch enabled: indices=%s, control_hz=%.1f, min_close_steps=%d, min_open_steps=%d",
                    cfg.gripper_latch_action_indices,
                    control_hz,
                    gripper_latch.config.min_close_steps,
                    gripper_latch.config.min_open_steps,
                )
        # Reset the policy and environments to prepare for rollout
            if hasattr(policy, "reset"):
                policy.reset()
            observation, info = env.reset(seed=episode+start_seed)
            if active_recorder is not None:
                active_recorder.log_state(0, observation.get("observation.state"), None)
            active_video_recorder = (
                diagnostics.start_video_recorder(
                    output_directory,
                    episode,
                    observation,
                    getattr(env, "ros_rate", 10),
                )
                if diagnostics is not None
                else None
            )
            if active_video_recorder is not None:
                active_video_recorder.submit(observation)
            active_snapshot_uploader = (
                diagnostics.start_snapshot_uploader(
                    episode,
                    observation,
                    recorder=active_recorder,
                )
                if diagnostics is not None
                else None
            )
            if active_snapshot_uploader is not None:
                active_snapshot_uploader.submit(observation, step=0, force=True)
            observation = preprocessor(observation)
        # log_file.write(f"~~~~~~~~~~~~~~~~~~preprocess observation ok!~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
            rewards = []

            cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]
            frame_map = {k: [] for k in cam_keys}

            average_exec_time = 0
            average_action_infer_time = 0
            average_step_time = 0

            step = 0
            done = False
            with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
                while not done:
                # --- Pause support: block here if pause_flag is set ---
                    while pause_flag.is_set() and not stop_flag.is_set():
                        log_model.info("Paused. Waiting for resume signal...")
                        time.sleep(0.5)
                    if stop_flag.is_set():
                        log_model.info("Stop flag detected during pause. Exiting loop.")
                        return
                
                    start_time = time.time()
                
                    with torch.inference_mode():
                        action = policy.select_action(observation)
                    action = postprocessor(action)
                    if gripper_latch is not None:
                        action = gripper_latch.process_chunk(action)
                    action_infer_time = time.time()
                    log_model.debug(f"action infer time: {action_infer_time - start_time:.3f}s")
                    average_action_infer_time += action_infer_time - start_time

                    numpy_action = action.squeeze(0).cpu().numpy()
                    if active_recorder is not None:
                        active_recorder.log_action(step, numpy_action, action_infer_time - start_time)
                    log_model.debug(f"numpy_action: {numpy_action}")

                # 执行动作
                    observation, reward, terminated, truncated, info = env.step(numpy_action)
                    if gripper_latch is not None:
                        gripper_latch.advance(1)
                    if active_video_recorder is not None:
                        active_video_recorder.submit(observation)
                    if active_snapshot_uploader is not None:
                        active_snapshot_uploader.submit(observation, step=step + 1)
                    exec_time = time.time()
                    if active_recorder is not None:
                        active_recorder.log_state(step + 1, observation.get("observation.state"), reward)
                        active_recorder.log_timing(
                            step,
                            action_infer_time_sec=action_infer_time - start_time,
                            exec_time_sec=exec_time - action_infer_time,
                        )
                    observation = preprocessor(observation)
                    log_model.debug(f"exec time: {exec_time - action_infer_time:.3f}s")
                    average_exec_time += exec_time - action_infer_time

                    rewards.append(reward)

                # 相机帧记录，真机请取消，否则会一直堆叠卡死

                # for k in cam_keys:
                #     frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                    done = terminated | truncated | done
                    step += 1
                    active_step = step

                    end_time = time.time()
                    log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")
                
                # Update progress bar
                    status = "Success" if terminated else "Running"
                    pbar.set_postfix({
                        "Reward": f"{reward:.3f}",
                        "Status": status,
                        "Total Reward": f"{sum(rewards):.3f}"
                    })
                    pbar.update(1)

            if terminated:
                success_count += 1
                log_model.info(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
            else:
                log_model.info(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
            fps = env.ros_rate
            active_fps = fps

            log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
            log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
            log_model.info(f"average step time: {average_step_time / step:.3f}s")
            log_model.info(f"average sleep time: {env.average_sleep_time / step:.3f}s")
        
        
        # Encode all frames into a mp4 video.
            if len(frame_map.keys()) == 0:
                for cam in cam_keys:
                    frames = frame_map[cam]
                    output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
                    imageio.mimsave(str(output_path), frames, fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

            with log_file_path.open("a") as log_file:
                log_file.write("\n")
                log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

            video_paths = (
                active_video_recorder.close() if active_video_recorder is not None else []
            )
            active_video_recorder = None
            if active_snapshot_uploader is not None:
                active_snapshot_uploader.close()
            active_snapshot_uploader = None
            _finalize_diagnostics_episode(
                diagnostics,
                active_recorder,
                episode=episode,
                success=bool(terminated),
                steps=step,
                fps=fps,
                video_paths=video_paths or sorted(output_directory.glob(f"rollout_{episode}_*.mp4")),
                extra={"entrypoint": "real_single_test", "partial": False},
            )
            active_finalized = True
            active_recorder = None

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Success Count: {success_count}\n")
            log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
        log_model.info("\n" + "="*50)
        log_model.info(f"🎯 Evaluation completed!")
        log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
        log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
        print(f"📁 Videos and logs saved to: {output_directory}")
        print("="*50)
    finally:
        partial_video_paths = []
        if active_video_recorder is not None:
            try:
                partial_video_paths = active_video_recorder.close()
            except Exception:
                pass
            active_video_recorder = None
        if active_snapshot_uploader is not None:
            try:
                active_snapshot_uploader.close()
            except Exception:
                pass
            active_snapshot_uploader = None
        if not active_finalized and active_recorder is not None:
            _finalize_diagnostics_episode(
                diagnostics,
                active_recorder,
                episode=active_episode,
                success=False,
                steps=active_step,
                fps=active_fps,
                video_paths=partial_video_paths
                or sorted(output_directory.glob(f"rollout_{active_episode}_*.mp4")),
                extra={"entrypoint": "real_single_test", "partial": True},
            )
        if diagnostics is not None:
            try:
                diagnostics.close()
            except Exception:
                pass

def kuavo_eval(config: KuavoConfig, env: gym.Env):
    main(config, env)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )
