"""
Chunked streaming rosbag converter - low memory version

Core optimization (refer to the on-demand reading method of Diffusion Policy):
1. First pass of scan: only read timestamp (memory occupied a few MB)
2. The second scan: read in blocks by time window + align + write to the dataset

Differences from original CvtRosbag2Lerobot.py:
- Original: Load the entire rosbag into memory at once→ Alignment→ Write (memory peak is huge)
- This version: chunked reading→ Instant alignment→ instant write→ Release memory (memory controllable)

How to use:
    python CvtRosbag2Lerobot_chunked.py --config-name=KuavoRosbag2Lerobot \
        rosbag.rosbag_dir=/path/to/rosbag \
        rosbag.lerobot_dir=/path/to/output \
        rosbag.chunk_size=100
"""
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import os
import gc
import inspect
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import dataclasses
from kuavo_data.common import kuavo_dataset as kuavo
from rich.logging import RichHandler
import logging

log_print = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    from rich.logging import RichHandler
    root.addHandler(
        RichHandler(
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
    )


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 1
    video_backend: str | None = "pyav"
    streaming_encoding: bool = True
    encoder_threads: int = 8
    batch_encoding_size: int = 1

DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset_chunked(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    
    #Determine whether it is the joint type of the half body and end according to the parameters of config
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = kuavo.DEFAULT_CAMERA_NAMES


    action_dim = (len(motors),)

    # set action name/dim, state name/dim,
    action_name =  motors

    state_dim = (len(motors),)

    # state_name = kuavo.DEFAULT_ARM_JOINT_NAMES[:len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2] + ["gripper_l"] + kuavo.DEFAULT_ARM_JOINT_NAMES[len(kuavo.DEFAULT_ARM_JOINT_NAMES)//2:] + ["gripper_r"]
    state_name = motors

    if not kuavo.ONLY_HALF_UP_BODY:
        action_dim = (action_dim[0] + 3 + 1,)  #cmd_pos_world3+breakpoint flag 1
        action_name += ["cmd_pos_x", "cmd_pos_y", "cmd_pos_yaw", "ctrl_change_cmd"]
        state_dim = (state_dim[0] + 0,)  #Robot base_pos_world3 + breakpoint flag 1
        state_name += []  #As above ["base_pos_x", "base_pos_y", "base_pos_yaw", "ctrl_change_flag"]

    # create corresponding features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_dim,
            "names": {
                "state_names": state_name
            }
        },
        "action": {
            "dtype": "float32",
            "shape": action_dim,
            "names": {
                "action_names": action_name
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.{cam}"] = {
                "dtype": mode, 
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),  # Attention: for datasets.features "image" and "video", it must be c,h,w style! 
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    create_kwargs = dict(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )

    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    for name in ("streaming_encoding", "encoder_threads", "batch_encoding_size"):
        if name in create_parameters:
            create_kwargs[name] = getattr(dataset_config, name)

    return LeRobotDataset.create(**create_kwargs)


def populate_dataset_chunked(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    chunk_size: int = 100,
) -> LeRobotDataset:
    """
    Populate a dataset using chunked streaming
    
    Core optimization:
    1. The first scan only reads the timestamp (a few MB of memory)
    2. The second scan is divided into blocks according to the time window to read + align + write
    3. Save and release memory immediately after each chunk is processed
    
    Args:
        dataset: LeRobotDatasetExample
        bag_files: rosbagfile path list
        task: Task description
        episodes: List of episode indexes to process
        chunk_size: The number of frames each chunk contains (default 100 frames)
    """
    if episodes is None:
        episodes = range(len(bag_files))
    
    failed_bags = []
    log_print.info(f"Total episodes to process: {len(episodes)}")
    bag_reader = kuavo.KuavoRosbagReader()
    
    #Memory monitoring
    process = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        pass
    
    def log_memory(prefix: str):
        if process:
            mem_mb = process.memory_info().rss / 1024 / 1024
            log_print.debug(f"{prefix} Memory: {mem_mb:.2f} MB")
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        log_print.warning(f"Processing {ep_path}")
        log_memory("Before processing")
        
        try:
            #Collect all frames of the current episode
            frames_buffer = []
            frame_count = [0]
            
            def on_frame(aligned_frame: dict, frame_idx: int):
                """Processing single frame alignment data"""

                def get_array(key, dtype, default_empty=True):
                    item = aligned_frame.get(key)
                    if item is None:
                        return np.array([], dtype=dtype) if default_empty else None
                    return np.array(item.get("data", []), dtype=dtype)

                # =========================
                # 1. state / action
                # =========================
                state = get_array('observation.state', np.float32)
                action = get_array('action', np.float32)
                

                if state.size == 0 or action.size == 0:
                    return

                # =========================
                #2. arm trajectory (alt priority)
                # =========================
                arm_traj     = get_array("action.kuavo_arm_traj", np.float32)
                arm_traj_alt = get_array("action.kuavo_arm_traj_alt", np.float32)
                if arm_traj_alt.size == 0 and arm_traj.size == 0:
                    return
                action[12:26] = arm_traj_alt if arm_traj_alt.size else arm_traj
                
                #Interface reserved
                velocity = None
                effort = None

                # =========================
                #3. Hand data reading
                # =========================
                claw_state     = get_array("observation.claw", np.float64)
                claw_action    = get_array("action.claw", np.float64)
                qiangnao_state = get_array("observation.qiangnao", np.float64)
                qiangnao_action= get_array("action.qiangnao", np.float64)
                rq2f85_state   = get_array("observation.rq2f85", np.float64)
                rq2f85_action  = get_array("action.rq2f85", np.float64)

                if claw_state.size == 0 and qiangnao_state.size == 0 and rq2f85_state.size==0:
                    return 
                if claw_action.size == 0 and qiangnao_action.size==0 and rq2f85_action.size ==0:
                    return
                # =========================
                #4. Hand normalization (maintain original logic)
                # =========================
                if kuavo.IS_BINARY:
                    qiangnao_state  = np.where(qiangnao_state > 50, 1, 0)
                    qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
                    claw_state      = np.where(claw_state > 50, 1, 0)
                    claw_action     = np.where(claw_action > 50, 1, 0)
                    rq2f85_state    = np.where(rq2f85_state > 0.4, 1, 0)
                    rq2f85_action   = np.where(rq2f85_action > 70, 1, 0)
                    # rq2f85_state = np.where(rq2f85_state > 0.1, 1, 0)
                    # rq2f85_action = np.where(rq2f85_action > 128, 1, 0)
                else:
                    if claw_state.size:      claw_state /= 100
                    if claw_action.size:     claw_action /= 100
                    if qiangnao_state.size:  qiangnao_state /= 100
                    if qiangnao_action.size: qiangnao_action /= 100
                    if rq2f85_state.size:    rq2f85_state /= 0.8
                    if rq2f85_action.size:   rq2f85_action /= 255
                    # rq2f85_state = rq2f85_state / 0.8
                    # rq2f85_action = rq2f85_action / 255

                if claw_action.size == 0 and qiangnao_action.size == 0:
                    claw_action = rq2f85_action
                    claw_state  = rq2f85_state

                # =========================
                #5. Build the final state/action
                # =========================
                if kuavo.USE_LEJU_CLAW or kuavo.USE_QIANGNAO:
                    hand_type = "LEJU" if kuavo.USE_LEJU_CLAW else "QIANGNAO"
                    s_list, a_list = [], []

                    def get_hand_slice(hand_side):
                        s_slice = kuavo.SLICE_ROBOT[hand_side]

                        if hand_type == "LEJU":
                            c_slice = kuavo.SLICE_CLAW[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                claw_state[c_slice[0]:c_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                claw_action[c_slice[0]:c_slice[-1]]))
                        else:
                            d_slice = kuavo.SLICE_DEX[hand_side]
                            s = np.concatenate((state[s_slice[0]:s_slice[-1]],
                                                qiangnao_state[d_slice[0]:d_slice[-1]]))
                            a = np.concatenate((action[s_slice[0]:s_slice[-1]],
                                                qiangnao_action[d_slice[0]:d_slice[-1]]))
                        return s, a

                    if kuavo.CONTROL_HAND_SIDE in ("left", "both"):
                        s, a = get_hand_slice(0)
                        s_list.append(s)
                        a_list.append(a)

                    if kuavo.CONTROL_HAND_SIDE in ("right", "both"):
                        s, a = get_hand_slice(1)
                        s_list.append(s)
                        a_list.append(a)

                    final_state  = np.concatenate(s_list).astype(np.float32)
                    final_action = np.concatenate(a_list).astype(np.float32)
                else:
                    raise ValueError(f"eef type are not supported! ")

                # =========================
                # 6. cmd_pos_world & gap_flag
                # =========================
                if not kuavo.ONLY_HALF_UP_BODY:
                    cmd_pos_world = get_array(
                        "action.cmd_pos_world", np.float32
                    )
                    if cmd_pos_world.size == 0:
                        raise ValueError(f"kuavo.ONLY_HALF_UP_BODY is {kuavo.ONLY_HALF_UP_BODY}, but no action.cmd_pos_world found! ")
                    gap_flag = 1.0 if arm_traj.size and np.any(arm_traj == 999.0) else 0.0

                    final_action = np.concatenate(
                        [final_action, cmd_pos_world, np.array([gap_flag], np.float32)],
                        axis=0
                    )

                # =========================
                #7. Build frame
                # =========================
                frame = {
                    "observation.state": torch.from_numpy(final_state).type(torch.float32),
                    "action": torch.from_numpy(final_action).type(torch.float32),
                }

                for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                    cam_data = aligned_frame.get(cam_key)
                    if cam_data and "data" in cam_data:
                        img = cam_data["data"]
                        if "depth" in cam_key:
                            min_d, max_d = kuavo.DEPTH_RANGE
                            depth = np.clip(img, min_d, max_d)
                            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
                            depth_uint8 = (depth_norm * 255).astype(np.uint8)
                            frame[f"observation.{cam_key}"] = depth_uint8[..., None].repeat(3, -1)
                        else:
                            frame[f"observation.images.{cam_key}"] = img
                    else:
                        return
                
                if velocity is not None:
                    frame["observation.velocity"] = velocity
                if effort is not None:
                    frame["observation.effort"] = effort
                
                frames_buffer.append(frame)
                frame_count[0] += 1

            
            def on_chunk_done():
                """Callback after each chunk is processed: save and release memory"""
                if len(frames_buffer) == 0:
                    return
                
                #Add all cached frames to dataset
                for frame in frames_buffer:
                    frame["task"] = task
                    dataset.add_frame(frame)
                
                #Clear the buffer and release memory
                frames_buffer.clear()
                gc.collect()
                
                log_memory(f"After saving chunk (total frames: {frame_count[0]})")
            
            #Using chunked streaming
            started_at = time.perf_counter()
            bag_reader.process_rosbag_chunked(
                bag_file=str(ep_path),
                frame_callback=on_frame,
                chunk_size=chunk_size,
                save_callback=on_chunk_done
            )
            read_finished_at = time.perf_counter()
             
            #Process remaining frames
            if len(frames_buffer) > 0:
                for frame in frames_buffer:
                    dataset.add_frame(frame, task=task)
            flush_finished_at = time.perf_counter()
            dataset.save_episode()
            save_finished_at = time.perf_counter()

            # Rebuilding the full Hugging Face dataset after every episode makes
            # later episodes progressively slower. LeRobot finalizes it when the
            # dataset is loaded/exported, so keep conversion append-only here.
            log_print.info(
                "Episode %s timing: read_align_add_chunks=%.2fs, "
                "flush_remaining_add=%.2fs, save_episode=%.2fs, total=%.2fs, "
                "frames=%s",
                ep_idx,
                read_finished_at - started_at,
                flush_finished_at - read_finished_at,
                save_finished_at - flush_finished_at,
                save_finished_at - started_at,
                frame_count[0],
            )
            frames_buffer.clear()
            gc.collect()
            
            log_print.info(f"Episode {ep_idx} completed: {frame_count[0]} frames")
            
        except Exception as e:
            log_print.error(f"Error processing {ep_path}: {e}")
            import traceback
            traceback.print_exc()
            failed_bags.append(str(ep_path))
            continue
        
        log_memory("After episode")
        gc.collect()
    
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        log_print.error(f"{len(failed_bags)} failed bags written to error.txt")
    
    return dataset


def port_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
    chunk_size: int = 100,
):
    """
    Chunked streaming of rosbag to LeRobot format
    
    Args:
        raw_dir: rosbagDirectory
        repo_id: Output dataset ID
        task: Task description
        chunk_size: Number of frames per chunk (default 100)
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    if isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Requested {n} bags, but only {num_available_bags} available. Using all available bags.")
            n = num_available_bags
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
    
    dataset = create_empty_dataset_chunked(
        repo_id,
        robot_type="kuavo4pro",
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
        root=root,
    )
    
    dataset = populate_dataset_chunked(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
        chunk_size=chunk_size,
    )
    
    return dataset


@hydra.main(
    config_path="../configs/data/",
    config_name="KuavoRosbag2Lerobot",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    Chunked streaming conversion portal
    
    How to use:
        python CvtRosbag2Lerobot_chunked.py \
            rosbag.rosbag_dir=/path/to/rosbag \
            rosbag.lerobot_dir=/path/to/output \
            rosbag.chunk_size=100
    """
    setup_logging()  # set logger 

    global DEFAULT_JOINT_NAMES_LIST
    kuavo.init_parameters(cfg)

    n = cfg.rosbag.num_used
    raw_dir = cfg.rosbag.rosbag_dir
    version = cfg.rosbag.lerobot_dir

    task_name = os.path.basename(raw_dir)
    repo_id = f'lerobot/{task_name}'
    lerobot_dir = os.path.join(raw_dir,"../",version,"lerobot")
    if os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)
    
    chunk_size = cfg.rosbag.get("chunk_size", 100)
    
    log_print.info(f"=== Chunked Streaming Rosbag Converter ===")
    log_print.info(f"Rosbag dir: {raw_dir}")
    log_print.info(f"Output dir: {lerobot_dir}")
    log_print.info(f"Chunk size: {chunk_size}")

    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    # if kuavo.ONLY_HALF_UP_BODY:
    if kuavo.USE_LEJU_CLAW:
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm), 
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
            ]
    elif kuavo.USE_QIANGNAO:  
        DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]               
        arm_slice = [
            (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm), 
            (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
            ]
    DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]  
    
    
    port_kuavo_rosbag_chunked(
        raw_dir=raw_dir,
        repo_id=repo_id,
        task=kuavo.TASK_DESCRIPTION,
        mode="video",
        root=lerobot_dir,
        n=n,
        chunk_size=chunk_size,
    )
    
    log_print.info("Conversion completed!")


if __name__ == "__main__":
    np.random.seed(42)
    main()




