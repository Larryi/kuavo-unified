# TASK2 DP training on H100

This workflow trains the repaired RGB-only TASK2 LeRobot dataset with a fixed
effective global batch of 128 on one, two, or four H100 GPUs.

## 1. Download the dataset

```bash
conda activate kdc_dev
hf auth login

export HF_USER="your-hf-user-or-org"
export TASK2_HF_REPO="$HF_USER/kuavo-task2-264"
export TASK2_DATASET_ROOT=/workspace/datasets/lerobot_task2_264

hf download "$TASK2_HF_REPO" \
  --repo-type dataset \
  --local-dir "$TASK2_DATASET_ROOT" \
  --max-workers 16
```

Validate the downloaded LeRobot v3 metadata before allocating GPUs:

```bash
python - <<'PY'
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = os.environ["TASK2_DATASET_ROOT"]
repo_id = os.environ["TASK2_HF_REPO"]
dataset = LeRobotDataset(repo_id=repo_id, root=root)
assert dataset.meta.total_episodes == 264
assert len(dataset) == 50042
assert dataset.meta.fps == 10
print(len(dataset), dataset.meta.total_episodes, dataset.meta.camera_keys)
PY
```

## 2. Run a short smoke test

```bash
export RUN_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-name=dp_r2_h100 \
  timestamp="$RUN_TAG" \
  training.batch_size=8 \
  training.max_training_step=100 \
  training.save_freq_epoch=1
```

Keep `training.torch_compile.enabled=false` for this first test. Enable it only
after eager mode is stable, and compare throughput after compilation warm-up.
For batch-size selection, repeat a timed 300-step smoke test with per-GPU batch
sizes 64, 96, 128, 160, and optionally 192. Use the largest batch that still
improves measured samples/s while keeping at least 10 GB of memory headroom.

## 3. Full training

Choose exactly one launch command. `training.batch_size` is per GPU, so every
variant below uses an effective global batch of 128.

### One H100

```bash
export RUN_TAG="h100x1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-name=dp_r2_h100 \
  timestamp="$RUN_TAG" \
  training.batch_size=128
```

### Two H100s

```bash
export RUN_TAG="h100x2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision bf16 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-name=dp_r2_h100 \
  timestamp="$RUN_TAG" \
  training.batch_size=64
```

### Four H100s

```bash
export RUN_TAG="h100x4_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-name=dp_r2_h100 \
  timestamp="$RUN_TAG" \
  training.batch_size=32
```

The run directory is:

```bash
export RUN_DIR="outputs/train/r2/dp_h100_rgb/run_$RUN_TAG"
```

The default run stops at 120,000 optimizer updates. With 48,194 usable window
starts after episode-tail dropping, this is roughly 319 nominal dataset epochs.

### Runtime estimate

The exact runtime is dominated by H.264 decoding, CPU count, and local NVMe
throughput, not H100 peak FLOPS alone. With effective global batch 128, the run
consumes 15.36 million training samples. A conservative starting estimate is:

| GPUs | Per-GPU batch | Expected aggregate samples/s | One epoch | 120k steps |
|---:|---:|---:|---:|---:|
| 1 | 128 | 140-240 | 3.3-5.7 min | 17.8-30.5 h |
| 2 | 64 | 240-400 | 2.0-3.3 min | 10.7-17.8 h |
| 4 | 32 | 380-650 | 1.2-2.1 min | 6.6-11.2 h |

Replace the estimate after the smoke test with:

```text
hours_for_120k = 120000 * 128 / samples_per_sec / 3600
minutes_per_epoch = 48194 / samples_per_sec / 60
```

If CPU/video decode saturates, adding GPUs will not approach linear scaling.
Increase `training.num_workers` only while aggregate samples/s still improves;
the configured value is per process, so four GPUs with 8 workers use 32 workers.

To resume the same run:

```bash
# One-GPU example; retain the original GPU count and per-GPU batch size.
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  kuavo_train/train_policy_with_accelerate.py \
  --config-name=dp_r2_h100 \
  timestamp="$RUN_TAG" \
  training.batch_size=128 \
  training.resume=true \
  training.resume_timestamp="run_$RUN_TAG"
```

## 4. Upload the deployable model

The policy loader expects `epochbest/` and the preprocessing files in its
parent directory, so upload both rather than uploading only `model.safetensors`.

```bash
export MODEL_REPO="$HF_USER/kuavo-task2-dp-h100"

hf repo create "$MODEL_REPO" --private --exist-ok

HF_XET_HIGH_PERFORMANCE=1 hf upload \
  "$MODEL_REPO" \
  "$RUN_DIR" \
  . \
  --include "epochbest/**" "policy_*" \
  --commit-message "train: add TASK2 DP H100 checkpoint"
```

Download it later with:

```bash
hf download "$MODEL_REPO" \
  --local-dir /workspace/models/kuavo-task2-dp-h100 \
  --max-workers 16
```

Use `/workspace/models/kuavo-task2-dp-h100/epochbest` as the policy checkpoint;
keep its parent directory intact so the policy pre/post-processors are found.
