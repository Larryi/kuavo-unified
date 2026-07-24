# TASK2 one-command A100 training pipeline

The standalone bootstrap script is
[`scripts/run_task2_a100_pipeline.sh`](../scripts/run_task2_a100_pipeline.sh).
It is designed for a Python 3.12 virtual environment and defaults to four A100
40 GB GPUs, per-GPU batch 64, global batch 256, and 60,000 optimizer updates.

## Required environment variables

```bash
# Google Drive URL for a zip archive containing this repository. It is only
# required when CODE_DIR does not already contain the codebase.
export CODEBASE_GDOWN_URL="https://drive.google.com/file/d/.../view"
export CODEBASE_ARCHIVE_NAME="kuavo_ship_classic.zip"
export CODEBASE_SHA256="<sha256-of-kuavo_ship_classic.zip>"

# A Hugging Face token with read access to the private dataset and write access
# to the target user/organization. Do not put this token in the script.
export HF_TOKEN="hf_..."
export DATASET_REPO="<hf-user-or-org>/kuavo-task2-264"
export MODEL_REPO="<hf-user-or-org>/kuavo-task2-dp-a100x4"

# ServerChan: choose either the SendKey or the complete safe-mode URL.
export SERVERCHAN_SENDKEY="sctp..."
# export SERVERCHAN_URL="https://<number>.push.ft07.com/send/<sendkey>.send"
```

`DATASET_REPO` and `MODEL_REPO` are required explicitly. The pipeline refuses
to guess an account or upload target, and forces the model repository to private
before every upload.

## Recommended launch

Copy the bootstrap script onto the rented machine, activate the provider's
virtual environment, then run:

```bash
chmod +x run_task2_a100_pipeline.sh

export HF_TOKEN="hf_..."
export DATASET_REPO="<hf-user-or-org>/kuavo-task2-264"
export MODEL_REPO="<hf-user-or-org>/kuavo-task2-dp-a100x4"
export CODEBASE_GDOWN_URL="https://drive.google.com/file/d/.../view"
export CODEBASE_SHA256="<sha256-of-kuavo_ship_classic.zip>"
export SERVERCHAN_SENDKEY="sctp..."

export WORK_ROOT=/workspace/kuavo_task2
export TENSORBOARD_PUBLIC_URL="http://<rented-host-ip>:6006"

./run_task2_a100_pipeline.sh
```

Make sure TCP port 6006 is exposed by the rental platform. If it is not, the
training still runs and TensorBoard remains available through an SSH tunnel:

```bash
ssh -L 6006:127.0.0.1:6006 user@rented-host
```

Before providing any secret, validate the calculated paths and training size:

```bash
DRY_RUN=1 \
DATASET_REPO="<hf-user-or-org>/kuavo-task2-264" \
MODEL_REPO="<hf-user-or-org>/kuavo-task2-dp-a100x4" \
CODE_DIR="$PWD" \
WORK_ROOT=/workspace/kuavo_task2 \
./scripts/run_task2_a100_pipeline.sh
```

Dry-run mode does not create directories, download data, start training, or
contact Hugging Face.

## Defaults and overrides

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_IDS` | `0,1,2,3` | CUDA devices |
| `GPU_COUNT` | `4` | Accelerate/DDP process count |
| `PER_GPU_BATCH` | `64` | Per-GPU batch |
| `GRAD_ACCUM_STEPS` | `1` | Gradient accumulation |
| `TARGET_TRAIN_SAMPLES` | `15360000` | Constant data exposure |
| `DATALOADER_WORKERS` | `6` | Workers per GPU process |
| `DATASET_REPO` | required | Exact private HF dataset |
| `MODEL_REPO` | required | Exact private HF model target |
| `TENSORBOARD_PORT` | `6006` | TensorBoard port |

`MAX_TRAINING_STEPS` is automatically calculated from total target samples and
global batch. With the defaults:

```text
global batch = 64 x 4 = 256
steps = 15,360,000 / 256 = 60,000
warmup = 256,000 / 256 = 1,000 steps
```

For a conservative first smoke test:

```bash
export TARGET_TRAIN_SAMPLES=51200   # 200 updates at global batch 256
export MODEL_REPO="<hf-user-or-org>/kuavo-task2-dp-a100x4-smoke"
./run_task2_a100_pipeline.sh
```

## Pipeline stages

1. Install `uv` and `gdown` with the active venv's pip.
2. Download/extract the codebase when it is not already present.
3. Install the minimal cloud-training dependencies with `uv`.
4. Validate the HF token and download the private LeRobot dataset resumably.
5. Validate dataset version, frame count, episodes, cameras, FPS, and absence of depth.
6. Start TensorBoard and send the training-start notification.
7. Launch BF16 training through Accelerate/DDP.
8. Save both `epochlast` and `epochbest` deployable policies.
9. Create or force the HF Model repository to private.
10. Upload and remotely verify weights plus pre/post-processors.

An `EXIT` trap stops TensorBoard and sends a failure notification when any
stage exits abnormally. Training and upload have separate start/success/failure
notifications. Network-facing download/upload operations are retried.
