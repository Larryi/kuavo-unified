# TASK1-345 LingBot full fine-tuning pipeline

The entry point is `scripts/run_task1_lingbot_full_pipeline.sh`. It downloads
the private LeRobot dataset and both model asset repositories, computes the
LingBot normalization file from the transformed low-dimensional stream,
launches four-process FSDP2 training, retains one resumable DCP checkpoint,
exports one final BF16 Hugging Face checkpoint, and uploads it to a private
model repository.

## Build the Kuavo cloud bundle

```bash
git submodule update --init --recursive third_party/lingbot-vla
bash scripts/build_task1_lingbot_cloud_bundle.sh
```

This produces `kuavo_task1_lingbot_cloud.zip` and `lingbot-vla-cloud.zip`.
Upload both to Google Drive. The archives intentionally exclude Docker assets,
deployment weights, caches, Git metadata, and local Python environments.
The bundle command refuses a LeRobot or LingBot-v1 checkout that differs from
the repository's reviewed pins, or a LingBot-v1 checkout with missing nested
submodules.

## Required environment

Use a CUDA 12.x image with Python 3.10 or 3.11 and a CUDA-enabled PyTorch 2.7.x
already installed. Export secrets only in the rented host's shell:

```bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export SERVERCHAN_SENDKEY=sctp...
export VAST_API_KEY=...
export CODEBASE_GDOWN_URL='https://drive.google.com/file/d/.../view'
export CODEBASE_SHA256='...'
export LINGBOT_CODE_GDOWN_URL='https://drive.google.com/file/d/.../view'
export LINGBOT_CODE_SHA256='...'
export DATASET_REPO='your-org/kuavo-task1-345'
export MODEL_REPO='your-org/kuavo-task1-345-lingbot-full'
# Optional model-source overrides; these are the script defaults.
export LINGBOT_MODEL_REPO='robbyant/lingbot-vla-4b'
export QWEN_MODEL_REPO='Qwen/Qwen2.5-VL-3B-Instruct'
```

`DATASET_REPO` and `MODEL_REPO` have no guessed account defaults. The two
model-source repository IDs are defaults, not fixed constants; override either
when testing a mirror, private copy, or another reviewed model revision.

Before providing credentials or creating the work directory, inspect the
effective non-secret settings:

```bash
DRY_RUN=1 bash scripts/run_task1_lingbot_full_pipeline.sh
```

The real run verifies both downloaded code archives against the required
SHA-256 values before extraction.

The training environment intentionally installs the bundled LeRobot editable
package with `--no-deps`. Its `rerun-sdk` visualization dependency is not used
for training and otherwise conflicts with dependency resolution. Runtime
packages are pinned separately to the locally validated NumPy 2.2.6 stack.
The bootstrap also installs the CUDA 12.6-compatible torchvision 0.22.1 and
torchdata 0.11.0 from the official PyTorch wheel index, plus pyserial and the
complete data/model import stack. A full training-entry import smoke test runs
before model and dataset downloads.

## Launch

Copy the bootstrap script itself to the rented host, then run:

```bash
chmod +x run_task1_lingbot_full_pipeline.sh
./run_task1_lingbot_full_pipeline.sh
```

The default run identity is stable (`task1_345_lingbot_full_v1`). Re-running
the command with the same `WORK_ROOT` and `RUN_ID` resumes the newest complete
checkpoint instead of creating a new timestamped run.

Important overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `MICRO_BATCH_SIZE` | 4 | Per-GPU micro batch; H100 default |
| `GRAD_ACCUM_STEPS` | 2 | Gradient accumulation; global batch remains 32 on four GPUs |
| `NUM_EPOCHS` | 10 | Total epochs, including resumed epochs |
| `LEARNING_RATE` | 2e-5 | Full-model peak learning rate |
| `SAVE_STEPS` | 500 | Maximum work lost after interruption |
| `KEEP_LAST_CHECKPOINTS` | 1 | Complete DCP checkpoints retained |
| `USE_COMPILE` | true | Enabled by default; set false only for diagnosis |
| `MIN_FREE_GB` | 120 | Startup disk-space guard |
| `AUTO_STOP_INSTANCE` | 1 | Stop the Vast instance after any pipeline exit |
| `VAST_INSTANCE_ID` | auto | Override the ID inferred from hostname `C.<id>` |
| `STOP_DELAY_SECONDS` | 15 | Delay after the final ServerChan message |

With automatic stopping enabled, both successful and failed pipeline exits
send their ServerChan status first. The script then invokes `vastai stop
instance`; a stop-command failure generates another notification. `stop`
retains the instance disk and may continue to incur storage charges. Use the
Vast console to destroy the instance only after deciding that local recovery
files are no longer needed.

For four H100 80/96 GB GPUs, the defaults use `MICRO_BATCH_SIZE=4` and
`GRAD_ACCUM_STEPS=2` to preserve global batch 32. For four A100 40 GB GPUs,
keep the defaults until a short eager run confirms memory headroom.

Video decoding uses PyAV. Bootstrap removes preinstalled TorchCodec packages
so LeRobot cannot select a CUDA TorchCodec wheel whose optional NPP libraries
are absent from the rented image.

## Norm contract

The pipeline requires raw state and action dimensions of 8 and accepts exactly
the two training cameras (`head_cam_h`, `wrist_cam_r`). With the pinned
LingBot-v1 API it computes statistics from the transformed low-dimensional
state/action stream without decoding video. This is necessary because the
right-arm robot mapping places joints in the unified action layout and converts
joint commands to deltas before normalization; raw marginal metadata cannot
derive those delta statistics exactly. The generated file is copied into the
final model repository as `norm_stats.json` together with the resolved
`lingbotvla_cli.yaml`.
