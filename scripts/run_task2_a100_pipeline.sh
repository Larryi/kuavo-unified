#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Required secrets/locations are injected at runtime. Never put them in this file.
: "${DRY_RUN:=0}"
: "${DATASET_REPO:?Set DATASET_REPO to the exact private Hugging Face dataset}"
: "${MODEL_REPO:?Set MODEL_REPO to the exact private Hugging Face model target}"
: "${WORK_ROOT:=/workspace/kuavo_task2}"
: "${CODE_DIR:=${WORK_ROOT}/kuavo_ship_classic}"
: "${DATASET_ROOT:=${WORK_ROOT}/datasets/lerobot_task2_264}"
: "${CODEBASE_ARCHIVE_NAME:=kuavo_ship_classic.zip}"
: "${CODEBASE_SHA256:=}"
: "${GPU_IDS:=0,1,2,3}"
: "${GPU_COUNT:=4}"
: "${PER_GPU_BATCH:=64}"
: "${GRAD_ACCUM_STEPS:=1}"
: "${TARGET_TRAIN_SAMPLES:=15360000}"
: "${DATALOADER_WORKERS:=6}"
: "${HF_DOWNLOAD_WORKERS:=16}"
: "${TENSORBOARD_PORT:=6006}"
: "${TENSORBOARD_HOST:=0.0.0.0}"
: "${SERVERCHAN_SENDKEY:=}"
: "${SERVERCHAN_URL:=}"
: "${RUN_TAG:=a100x${GPU_COUNT}_bs${PER_GPU_BATCH}_$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${WORK_ROOT}/logs/${RUN_TAG}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
TRAIN_LOG="${LOG_DIR}/train.log"
TB_LOG="${LOG_DIR}/tensorboard.log"
OUTPUT_BASE="${WORK_ROOT}/outputs/train/r2/dp_a100_rgb"
RUN_DIR="${OUTPUT_BASE}/run_${RUN_TAG}"
MODEL_STAGE="${WORK_ROOT}/model_stage/${RUN_TAG}"
TB_PID=""
PIPELINE_PHASE="bootstrap"
FAIL_NOTIFIED=0

serverchan_url() {
    if [[ -n "${SERVERCHAN_URL}" ]]; then
        printf '%s' "${SERVERCHAN_URL}"
    elif [[ "${SERVERCHAN_SENDKEY}" =~ ^sctp([0-9]+)t ]]; then
        printf 'https://%s.push.ft07.com/send/%s.send' "${BASH_REMATCH[1]}" "${SERVERCHAN_SENDKEY}"
    elif [[ -n "${SERVERCHAN_SENDKEY}" ]]; then
        printf 'https://sctapi.ftqq.com/%s.send' "${SERVERCHAN_SENDKEY}"
    fi
}

notify() {
    local text="$1"
    local desp="${2:-}"
    local url
    url="$(serverchan_url)"
    if [[ -z "${url}" ]]; then
        echo "[notify disabled] ${text}: ${desp}"
        return 0
    fi
    curl --fail --silent --show-error --max-time 20 \
        --retry 3 --retry-delay 2 \
        --request POST "${url}" \
        --header 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode "text=${text}" \
        --data-urlencode "desp=${desp}" >/dev/null || {
            echo "ServerChan notification failed: ${text}" >&2
            return 0
        }
}

retry() {
    local attempts="$1"
    shift
    local count=1 rc=0
    while true; do
        if "$@"; then
            return 0
        else
            rc=$?
        fi
        if (( count >= attempts )); then
            return "${rc}"
        fi
        echo "Command failed (attempt ${count}/${attempts}); retrying in $((count * 5))s: $*" >&2
        sleep $((count * 5))
        count=$((count + 1))
    done
}

tail_for_notification() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        tail -n 40 "${path}"
    fi
}

cleanup() {
    if [[ -n "${TB_PID}" ]] && kill -0 "${TB_PID}" 2>/dev/null; then
        kill "${TB_PID}" 2>/dev/null || true
        wait "${TB_PID}" 2>/dev/null || true
    fi
}

on_exit() {
    local rc=$?
    cleanup
    if (( rc != 0 )) && (( FAIL_NOTIFIED == 0 )); then
        notify "TASK2流水线异常退出" $'阶段：'"${PIPELINE_PHASE}"$'\n退出码：'"${rc}"$'\n主机：'"$(hostname)"$'\n日志尾部：\n\n'"$(tail_for_notification "${PIPELINE_LOG}")"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_positive_int() {
    local name="$1" value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
}

require_positive_int GPU_COUNT "${GPU_COUNT}"
require_positive_int PER_GPU_BATCH "${PER_GPU_BATCH}"
require_positive_int GRAD_ACCUM_STEPS "${GRAD_ACCUM_STEPS}"
require_positive_int TARGET_TRAIN_SAMPLES "${TARGET_TRAIN_SAMPLES}"
require_positive_int DATALOADER_WORKERS "${DATALOADER_WORKERS}"
require_positive_int HF_DOWNLOAD_WORKERS "${HF_DOWNLOAD_WORKERS}"

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if (( ${#gpu_array[@]} != GPU_COUNT )); then
    echo "GPU_IDS=${GPU_IDS} contains ${#gpu_array[@]} IDs but GPU_COUNT=${GPU_COUNT}" >&2
    exit 2
fi

GLOBAL_BATCH=$((PER_GPU_BATCH * GPU_COUNT * GRAD_ACCUM_STEPS))
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-$(((TARGET_TRAIN_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH))}"
WARMUP_STEPS="${WARMUP_STEPS:-$(((256000 + GLOBAL_BATCH - 1) / GLOBAL_BATCH))}"
require_positive_int MAX_TRAINING_STEPS "${MAX_TRAINING_STEPS}"
require_positive_int WARMUP_STEPS "${WARMUP_STEPS}"

echo "Run: ${RUN_TAG}"
echo "GPUs: ${GPU_IDS}; per-GPU batch=${PER_GPU_BATCH}; global batch=${GLOBAL_BATCH}"
echo "Training steps=${MAX_TRAINING_STEPS}; warmup=${WARMUP_STEPS}; target samples=${TARGET_TRAIN_SAMPLES}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dataset: ${DATASET_REPO} -> ${DATASET_ROOT}"
    echo "Model: ${MODEL_REPO}"
    echo "Code: ${CODE_DIR}"
    echo "Dry run complete; no token, download, training, or upload was attempted."
    exit 0
fi

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with dataset read and model write access}"
export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${WORK_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

PIPELINE_PHASE="install uv and gdown"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
retry 3 python -m pip install --upgrade uv gdown

PIPELINE_PHASE="download codebase"
if [[ ! -f "${CODE_DIR}/kuavo_train/train_policy_with_accelerate.py" ]]; then
    : "${CODEBASE_GDOWN_URL:?Set CODEBASE_GDOWN_URL when CODE_DIR is absent}"
    : "${CODEBASE_SHA256:?Set CODEBASE_SHA256 for the codebase archive}"
    download_root="${WORK_ROOT}/code_download"
    extract_root="${download_root}/extracted"
    rm -rf "${extract_root}"
    mkdir -p "${download_root}" "${extract_root}"
    archive="${download_root}/${CODEBASE_ARCHIVE_NAME}"
    retry 3 gdown "${CODEBASE_GDOWN_URL}" -O "${archive}"
    printf '%s  %s\n' "${CODEBASE_SHA256}" "${archive}" | sha256sum --check -
    ARCHIVE_PATH="${archive}" EXTRACT_ROOT="${extract_root}" python - <<'PY'
import os
import shutil
shutil.unpack_archive(os.environ["ARCHIVE_PATH"], os.environ["EXTRACT_ROOT"])
PY
    train_entry="$(find "${extract_root}" -type f -path '*/kuavo_train/train_policy_with_accelerate.py' -print -quit)"
    if [[ -z "${train_entry}" ]]; then
        echo "Downloaded codebase does not contain kuavo_train/train_policy_with_accelerate.py" >&2
        exit 3
    fi
    source_root="$(dirname "$(dirname "${train_entry}")")"
    mkdir -p "${CODE_DIR}"
    cp -a "${source_root}/." "${CODE_DIR}/"
fi

cd "${CODE_DIR}"

PIPELINE_PHASE="install training dependencies"
retry 3 uv pip install --python "$(command -v python)" -r requirements_train_cloud.txt
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
python - <<'PY'
import accelerate
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} accelerate={accelerate.__version__}")
assert torch.cuda.is_available(), "CUDA is unavailable"
PY

available_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( available_gpus != GPU_COUNT )); then
    echo "CUDA_VISIBLE_DEVICES=${GPU_IDS} selects ${GPU_COUNT} GPUs but PyTorch sees ${available_gpus}" >&2
    exit 4
fi

PIPELINE_PHASE="authenticate Hugging Face"
retry 3 hf auth whoami

PIPELINE_PHASE="download private dataset"
notify "TASK2数据下载开始" $'数据集：'"${DATASET_REPO}"$'\n主机：'"$(hostname)"
mkdir -p "${DATASET_ROOT}"
retry 3 hf download "${DATASET_REPO}" \
    --repo-type dataset \
    --local-dir "${DATASET_ROOT}" \
    --max-workers "${HF_DOWNLOAD_WORKERS}"

DATASET_ROOT="${DATASET_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0", info
assert info["total_episodes"] == 264, info["total_episodes"]
assert info["total_frames"] == 50042, info["total_frames"]
assert info["fps"] == 10, info["fps"]
expected = {
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
}
assert expected.issubset(info["features"]), info["features"].keys()
assert "observation.depth" not in " ".join(info["features"]), info["features"].keys()
print("Dataset validation passed:", info["total_episodes"], info["total_frames"])
PY
notify "TASK2数据下载完成" $'目录：'"${DATASET_ROOT}"$'\nEpisode：264\nFrames：50042'

PIPELINE_PHASE="start TensorBoard"
mkdir -p "${OUTPUT_BASE}" "${RUN_DIR}"
tensorboard --logdir "${OUTPUT_BASE}" \
    --host "${TENSORBOARD_HOST}" \
    --port "${TENSORBOARD_PORT}" >"${TB_LOG}" 2>&1 &
TB_PID=$!
sleep 2
if ! kill -0 "${TB_PID}" 2>/dev/null; then
    echo "TensorBoard failed to start" >&2
    tail -n 50 "${TB_LOG}" >&2 || true
    exit 5
fi
TB_URL="${TENSORBOARD_PUBLIC_URL:-http://$(hostname -I 2>/dev/null | awk '{print $1}'):${TENSORBOARD_PORT}}"

export TASK2_HF_REPO="${DATASET_REPO}"
export TASK2_DATASET_ROOT="${DATASET_ROOT}"

launch_args=(--num_processes "${GPU_COUNT}" --mixed_precision bf16)
if (( GPU_COUNT > 1 )); then
    launch_args=(--multi_gpu "${launch_args[@]}")
fi

common_overrides=(
    --config-name=dp_r2_h100
    "timestamp=${RUN_TAG}"
    "repoid=${DATASET_REPO}"
    "root=${DATASET_ROOT}"
    "training.output_directory=${OUTPUT_BASE}"
    "training.batch_size=${PER_GPU_BATCH}"
    "training.accumulation_steps=${GRAD_ACCUM_STEPS}"
    "training.max_training_step=${MAX_TRAINING_STEPS}"
    "training.num_workers=${DATALOADER_WORKERS}"
    "policy.scheduler_warmup_steps=${WARMUP_STEPS}"
)

python kuavo_train/train_policy_with_accelerate.py \
    "${common_overrides[@]}" --cfg job >"${RUN_DIR}/resolved_training_config.yaml"

PIPELINE_PHASE="train"
notify "TASK2训练开始" $'Run：'"${RUN_TAG}"$'\n主机：'"$(hostname)"$'\nGPU：'"${GPU_IDS}"$'\n单卡BS：'"${PER_GPU_BATCH}"$'\n全局BS：'"${GLOBAL_BATCH}"$'\nSteps：'"${MAX_TRAINING_STEPS}"$'\nTensorBoard：'"${TB_URL}"

set +e
accelerate launch "${launch_args[@]}" \
    kuavo_train/train_policy_with_accelerate.py \
    "${common_overrides[@]}" 2>&1 | tee "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e
if (( train_rc != 0 )); then
    FAIL_NOTIFIED=1
    notify "TASK2训练异常" $'Run：'"${RUN_TAG}"$'\n退出码：'"${train_rc}"$'\n日志尾部：\n\n'"$(tail_for_notification "${TRAIN_LOG}")"
    exit "${train_rc}"
fi

for required in \
    "${RUN_DIR}/epochlast/config.json" \
    "${RUN_DIR}/epochlast/model.safetensors" \
    "${RUN_DIR}/epochbest/config.json" \
    "${RUN_DIR}/epochbest/model.safetensors" \
    "${RUN_DIR}/policy_preprocessor.json" \
    "${RUN_DIR}/policy_postprocessor.json"; do
    [[ -s "${required}" ]] || { echo "Missing training artifact: ${required}" >&2; exit 6; }
done
notify "TASK2训练完成" $'Run：'"${RUN_TAG}"$'\n目录：'"${RUN_DIR}"$'\n即将上传私有模型：'"${MODEL_REPO}"

PIPELINE_PHASE="stage model"
rm -rf "${MODEL_STAGE}"
mkdir -p "${MODEL_STAGE}"
cp -a "${RUN_DIR}/epochlast" "${MODEL_STAGE}/"
cp -a "${RUN_DIR}/epochbest" "${MODEL_STAGE}/"
cp -a "${RUN_DIR}"/policy_* "${MODEL_STAGE}/"
cp -a "${RUN_DIR}/resolved_training_config.yaml" "${MODEL_STAGE}/"
cat >"${MODEL_STAGE}/README.md" <<EOF
---
library_name: lerobot
tags:
  - robotics
  - diffusion-policy
  - kuavo
---

# Kuavo TASK2 Diffusion Policy

- Dataset: `${DATASET_REPO}`
- Run: `${RUN_TAG}`
- GPUs: `${GPU_COUNT} x A100`
- Per-GPU batch: `${PER_GPU_BATCH}`
- Global batch: `${GLOBAL_BATCH}`
- Optimizer steps: `${MAX_TRAINING_STEPS}`
- Mixed precision: `bf16`

Use `epochlast/` for the final optimizer state policy or `epochbest/` for the
lowest training-loss policy. Keep the parent processor files beside them.
EOF

PIPELINE_PHASE="create private model repository"
notify "TASK2模型上传开始" $'Model：'"${MODEL_REPO}"$'\nRun：'"${RUN_TAG}"
MODEL_REPO="${MODEL_REPO}" python - <<'PY'
import os
from huggingface_hub import HfApi
repo_id = os.environ["MODEL_REPO"]
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
assert api.model_info(repo_id).private, f"Model repo is not private: {repo_id}"
print(f"Private model repository ready: {repo_id}")
PY

PIPELINE_PHASE="upload model"
set +e
retry 3 hf upload "${MODEL_REPO}" "${MODEL_STAGE}" . \
    --commit-message "train: upload TASK2 ${RUN_TAG}"
upload_rc=$?
set -e
if (( upload_rc != 0 )); then
    FAIL_NOTIFIED=1
    notify "TASK2模型上传异常" $'Model：'"${MODEL_REPO}"$'\n退出码：'"${upload_rc}"$'\n阶段：上传模型'
    exit "${upload_rc}"
fi

MODEL_REPO="${MODEL_REPO}" python - <<'PY'
import os
from huggingface_hub import HfApi
info = HfApi().model_info(os.environ["MODEL_REPO"])
assert info.private, "Uploaded model repository unexpectedly became public"
siblings = {item.rfilename for item in info.siblings}
required = {
    "epochlast/config.json", "epochlast/model.safetensors",
    "epochbest/config.json", "epochbest/model.safetensors",
    "policy_preprocessor.json", "policy_postprocessor.json",
}
missing = sorted(required - siblings)
assert not missing, f"Upload verification missing files: {missing}"
print("Upload verification passed")
PY

PIPELINE_PHASE="complete"
notify "TASK2模型上传完成" $'Model：https://huggingface.co/'"${MODEL_REPO}"$'\nRun：'"${RUN_TAG}"$'\n状态：private\n包含：epochlast、epochbest、pre/post processors'
echo "Pipeline complete: https://huggingface.co/${MODEL_REPO}"
