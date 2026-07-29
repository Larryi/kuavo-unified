#!/usr/bin/env bash
# Unified LingBot-v1 Task1/Task2 training, export, and upload pipeline.
set -Eeuo pipefail
umask 077

: "${DRY_RUN:=0}"
: "${TRAINING_TASK:=task1}"
: "${LINGBOT_MODEL_REPO:=robbyant/lingbot-vla-4b}"
: "${QWEN_MODEL_REPO:=Qwen/Qwen2.5-VL-3B-Instruct}"
: "${GPU_IDS:=0,1,2,3}"
: "${GPU_COUNT:=4}"
: "${MICRO_BATCH_SIZE:=4}"
: "${GRAD_ACCUM_STEPS:=2}"
: "${NUM_EPOCHS:=10}"
: "${LEARNING_RATE:=2e-5}"
: "${MIN_LEARNING_RATE:=1e-6}"
: "${WARMUP_RATIO:=0.03}"
: "${SAVE_STEPS:=500}"
: "${KEEP_LAST_CHECKPOINTS:=1}"
: "${DATALOADER_WORKERS:=8}"
: "${DATALOADER_PREFETCH:=4}"
: "${MIN_FREE_GB:=120}"
: "${HEARTBEAT_SECONDS:=600}"
: "${USE_COMPILE:=true}"
: "${FLASH_ATTN_VERSION:=2.8.3}"
: "${FLASH_ATTN_WHEEL_URL:=}"
: "${WANDB_PROJECT:=LingBotVLA-Kuavo}"
: "${WANDB_API_KEY:=}"
: "${SERVERCHAN_SENDKEY:=}"
: "${SERVERCHAN_URL:=}"
: "${DELETE_LOCAL_DCP_AFTER_UPLOAD:=0}"
: "${AUTO_STOP_INSTANCE:=1}"
: "${VAST_INSTANCE_ID:=}"
: "${VAST_API_KEY:=}"
: "${STOP_DELAY_SECONDS:=15}"
: "${RESUME:=0}"
: "${RESUME_REPO:=}"

case "${TRAINING_TASK}" in
    task1)
        : "${WORK_ROOT:=/workspace/kuavo_task1_lingbot}"
        : "${RUN_ID:=task1_345_lingbot_full_v1}"
        : "${LINGBOT_V1_TRAIN_CONFIG:=configs/policy/lingbot/task1_345_full.yaml}"
        : "${LINGBOT_V1_DATA_NAME:=kuavo_v1_right_arm}"
        DATASET_DIR_NAME="lerobot_task1_345"
        NORM_FILE_NAME="task1_345.json"
        TASK_LABEL="TASK1"
        EXPECTED_EPISODES=345
        EXPECTED_FRAMES=81142
        EXPECTED_ACTION_DIM=8
        EXPECTED_CAMERAS="observation.images.head_cam_h,observation.images.wrist_cam_r"
        ;;
    task2)
        : "${WORK_ROOT:=/workspace/kuavo_task2_lingbot}"
        : "${RUN_ID:=task2_264_lingbot_full_v1}"
        : "${LINGBOT_V1_TRAIN_CONFIG:=configs/policy/lingbot/task2_264_full.yaml}"
        : "${LINGBOT_V1_DATA_NAME:=kuavo_v1_bimanual}"
        DATASET_DIR_NAME="lerobot_task2_264"
        NORM_FILE_NAME="task2_264.json"
        TASK_LABEL="TASK2"
        EXPECTED_EPISODES=264
        EXPECTED_FRAMES=50042
        EXPECTED_ACTION_DIM=16
        EXPECTED_CAMERAS="observation.images.head_cam_h,observation.images.wrist_cam_l,observation.images.wrist_cam_r"
        ;;
    *)
        echo "TRAINING_TASK must be task1 or task2; got ${TRAINING_TASK}" >&2
        exit 2
        ;;
esac
export EXPECTED_EPISODES EXPECTED_FRAMES EXPECTED_ACTION_DIM EXPECTED_CAMERAS

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'LingBot-v1 pipeline dry-run\n'
    printf '  task=%s\n  train_config=%s\n  data_name=%s\n' \
        "${TRAINING_TASK}" "${LINGBOT_V1_TRAIN_CONFIG}" "${LINGBOT_V1_DATA_NAME}"
    printf '  work_root=%s\n  run_id=%s\n  gpu_ids=%s\n  gpu_count=%s\n' \
        "${WORK_ROOT}" "${RUN_ID}" "${GPU_IDS}" "${GPU_COUNT}"
    for required in HF_TOKEN CODEBASE_GDOWN_URL CODEBASE_SHA256 \
        LINGBOT_CODE_GDOWN_URL LINGBOT_CODE_SHA256 DATASET_REPO MODEL_REPO; do
        if [[ -n "${!required:-}" ]]; then
            printf '  %s=set\n' "${required}"
        else
            printf '  %s=unset\n' "${required}"
        fi
    done
    exit 0
fi

: "${HF_TOKEN:?Set HF_TOKEN with private dataset read and model write access}"
: "${DATASET_REPO:?Set the private Hugging Face dataset repository ID}"
: "${MODEL_REPO:?Set the private Hugging Face destination model repository ID}"
if [[ "${RESUME}" == "1" ]]; then
    : "${RESUME_REPO:?Set RESUME_REPO to a model repository containing the full DCP run}"
fi

export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export GPU_COUNT FLASH_ATTN_VERSION
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_ROOT}/matplotlib_cache}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY WANDB_MODE=online
fi

: "${CODE_DIR:=${WORK_ROOT}/kuavo_unified_stack}"
: "${LINGBOT_ROOT:=${WORK_ROOT}/lingbot-vla}"
DATASET_ROOT="${WORK_ROOT}/datasets/${DATASET_DIR_NAME}"
LINGBOT_MODEL_ROOT="${WORK_ROOT}/models/LingBotVLA"
QWEN_MODEL_ROOT="${WORK_ROOT}/models/Qwen2.5_VL"
RUN_DIR="${WORK_ROOT}/runs/${RUN_ID}"
NORM_FILE="${RUN_DIR}/norm_stats/${NORM_FILE_NAME}"
FINAL_HF_DIR="${RUN_DIR}/final_hf_ckpt"
LOG_DIR="${WORK_ROOT}/logs/${RUN_ID}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
TRAIN_LOG="${LOG_DIR}/train.log"
TRAIN_CONFIG="${CODE_DIR}/${LINGBOT_V1_TRAIN_CONFIG}"
TRAIN_ENTRY="${CODE_DIR}/kuavo_train/lingbot/tasks/vla/train_lingbotvla.py"
HEARTBEAT_PID=""
PIPELINE_PHASE="bootstrap"
FAIL_NOTIFIED=0

mkdir -p "${LOG_DIR}" "${WORK_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

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
    local text="$1" desp="${2:-}" url
    url="$(serverchan_url)"
    if [[ -z "${url}" ]]; then
        echo "[notify disabled] ${text}: ${desp}"
        return 0
    fi
    curl --fail --silent --show-error --max-time 20 --retry 3 --retry-delay 2 \
        --request POST "${url}" \
        --header 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode "text=${text}" \
        --data-urlencode "desp=${desp}" >/dev/null || true
}

retry() {
    local attempts="$1"
    shift
    local count=1 rc=0
    while true; do
        if "$@"; then return 0; else rc=$?; fi
        if (( count >= attempts )); then return "${rc}"; fi
        sleep $((count * 10))
        count=$((count + 1))
    done
}

latest_checkpoint() {
    find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print 2>/dev/null \
        | sort -V | tail -n 1
}

cleanup() {
    if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        kill "${HEARTBEAT_PID}" 2>/dev/null || true
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
}

resolve_vast_instance_id() {
    local hostname_value
    if [[ -n "${VAST_INSTANCE_ID}" ]]; then
        printf '%s' "${VAST_INSTANCE_ID}"
        return 0
    fi
    hostname_value="$(hostname)"
    if [[ "${hostname_value}" =~ ^C\.([0-9]+)$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

stop_vast_instance() {
    local rc="$1" instance_id status
    [[ "${AUTO_STOP_INSTANCE}" == "1" ]] || return 0
    status="success"
    (( rc == 0 )) || status="failure(rc=${rc}, phase=${PIPELINE_PHASE})"

    if ! command -v vastai >/dev/null 2>&1; then
        notify "${TASK_LABEL} LingBot自动停机失败" $'原因：vastai CLI不可用\nRun：'"${RUN_ID}"$'\n状态：'"${status}"
        return 0
    fi
    if ! instance_id="$(resolve_vast_instance_id)"; then
        notify "${TASK_LABEL} LingBot自动停机失败" $'原因：无法识别Vast实例ID\nRun：'"${RUN_ID}"$'\n请设置VAST_INSTANCE_ID'
        return 0
    fi

    notify "${TASK_LABEL} LingBot流水线结束，准备停机" $'Run：'"${RUN_ID}"$'\n状态：'"${status}"$'\n实例：'"${instance_id}"$'\n等待：'"${STOP_DELAY_SECONDS}"'秒'
    sleep "${STOP_DELAY_SECONDS}"
    if ! vastai stop instance "${instance_id}"; then
        notify "${TASK_LABEL} LingBot自动停机失败" $'Run：'"${RUN_ID}"$'\n实例：'"${instance_id}"$'\n请手动停止实例'
    fi
}

on_exit() {
    local rc=$?
    trap - EXIT
    set +e
    cleanup
    if (( rc != 0 )) && (( FAIL_NOTIFIED == 0 )); then
        notify "${TASK_LABEL} LingBot流水线异常" $'阶段：'"${PIPELINE_PHASE}"$'\n退出码：'"${rc}"$'\nRun：'"${RUN_ID}"$'\n日志：\n'"$(tail -n 35 "${PIPELINE_LOG}" 2>/dev/null || true)"
    fi
    stop_vast_instance "${rc}"
    exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if (( ${#gpu_array[@]} != GPU_COUNT )); then
    echo "GPU_IDS contains ${#gpu_array[@]} devices, expected ${GPU_COUNT}" >&2
    exit 2
fi
for value in GPU_COUNT MICRO_BATCH_SIZE GRAD_ACCUM_STEPS NUM_EPOCHS SAVE_STEPS KEEP_LAST_CHECKPOINTS DATALOADER_WORKERS DATALOADER_PREFETCH; do
    [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "${value} must be a positive integer" >&2; exit 2; }
done
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * GPU_COUNT * GRAD_ACCUM_STEPS))

free_gb="$(df -Pk "${WORK_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')"
if (( free_gb < MIN_FREE_GB )); then
    echo "Only ${free_gb}GB free under ${WORK_ROOT}; require at least ${MIN_FREE_GB}GB" >&2
    exit 3
fi

PIPELINE_PHASE="bootstrap tools"
python -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
if [[ "${AUTO_STOP_INSTANCE}" == "1" ]]; then
    retry 3 python -m pip install --upgrade uv gdown vastai
    if [[ -n "${VAST_API_KEY}" ]]; then
        vastai set api-key "${VAST_API_KEY}"
    fi
    if ! resolve_vast_instance_id >/dev/null; then
        echo "Cannot infer Vast instance ID; set VAST_INSTANCE_ID" >&2
        exit 7
    fi
    if ! retry 3 vastai show instances >/dev/null; then
        echo "Vast CLI authentication failed; set VAST_API_KEY or run 'vastai set api-key'" >&2
        exit 7
    fi
else
    retry 3 python -m pip install --upgrade uv gdown
fi

download_and_extract() {
    local url="$1" checksum="$2" archive="$3" extract_root="$4" marker="$5"
    if [[ -e "${marker}" ]]; then return 0; fi
    mkdir -p "$(dirname "${archive}")" "${extract_root}"
    retry 3 gdown "${url}" -O "${archive}"
    printf '%s  %s\n' "${checksum}" "${archive}" | sha256sum --check -
    ARCHIVE_PATH="${archive}" EXTRACT_ROOT="${extract_root}" python - <<'PY'
import os, shutil
shutil.unpack_archive(os.environ["ARCHIVE_PATH"], os.environ["EXTRACT_ROOT"])
PY
}

PIPELINE_PHASE="download code"
if [[ ! -e "${CODE_DIR}/kuavo_train/lingbot/tasks/vla/train_lingbotvla.py" ]]; then
    : "${CODEBASE_GDOWN_URL:?Set CODEBASE_GDOWN_URL when the synchronized Kuavo source is absent}"
    : "${CODEBASE_SHA256:?Set CODEBASE_SHA256 when the synchronized Kuavo source is absent}"
    download_and_extract "${CODEBASE_GDOWN_URL}" "${CODEBASE_SHA256}" \
        "${WORK_ROOT}/downloads/kuavo.zip" "${WORK_ROOT}/code_extract" \
        "${CODE_DIR}/kuavo_train/lingbot/tasks/vla/train_lingbotvla.py"
fi
if [[ ! -e "${CODE_DIR}/kuavo_train/lingbot/tasks/vla/train_lingbotvla.py" ]]; then
    source_root="$(find "${WORK_ROOT}/code_extract" -type f -path '*/kuavo_train/lingbot/tasks/vla/train_lingbotvla.py' -print -quit)"
    [[ -n "${source_root}" ]] || { echo "Kuavo archive lacks the LingBot trainer" >&2; exit 4; }
    source_root="$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "${source_root}")")")")")"
    mkdir -p "${CODE_DIR}"
    cp -a "${source_root}/." "${CODE_DIR}/"
fi

if [[ ! -e "${LINGBOT_ROOT}/lingbotvla/models/auto.py" ]]; then
    : "${LINGBOT_CODE_GDOWN_URL:?Set LINGBOT_CODE_GDOWN_URL when synchronized LingBot-v1 source is absent}"
    : "${LINGBOT_CODE_SHA256:?Set LINGBOT_CODE_SHA256 when synchronized LingBot-v1 source is absent}"
    download_and_extract "${LINGBOT_CODE_GDOWN_URL}" "${LINGBOT_CODE_SHA256}" \
        "${WORK_ROOT}/downloads/lingbot-vla.zip" "${WORK_ROOT}/lingbot_extract" \
        "${LINGBOT_ROOT}/lingbotvla/models/auto.py"
fi
if [[ ! -e "${LINGBOT_ROOT}/lingbotvla/models/auto.py" ]]; then
    lingbot_marker="$(find "${WORK_ROOT}/lingbot_extract" -type f -path '*/lingbotvla/models/auto.py' -print -quit)"
    [[ -n "${lingbot_marker}" ]] || { echo "LingBot archive lacks lingbotvla/models/auto.py" >&2; exit 4; }
    source_root="$(dirname "$(dirname "$(dirname "${lingbot_marker}")")")"
    mkdir -p "${LINGBOT_ROOT}"
    cp -a "${source_root}/." "${LINGBOT_ROOT}/"
fi

cd "${CODE_DIR}"
PIPELINE_PHASE="install dependencies"
retry 3 uv pip install --python "$(command -v python)" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --reinstall-package torch \
    --reinstall-package torchvision \
    --reinstall-package torchdata \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchdata==0.11.0
# LingBot uses PyAV explicitly. Remove CUDA TorchCodec from rented images:
# LeRobot otherwise selects it by module presence even if CUDA NPP is missing.
uv pip uninstall --python "$(command -v python)" torchcodec >/dev/null 2>&1 || true
retry 3 uv pip install --python "$(command -v python)" -r requirements_lingbot_cloud.txt
retry 3 uv pip install --python "$(command -v python)" --no-deps -e "${CODE_DIR}/third_party/lerobot"
retry 3 uv pip install --python "$(command -v python)" -e "${LINGBOT_ROOT}" --no-deps

# FlashAttention's PyPI artifact may be an sdist. Resolve the exact official
# release wheel from the installed Torch ABI and never fall back to compilation.
if ! FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" python - <<'PY'
import importlib.metadata
import os
raise SystemExit(0 if importlib.metadata.version("flash-attn") == os.environ["FLASH_ATTN_VERSION"] else 1)
PY
then
    flash_wheel_url="${FLASH_ATTN_WHEEL_URL}"
    if [[ -z "${flash_wheel_url}" ]]; then
        flash_wheel_url="$(FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" python - <<'PY'
import os
import sys
import torch

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_tag = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
version = os.environ["FLASH_ATTN_VERSION"]
wheel = (
    f"flash_attn-{version}+cu12torch{torch_tag}cxx11abi{abi}-"
    f"{py_tag}-{py_tag}-linux_x86_64.whl"
)
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{wheel}")
PY
)"
    fi
    flash_wheel="${WORK_ROOT}/downloads/$(basename "${flash_wheel_url}")"
    mkdir -p "$(dirname "${flash_wheel}")"
    retry 3 curl --fail --location --silent --show-error --retry 3 \
        --retry-all-errors --output "${flash_wheel}" "${flash_wheel_url}"
    retry 3 uv pip install --python "$(command -v python)" --no-deps "${flash_wheel}"
fi
PYTHONPATH="${CODE_DIR}:${LINGBOT_ROOT}:${CODE_DIR}/third_party/lerobot/src:${PYTHONPATH:-}" python - <<'PY'
import av
import datasets
import diffusers
import flash_attn
import pandas
import serial
import torch
import torchdata
import torchvision
import transformers
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision.transforms.v2 import Resize
from kuavo_train.lingbot.tasks.vla import train_lingbotvla

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == int(__import__('os').environ['GPU_COUNT']), torch.cuda.device_count()
assert torch.__version__.split('+')[0] == "2.7.1", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torchvision.__version__.split('+')[0] == "0.22.1", torchvision.__version__
assert torchdata.__version__.split('+')[0] == "0.11.0", torchdata.__version__
assert flash_attn.__version__ == __import__('os').environ['FLASH_ATTN_VERSION'], flash_attn.__version__
major, minor = torch.cuda.get_device_capability(0)
capability = f"sm_{major}{minor}"
assert capability in torch.cuda.get_arch_list(), (
    capability,
    torch.cuda.get_arch_list(),
)
assert torch.ones(1, device="cuda").item() == 1
print(
    f"training import smoke test passed: torch={torch.__version__}, "
    f"cuda={torch.version.cuda}, torchvision={torchvision.__version__}, "
    f"torchdata={torchdata.__version__}, video_backend=pyav, transformers={transformers.__version__}, "
    f"flash_attn={flash_attn.__version__}, capability={capability}"
)
PY

PIPELINE_PHASE="authenticate Hugging Face"
retry 3 hf auth whoami

PIPELINE_PHASE="download dataset and models"
mkdir -p "${DATASET_ROOT}" "${LINGBOT_MODEL_ROOT}" "${QWEN_MODEL_ROOT}"
if [[ -n "${DATASET_MIX_JSON:-}" ]]; then
    MIXTURE_MANIFEST="${LOG_DIR}/dataset_mix.resolved.json"
    DATASET_MIX_JSON="${DATASET_MIX_JSON}" \
        python "${CODE_DIR}/tools/resolve_hf_dataset_mixture.py" \
        --task "${TRAINING_TASK}" \
        --output-root "${WORK_ROOT}/datasets/mixture" \
        --resolved-output "${MIXTURE_MANIFEST}"
    KUAVO_DATASET_MIX_JSON="$(
        python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' \
            "${MIXTURE_MANIFEST}"
    )"
    DATASET_ROOT="$(
        python -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["root"])' \
            "${MIXTURE_MANIFEST}"
    )"
    export KUAVO_DATASET_MIX_JSON
else
    retry 3 hf download "${DATASET_REPO}" --repo-type dataset --local-dir "${DATASET_ROOT}" --max-workers 16
fi
retry 3 hf download "${LINGBOT_MODEL_REPO}" --local-dir "${LINGBOT_MODEL_ROOT}" --max-workers 16
retry 3 hf download "${QWEN_MODEL_REPO}" \
    --local-dir "${QWEN_MODEL_ROOT}" --max-workers 16 \
    --include \
        "config.json" \
        "tokenizer*" \
        "special_tokens_map.json" \
        "added_tokens.json" \
        "vocab.json" \
        "merges.txt" \
        "*processor_config.json" \
        "chat_template*"

PIPELINE_PHASE="validate dataset and build norm"
if [[ -z "${DATASET_MIX_JSON:-}" ]]; then
DATASET_ROOT="${DATASET_ROOT}" python - <<'PY'
import json, os
from pathlib import Path
import av
import pyarrow.parquet as pq
root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0", info
assert info["total_episodes"] == int(os.environ["EXPECTED_EPISODES"]), info["total_episodes"]
assert info["total_frames"] == int(os.environ["EXPECTED_FRAMES"]), info["total_frames"]
assert info["fps"] == 10, info["fps"]
expected_dim = int(os.environ["EXPECTED_ACTION_DIM"])
assert info["features"]["observation.state"]["shape"] == [expected_dim]
assert info["features"]["action"]["shape"] == [expected_dim]
expected_cameras = set(os.environ["EXPECTED_CAMERAS"].split(","))
assert set(k for k in info["features"] if "images." in k) == expected_cameras
assert not any("depth" in key for key in info["features"])
tasks = pq.read_table(root / "meta/tasks.parquet").to_pydict()
assert len(tasks.get("task_index", [])) == 1, tasks
video = next((root / "videos").rglob("*.mp4"))
with av.open(str(video)) as container:
    frame = next(container.decode(video=0))
    assert frame.width > 0 and frame.height > 0
print("Dataset validation passed; task metadata:", tasks)
PY
fi
mkdir -p "$(dirname "${NORM_FILE}")"
PYTHONPATH="${CODE_DIR}:${LINGBOT_ROOT}:${CODE_DIR}/third_party/lerobot/src:${PYTHONPATH:-}" \
python "${CODE_DIR}/kuavo_train/lingbot/compute_mixture_norm.py" "${TRAIN_CONFIG}" \
    --data.data_name "${LINGBOT_V1_DATA_NAME}" \
    --data.train_path "${DATASET_ROOT}" \
    --data.norm_stats_file "${NORM_FILE}" \
    --data.num_workers "${DATALOADER_WORKERS}" \
    --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
    --train.chunk_size 50
for required in \
    "${LINGBOT_MODEL_ROOT}/config.json" \
    "${QWEN_MODEL_ROOT}/config.json" \
    "${QWEN_MODEL_ROOT}/tokenizer_config.json" \
    "${QWEN_MODEL_ROOT}/tokenizer.json" \
    "${QWEN_MODEL_ROOT}/preprocessor_config.json"; do
    [[ -s "${required}" ]] || { echo "Required model asset missing: ${required}" >&2; exit 5; }
done

PIPELINE_PHASE="start heartbeat"
mkdir -p "${RUN_DIR}"
if [[ "${RESUME}" == "1" ]] && \
   ! find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -name 'global_step_*' -print -quit 2>/dev/null | grep -q .; then
    PIPELINE_PHASE="download resume DCP"
    retry 3 hf download "${RESUME_REPO}" \
        --local-dir "${RUN_DIR}" \
        --max-workers 16
fi
(
    while sleep "${HEARTBEAT_SECONDS}"; do
        loss_line="$(tail -n 1 "${RUN_DIR}/checkpoints/loss.jsonl" 2>/dev/null || true)"
        disk="$(df -h "${WORK_ROOT}" | awk 'NR==2 {print $4 " free"}')"
        notify "${TASK_LABEL} LingBot训练心跳" $'Run：'"${RUN_ID}"$'\n'"${loss_line:-no loss yet}"$'\n磁盘：'"${disk}"
    done
) &
HEARTBEAT_PID=$!

use_wandb=false
[[ -n "${WANDB_API_KEY}" ]] && use_wandb=true

PIPELINE_PHASE="train"
notify "${TASK_LABEL} LingBot全量训练开始" $'Run：'"${RUN_ID}"$'\nGPU：'"${GPU_COUNT}"$'\nMicro BS：'"${MICRO_BATCH_SIZE}"$'\nGlobal BS：'"${GLOBAL_BATCH_SIZE}"$'\nEpoch：'"${NUM_EPOCHS}"$'\nLR：'"${LEARNING_RATE}"
set +e
PYTHONPATH="${CODE_DIR}:${LINGBOT_ROOT}:${CODE_DIR}/third_party/lerobot/src:${PYTHONPATH:-}" \
torchrun --standalone --nproc_per_node "${GPU_COUNT}" "${TRAIN_ENTRY}" "${TRAIN_CONFIG}" \
    --model.model_path "${LINGBOT_MODEL_ROOT}" \
    --model.tokenizer_path "${QWEN_MODEL_ROOT}" \
    --data.data_name "${LINGBOT_V1_DATA_NAME}" \
    --data.train_path "${DATASET_ROOT}" \
    --data.norm_stats_file "${NORM_FILE}" \
    --data.num_workers "${DATALOADER_WORKERS}" \
    --data.prefetch_factor "${DATALOADER_PREFETCH}" \
    --data.pin_memory true \
    --train.output_dir "${RUN_DIR}" \
    --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
    --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --train.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --train.num_train_epochs "${NUM_EPOCHS}" \
    --train.lr "${LEARNING_RATE}" \
    --train.lr_min "${MIN_LEARNING_RATE}" \
    --train.lr_warmup_ratio "${WARMUP_RATIO}" \
    --train.save_steps "${SAVE_STEPS}" \
    --train.save_epochs 10000 \
    --train.save_hf_weights false \
    --train.keep_last_checkpoints "${KEEP_LAST_CHECKPOINTS}" \
    --train.enable_resume true \
    --train.enable_full_shard true \
    --train.enable_mixed_precision true \
    --train.enable_fp32 false \
    --train.ignore_depth true \
    --train.use_compile "${USE_COMPILE}" \
    --train.use_wandb "${use_wandb}" \
    --train.wandb_project "${WANDB_PROJECT}" \
    --train.wandb_name "${RUN_ID}" 2>&1 | tee -a "${TRAIN_LOG}"
train_rc=${PIPESTATUS[0]}
set -e
if (( train_rc != 0 )); then
    FAIL_NOTIFIED=1
    notify "${TASK_LABEL} LingBot训练失败" $'Run：'"${RUN_ID}"$'\n退出码：'"${train_rc}"$'\n日志：\n'"$(tail -n 35 "${TRAIN_LOG}" 2>/dev/null || true)"
    exit "${train_rc}"
fi

PIPELINE_PHASE="export final HF checkpoint"
checkpoint="$(latest_checkpoint)"
[[ -n "${checkpoint}" ]] || { echo "Training completed without a checkpoint" >&2; exit 6; }
python tools/export_lingbot_full_checkpoint.py "${checkpoint}" \
    --lingbot-root "${LINGBOT_ROOT}" \
    --output "${FINAL_HF_DIR}" \
    --save-dtype bfloat16 \
    --force
cp "${NORM_FILE}" "${FINAL_HF_DIR}/norm_stats.json"
cp "${RUN_DIR}/lingbotvla_cli.yaml" "${FINAL_HF_DIR}/lingbotvla_cli.yaml"
cp "${TRAIN_CONFIG}" "${FINAL_HF_DIR}/source_training_config.yaml"
if [[ -f "${MIXTURE_MANIFEST:-}" ]]; then
    cp "${MIXTURE_MANIFEST}" "${FINAL_HF_DIR}/dataset_mix.json"
fi

PIPELINE_PHASE="upload final model"
MODEL_REPO="${MODEL_REPO}" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ["MODEL_REPO"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.update_repo_settings(repo_id=repo, repo_type="model", private=True)
assert api.model_info(repo).private
PY
retry 3 hf upload "${MODEL_REPO}" "${FINAL_HF_DIR}" . --commit-message "train: ${TASK_LABEL} LingBot full ${RUN_ID}"
retry 3 hf upload "${MODEL_REPO}" "${checkpoint}" \
    "checkpoints/$(basename "${checkpoint}")" \
    --commit-message "train-state: latest LingBot DCP ${RUN_ID}"
if [[ -f "${RUN_DIR}/checkpoints/loss.jsonl" ]]; then
    retry 3 hf upload "${MODEL_REPO}" "${RUN_DIR}/checkpoints/loss.jsonl" \
        "checkpoints/loss.jsonl" \
        --commit-message "train-state: LingBot loss history ${RUN_ID}"
fi
MODEL_REPO="${MODEL_REPO}" LATEST_DCP="$(basename "${checkpoint}")" python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ["MODEL_REPO"]
latest = os.environ["LATEST_DCP"]
files = set(api.list_repo_files(repo, repo_type="model"))
stale = [
    name
    for name in files
    if name.startswith("checkpoints/global_step_")
    and name.split("/", 2)[1] != latest
]
if stale:
    api.delete_files(
        repo_id=repo,
        repo_type="model",
        delete_patterns=stale,
        commit_message=f"cleanup: retain only latest LingBot DCP {latest}",
    )
info = api.model_info(repo)
files = {item.rfilename for item in info.siblings}
assert info.private
assert "config.json" in files and "model.safetensors.index.json" in files
assert "norm_stats.json" in files and "lingbotvla_cli.yaml" in files
assert any(name.endswith(".safetensors") for name in files)
assert any(
    name.startswith("checkpoints/global_step_") and name.endswith("/model/.metadata")
    for name in files
)
assert any(
    name.startswith("checkpoints/global_step_") and name.endswith("/optimizer/.metadata")
    for name in files
)
print("Remote model verification passed")
PY

if [[ "${DELETE_LOCAL_DCP_AFTER_UPLOAD}" == "1" ]]; then
    rm -rf "${RUN_DIR}/checkpoints"
fi

PIPELINE_PHASE="complete"
notify "${TASK_LABEL} LingBot训练与上传完成" $'Run：'"${RUN_ID}"$'\n模型：https://huggingface.co/'"${MODEL_REPO}"$'\n最新DCP：'"${checkpoint}"
echo "Pipeline complete: https://huggingface.co/${MODEL_REPO}"
