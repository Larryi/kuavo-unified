#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_BACKEND:?Set MODEL_BACKEND to dp, act, openpi, lingbot-v1, or lingbot-v2}"
: "${TRAINING_TASK:?Set TRAINING_TASK to task1, task2, or task3}"
: "${DRY_RUN:=0}"
: "${CODE_DIR:=${ROOT}}"
: "${WORK_ROOT:=/workspace/kuavo_runs/${MODEL_BACKEND}}"
: "${RUN_ID:=${MODEL_BACKEND//-/_}_$(date +%Y%m%d_%H%M%S)}"
: "${GPU_IDS:=0}"
: "${GPU_COUNT:=1}"
: "${AUTO_STOP_INSTANCE:=1}"
: "${AUTO_STOP_ON_FAILURE:=0}"
: "${AUTO_STOP_ON_UPLOAD_FAILURE:=0}"
: "${MODEL_REPO_PRIVATE:=0}"
: "${SERVERCHAN_SENDKEY:=}"
: "${SERVERCHAN_URL:=}"
: "${WANDB_API_KEY:=}"
: "${WANDB_PROJECT:=kuavo-${MODEL_BACKEND}}"
: "${VAST_INSTANCE_ID:=}"
: "${VAST_API_KEY:=}"
: "${HF_DOWNLOAD_WORKERS:=16}"
: "${PYTHON_BIN:=python}"
: "${CLASSIC_TORCH_VERSION:=2.7.1}"
: "${CLASSIC_TORCHVISION_VERSION:=0.22.1}"
: "${CLASSIC_PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${PREPARE_ENV:=1}"
: "${RESUME_MODE:=none}"
: "${RESUME_REPO:=}"
: "${RESUME_RUN_ID:=}"
: "${DATASET_MIX_JSON:=}"
: "${VAST_PYPI_INDEX:=https://pypi.org/simple}"

export PIP_INDEX_URL="${VAST_PYPI_INDEX}"
export UV_DEFAULT_INDEX="${VAST_PYPI_INDEX}"
unset PIP_EXTRA_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL

case "${MODEL_BACKEND}" in
  dp|act|openpi|lingbot-v1|lingbot-v2) ;;
  *) echo "Unsupported MODEL_BACKEND=${MODEL_BACKEND}" >&2; exit 2 ;;
esac
source "${ROOT}/scripts/vast/job_profile.sh"
apply_vast_job_profile

# uv treats a bare name such as "python" as a virtual-environment selector.
# Resolve it to the concrete interpreter used by the pipeline so classic
# backends can install into the existing VastAI system environment.
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
PYTHON_BIN="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
export PYTHON_BIN

for value in DRY_RUN AUTO_STOP_INSTANCE AUTO_STOP_ON_FAILURE AUTO_STOP_ON_UPLOAD_FAILURE MODEL_REPO_PRIVATE PREPARE_ENV; do
  [[ "${!value}" == "0" || "${!value}" == "1" ]] || {
    echo "${value} must be 0 or 1, got ${!value}" >&2
    exit 2
  }
done
[[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_COUNT must be a positive integer" >&2; exit 2; }
case "${RESUME_MODE}" in
  none|hf) ;;
  *) echo "RESUME_MODE must be none or hf" >&2; exit 2 ;;
esac
if [[ "${RESUME_MODE}" == "hf" ]]; then
  : "${RESUME_REPO:=${MODEL_REPO:-}}"
  [[ -n "${RESUME_REPO}" && -n "${RESUME_RUN_ID}" ]] || {
    echo "RESUME_MODE=hf requires RESUME_REPO and RESUME_RUN_ID" >&2
    exit 2
  }
fi
IFS=',' read -r -a selected_gpu_ids <<<"${GPU_IDS}"
(( ${#selected_gpu_ids[@]} == GPU_COUNT )) || {
  echo "GPU_IDS=${GPU_IDS} selects ${#selected_gpu_ids[@]} devices, expected ${GPU_COUNT}" >&2
  exit 2
}

describe_profile() {
  echo "Backend: ${MODEL_BACKEND}"
  echo "Task: ${TRAINING_TASK}"
  echo "Code: ${CODE_DIR}"
  echo "Work root: ${WORK_ROOT}"
  echo "Run: ${RUN_ID}"
  echo "GPUs: ${GPU_IDS}"
  echo "Resume: ${RESUME_MODE}${RESUME_RUN_ID:+ run=${RESUME_RUN_ID}}"
  case "${MODEL_BACKEND}" in
    dp)
      echo "Pretrained: torchvision ResNet18 ImageNet weights (resolved by the training environment)"
      echo "Environment: requirements_train_cloud.txt"
      echo "Torch: ${CLASSIC_TORCH_VERSION} / torchvision ${CLASSIC_TORCHVISION_VERSION} from cu128"
      echo "Batch: ${TRAIN_BATCH_SIZE:-adaptive by per-GPU VRAM (24G=32, 32G=40, 48G=64, 80G=128)}"
      echo "Dispatch: configs/policy/${TRAIN_CONFIG_NAME:-dp_r1.yaml}"
      ;;
    act)
      echo "Pretrained: torchvision ResNet18 ImageNet weights (resolved by the training environment)"
      echo "Environment: requirements_train_cloud.txt"
      echo "Torch: ${CLASSIC_TORCH_VERSION} / torchvision ${CLASSIC_TORCHVISION_VERSION} from cu128"
      echo "Batch: ${TRAIN_BATCH_SIZE:-adaptive by per-GPU VRAM (24G=32, 32G=40, 48G=64, 80G=128)}"
      echo "Dispatch: configs/policy/${TRAIN_CONFIG_NAME:-act_config.yaml}"
      ;;
    openpi)
      echo "Pretrained: ${BASE_PARAMS:-gs://openpi-assets/checkpoints/pi05_base/params}"
      echo "Tokenizer: ${PALIGEMMA_REPO:-google/paligemma-3b-pt-224}"
      echo "Dispatch: third_party/openpi-kuavo/scripts/vast/run_pi05_pipeline.sh"
      ;;
    lingbot-v1)
      echo "Pretrained: ${LINGBOT_MODEL_REPO:-robbyant/lingbot-vla-4b} -> ${WORK_ROOT}/models/LingBotVLA"
      echo "Tokenizer: ${QWEN_MODEL_REPO:-Qwen/Qwen2.5-VL-3B-Instruct} -> ${WORK_ROOT}/models/Qwen2.5_VL"
      echo "Dispatch: scripts/run_lingbot_v1_full_pipeline.sh"
      ;;
    lingbot-v2)
      echo "Pretrained: ${LINGBOT_V2_MODEL_REPO:-robbyant/lingbot-vla-v2-6b} -> ${WORK_ROOT}/models/lingbot-vla-v2-6b"
      echo "Tokenizer: ${QWEN3_MODEL_REPO:-Qwen/Qwen3-VL-4B-Instruct} -> ${WORK_ROOT}/models/Qwen3-VL-4B-Instruct"
      echo "MoGe: ${MOGE_MODEL_REPO:-Ruicheng/moge-2-vitb-normal} -> ${WORK_ROOT}/models/moge-2-vitb-normal"
      echo "Depth/DINO: included in the LingBot-v2 base repository"
      echo "Environment: dedicated Python 3.12 / PyTorch 2.8.0 LingBot-v2 image"
      echo "Dispatch: python -m kuavo_train.train_lingbot_v2"
      ;;
  esac
}

dataset_mix_count=0
if [[ -n "${DATASET_MIX_JSON}" ]]; then
  dataset_mix_count="$(DATASET_MIX_JSON="${DATASET_MIX_JSON}" "${PYTHON_BIN}" - <<'PY'
import json
import os

payload = json.loads(os.environ["DATASET_MIX_JSON"])
if not isinstance(payload, list) or not payload:
    raise SystemExit("DATASET_MIX_JSON must be a non-empty JSON list")
if any(float(item["weight"]) <= 0 for item in payload):
    raise SystemExit("Every dataset mixture weight must be positive")
print(len(payload))
PY
)"
fi
describe_profile
echo "Datasets: $(( dataset_mix_count > 0 ? dataset_mix_count : 1 ))"
if [[ "${DRY_RUN}" == "1" ]]; then
  for name in HF_TOKEN WANDB_API_KEY SERVERCHAN_SENDKEY VAST_API_KEY DATASET_REPO MODEL_REPO RESUME_REPO; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name}=set"
    else
      echo "${name}=unset"
    fi
  done
  echo "Dry run complete; no network, training, upload, notification, or shutdown was attempted."
  exit 0
fi

: "${HF_TOKEN:?Set HF_TOKEN with dataset read and model write access}"
: "${DATASET_REPO:?Set DATASET_REPO}"
: "${MODEL_REPO:?Set MODEL_REPO}"
[[ -d "${CODE_DIR}/.git" || -f "${CODE_DIR}/MIGRATION_HANDOFF.md" ]] || {
  echo "Unified source is missing at CODE_DIR=${CODE_DIR}" >&2
  exit 3
}

export HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export WANDB_PROJECT
if [[ -n "${WANDB_API_KEY}" ]]; then
  export WANDB_API_KEY WANDB_MODE=online
fi

LOG_DIR="${WORK_ROOT}/logs/${RUN_ID}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
STATUS_FILE="${LOG_DIR}/status.json"
PIPELINE_PHASE="bootstrap"
UPLOAD_STATUS="not_started"
TRAIN_STARTED=0
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

write_status() {
  local state="$1"
  STATUS_STATE="${state}" STATUS_PHASE="${PIPELINE_PHASE}" \
    STATUS_UPLOAD="${UPLOAD_STATUS}" STATUS_FILE="${STATUS_FILE}" \
    STATUS_RUN_ID="${RUN_ID}" STATUS_TASK="${TRAINING_TASK}" \
    STATUS_BACKEND="${MODEL_BACKEND}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["STATUS_FILE"])
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps({
    "schema_version": 1,
    "state": os.environ["STATUS_STATE"],
    "phase": os.environ["STATUS_PHASE"],
    "upload": os.environ["STATUS_UPLOAD"],
    "run_id": os.environ["STATUS_RUN_ID"],
    "task": os.environ["STATUS_TASK"],
    "backend": os.environ["STATUS_BACKEND"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2))
tmp.replace(path)
PY
}
write_status running

set_phase() {
  PIPELINE_PHASE="$1"
  write_status running
  echo "[phase] ${PIPELINE_PHASE}"
}

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
  [[ -n "${url}" ]] || { echo "[notify disabled] ${text}"; return 0; }
  curl --fail --silent --show-error --max-time 20 --retry 3 --retry-delay 2 \
    --request POST "${url}" \
    --header 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode "text=${text}" \
    --data-urlencode "desp=${desp}" >/dev/null || {
      echo "ServerChan notification failed (URL hidden)" >&2
      return 0
    }
}

stop_instance() {
  local rc="$1"
  [[ "${AUTO_STOP_INSTANCE}" == "1" ]] || return 0
  if (( rc != 0 )); then
    if [[ "${UPLOAD_STATUS}" != "success" ]]; then
      if [[ "${AUTO_STOP_ON_UPLOAD_FAILURE}" != "1" ]]; then
        echo "Upload incomplete; leaving Vast instance running" >&2
        return 0
      fi
    elif [[ "${AUTO_STOP_ON_FAILURE}" != "1" ]]; then
      echo "Pipeline failed; leaving Vast instance running" >&2
      return 0
    fi
  fi
  if [[ -z "${VAST_API_KEY}" || -z "${VAST_INSTANCE_ID}" ]]; then
    echo "AUTO_STOP_INSTANCE needs VAST_API_KEY and VAST_INSTANCE_ID; leaving instance running" >&2
    return 0
  fi
  if ! command -v uvx >/dev/null 2>&1; then
    echo "uvx is unavailable; cannot stop Vast instance automatically" >&2
    return 0
  fi
  VAST_API_KEY="${VAST_API_KEY}" uvx --from vastai vastai stop instance "${VAST_INSTANCE_ID}" --raw || {
    echo "Vast stop request failed for instance ${VAST_INSTANCE_ID}" >&2
    return 0
  }
}

on_exit() {
  local rc=$?
  trap - EXIT
  set +e
  if (( rc == 0 )); then
    PIPELINE_PHASE="complete"
    write_status success
    notify "Kuavo ${MODEL_BACKEND} 云训练完成" "Run: ${RUN_ID}
Model: https://huggingface.co/${MODEL_REPO}
Upload: ${UPLOAD_STATUS}"
  else
    write_status failed
    notify "Kuavo ${MODEL_BACKEND} 云训练失败" "Run: ${RUN_ID}
Phase: ${PIPELINE_PHASE}
Exit: ${rc}
Upload: ${UPLOAD_STATUS}"
  fi
  stop_instance "${rc}"
  exit "${rc}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command is missing: $1" >&2
    exit 4
  }
}

prepare_classic_environment() {
  [[ "${PREPARE_ENV}" == "1" ]] || return 0
  set_phase "prepare ${MODEL_BACKEND} environment"
  "${PYTHON_BIN}" -m pip install --upgrade uv
  uv pip install --python "${PYTHON_BIN}" \
    --index-url "${CLASSIC_PYTORCH_INDEX_URL}" \
    "torch==${CLASSIC_TORCH_VERSION}" \
    "torchvision==${CLASSIC_TORCHVISION_VERSION}"
  uv pip install --python "${PYTHON_BIN}" -r "${CODE_DIR}/requirements_train_cloud.txt"
}

validate_classic_cuda() {
  set_phase "validate ${MODEL_BACKEND} CUDA environment"
  "${PYTHON_BIN}" - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"
capability = torch.cuda.get_device_capability(0)
wheel_cuda = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
if capability >= (12, 0):
    assert wheel_cuda >= (12, 8), (
        f"Blackwell {capability} requires a CUDA 12.8+ PyTorch wheel, "
        f"got torch={torch.__version__}, CUDA={torch.version.cuda}"
    )
    assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.randn((1024, 1024), device="cuda")
torch.cuda.synchronize()
result = (x @ x).mean()
torch.cuda.synchronize()
print(
    "Classic CUDA validation passed:",
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": capability,
        "matmul": float(result),
    },
)
PY
}

configure_classic_batch_size() {
  if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
    echo "Classic batch size: ${TRAIN_BATCH_SIZE} (explicit override)"
    return
  fi
  local gpu_id="${selected_gpu_ids[0]}" memory_mb
  memory_mb="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  [[ "${memory_mb}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Could not determine GPU memory for adaptive batch size" >&2
    exit 4
  }
  case "${MODEL_BACKEND}:${memory_mb}" in
    act:*) TRAIN_BATCH_SIZE=32 ;;
    dp:*) TRAIN_BATCH_SIZE=32 ;;
  esac
  if (( memory_mb >= 30000 )); then
    TRAIN_BATCH_SIZE=40
  fi
  if (( memory_mb >= 45000 )); then
    TRAIN_BATCH_SIZE=64
  fi
  if (( memory_mb >= 70000 )); then
    TRAIN_BATCH_SIZE=128
  fi
  export TRAIN_BATCH_SIZE
  echo "Classic adaptive batch size: backend=${MODEL_BACKEND}, memory=${memory_mb} MiB, batch=${TRAIN_BATCH_SIZE}"
}

lingbot_v2_environment_ready() {
  local env_name="$1"
  local env_python
  env_python="$(conda run -n "${env_name}" python -c 'import sys; print(sys.executable)')" \
    || return 1
  [[ -x "${env_python}" ]] || return 1
  "${env_python}" - <<'PY'
import sys
from importlib.metadata import version

expected = {
    "torch": "2.8.0",
    "transformers": "4.57.3",
    "accelerate": "1.7.0",
}
if sys.version_info[:2] != (3, 12):
    raise SystemExit(1)
if {name: version(name) for name in expected} != expected:
    raise SystemExit(1)
hydra_version = tuple(int(part) for part in version("hydra-core").split(".")[:3])
if not (hydra_version >= (1, 3, 2) and hydra_version < (1, 4, 0)):
    raise SystemExit(1)

import flash_attn  # noqa: F401
import hydra  # noqa: F401
import utils3d

if not hasattr(getattr(utils3d, "pt", None), "intrinsics_from_focal_center"):
    raise SystemExit(1)
PY
}

prepare_lingbot_v2_environment() {
  local env_name="${LINGBOT_V2_ENV_NAME:-lingbotvla_v2}"
  if [[ "${PREPARE_ENV}" == "1" ]]; then
    set_phase "prepare LingBot-v2 environment"
    if ! command -v conda >/dev/null 2>&1; then
      local miniforge_root="${WORK_ROOT}/miniforge3"
      if [[ ! -x "${miniforge_root}/bin/conda" ]]; then
        local installer="${WORK_ROOT}/Miniforge3-Linux-x86_64.sh"
        curl --fail --location --retry 5 --retry-all-errors \
          --output "${installer}" \
          "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        bash "${installer}" -b -p "${miniforge_root}"
      fi
      export PATH="${miniforge_root}/bin:${PATH}"
    fi
    if conda env list | awk '{print $1}' | grep -Fxq "${env_name}" \
      && lingbot_v2_environment_ready "${env_name}" >/dev/null 2>&1; then
      echo "LingBot-v2 environment is ready; skipping dependency installation."
    else
      echo "LingBot-v2 environment is missing or incomplete; preparing it once."
      PIP_INDEX_URL="${PIP_INDEX_URL}" \
        FLASH_ATTN_INSTALL_MODE="${FLASH_ATTN_INSTALL_MODE:-auto}" \
        LINGBOT_V2_ENV_NAME="${env_name}" \
        bash "${CODE_DIR}/docker/create_lingbot_v2_env.sh"
    fi
  fi
  command -v conda >/dev/null 2>&1 || {
    echo "LingBot-v2 requires conda or PREPARE_ENV=1." >&2
    exit 4
  }
  PYTHON_BIN="$(conda run -n "${env_name}" python -c 'import sys; print(sys.executable)')"
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "LingBot-v2 Python is missing: ${PYTHON_BIN}" >&2
    exit 4
  }
  export PYTHON_BIN
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
}

validate_lingbot_v2_environment() {
  PIPELINE_PHASE="validate LingBot-v2 environment"
  "${PYTHON_BIN}" - <<'PY'
import sys
from importlib.metadata import version

assert sys.version_info[:2] == (3, 12), sys.version
expected = {
    "torch": "2.8.0",
    "transformers": "4.57.3",
    "accelerate": "1.7.0",
}
actual = {name: version(name) for name in expected}
assert actual == expected, (actual, expected)
hydra_version = version("hydra-core")
hydra_parts = tuple(int(part) for part in hydra_version.split(".")[:3])
assert (1, 3, 2) <= hydra_parts < (1, 4, 0), hydra_version
actual["hydra-core"] = hydra_version
import hydra  # noqa: F401
import flash_attn  # noqa: F401
print("LingBot-v2 environment passed:", actual)
PY
}

download_hf() {
  local repo_id="$1" local_dir="$2" repo_type="${3:-model}"
  mkdir -p "${local_dir}"
  hf download "${repo_id}" --repo-type "${repo_type}" \
    --local-dir "${local_dir}" --max-workers "${HF_DOWNLOAD_WORKERS}"
}

download_hf_processor() {
  local repo_id="$1" local_dir="$2"
  mkdir -p "${local_dir}"
  hf download "${repo_id}" \
    --local-dir "${local_dir}" --max-workers "${HF_DOWNLOAD_WORKERS}" \
    --include \
      "config.json" \
      "tokenizer*" \
      "special_tokens_map.json" \
      "added_tokens.json" \
      "vocab.json" \
      "merges.txt" \
      "*processor_config.json" \
      "chat_template*"
}

download_resume() {
  local destination="$1"
  [[ "${RESUME_MODE}" == "hf" ]] || return 0
  set_phase "download resume checkpoint"
  download_hf "${RESUME_REPO}" "${destination}" model
}

resolve_dataset_mixture() {
  local resolved_output="${LOG_DIR}/dataset_mix.resolved.json"
  if [[ -z "${DATASET_MIX_JSON}" ]]; then
    set_phase "download dataset"
    download_hf "${DATASET_REPO}" "${DATASET_ROOT}" dataset
    return 0
  fi
  set_phase "download and validate dataset mixture"
  DATASET_MIX_JSON="${DATASET_MIX_JSON}" \
    "${PYTHON_BIN}" "${CODE_DIR}/tools/resolve_hf_dataset_mixture.py" \
      --task "${TRAINING_TASK}" \
      --output-root "${WORK_ROOT}/datasets/mixture" \
      --resolved-output "${resolved_output}"
  KUAVO_DATASET_MIX_JSON="$(
    "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' \
      "${resolved_output}"
  )"
  DATASET_ROOT="$(
    "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["root"])' \
      "${resolved_output}"
  )"
  export KUAVO_DATASET_MIX_JSON
}

upload_directory() {
  local source_dir="$1"
  [[ -d "${source_dir}" ]] || { echo "Upload directory is missing: ${source_dir}" >&2; return 1; }
  set_phase "upload model"
  local repo_args=(repo create "${MODEL_REPO}" --repo-type model --exist-ok)
  [[ "${MODEL_REPO_PRIVATE}" == "1" ]] && repo_args+=(--private)
  hf "${repo_args[@]}"
  hf upload "${MODEL_REPO}" "${source_dir}" . --repo-type model \
    --commit-message "train: ${MODEL_BACKEND} ${RUN_ID}"
  UPLOAD_STATUS="success"
}

run_directory_name() {
  local run_id="$1"
  if [[ "${run_id}" == run_* ]]; then
    printf '%s' "${run_id}"
  else
    printf 'run_%s' "${run_id}"
  fi
}

run_classic() {
  local config_name task_name method_name output_base run_dir
  config_name="${TRAIN_CONFIG_NAME:-$([[ "${MODEL_BACKEND}" == "dp" ]] && echo dp_r1.yaml || echo act_config.yaml)}"
  task_name="${TASK_NAME:-r1}"
  method_name="${METHOD_NAME:-${MODEL_BACKEND}_cloud}"
  output_base="${WORK_ROOT}/outputs/${MODEL_BACKEND}"
  run_dir="${output_base}/$(run_directory_name "${RUN_ID}")"
  resolve_dataset_mixture
  if [[ "${RESUME_MODE}" == "hf" ]]; then
    run_dir="${output_base}/$(run_directory_name "${RESUME_RUN_ID}")"
    download_resume "${run_dir}"
  fi
  set_phase "train ${MODEL_BACKEND}"
  TRAIN_STARTED=1
  local overrides=(
    --config-path=../configs/policy
    "--config-name=${config_name}"
    "task=${task_name}"
    "method=${method_name}"
    "root=${DATASET_ROOT}"
    "repoid=${DATASET_REPO}"
    "timestamp=${RUN_ID}"
    "training.output_directory=${output_base}"
    "training.device=cuda"
  )
  [[ -n "${TRAIN_BATCH_SIZE:-}" ]] && overrides+=("training.batch_size=${TRAIN_BATCH_SIZE}")
  [[ -n "${GRAD_ACCUM_STEPS:-}" ]] && overrides+=("training.accumulation_steps=${GRAD_ACCUM_STEPS}")
  [[ -n "${TRAIN_MAX_STEPS:-}" ]] && overrides+=("training.max_training_step=${TRAIN_MAX_STEPS}")
  if [[ "${RESUME_MODE}" == "hf" ]]; then
    overrides+=("training.resume=true" "training.resume_timestamp=run_${RESUME_RUN_ID#run_}")
  fi
  if (( GPU_COUNT > 1 )); then
    require_command accelerate
    (cd "${CODE_DIR}" && accelerate launch --multi_gpu \
      --num_processes "${GPU_COUNT}" \
      --mixed_precision "${MIXED_PRECISION:-bf16}" \
      kuavo_train/train_policy_with_accelerate.py "${overrides[@]}")
  else
    (cd "${CODE_DIR}" && "${PYTHON_BIN}" kuavo_train/train_policy.py "${overrides[@]}")
  fi
  if [[ -f "${LOG_DIR}/dataset_mix.resolved.json" ]]; then
    cp "${LOG_DIR}/dataset_mix.resolved.json" "${run_dir}/dataset_mix.json"
  fi
  upload_directory "${run_dir}"
}

run_lingbot_v2() {
  local model_root qwen_root moge_root output_base run_dir norm_file
  model_root="${WORK_ROOT}/models/lingbot-vla-v2-6b"
  qwen_root="${WORK_ROOT}/models/Qwen3-VL-4B-Instruct"
  moge_root="${WORK_ROOT}/models/moge-2-vitb-normal"
  output_base="${WORK_ROOT}/outputs/lingbot_v2"
  run_dir="${output_base}/$(run_directory_name "${RUN_ID}")"
  resolve_dataset_mixture
  PIPELINE_PHASE="download LingBot-v2 assets"
  download_hf "${LINGBOT_V2_MODEL_REPO:-robbyant/lingbot-vla-v2-6b}" "${model_root}"
  download_hf_processor "${QWEN3_MODEL_REPO:-Qwen/Qwen3-VL-4B-Instruct}" "${qwen_root}"
  download_hf "${MOGE_MODEL_REPO:-Ruicheng/moge-2-vitb-normal}" "${moge_root}"
  export LINGBOT_V2_MOGE_PATH="${moge_root}/model.pt"
  export LINGBOT_V2_DEPTH_PATH="${model_root}/depth/model.pt"
  export LINGBOT_V2_DINO_CKPT="${model_root}/dino_video/teacher_step_10000.pth"
  export LINGBOT_V2_DINO_CONFIG="${model_root}/dino_video/config.yaml"
  for required_asset in \
    "${model_root}/config.json" \
    "${model_root}/model.safetensors.index.json" \
    "${LINGBOT_V2_MOGE_PATH}" \
    "${LINGBOT_V2_DEPTH_PATH}" \
    "${LINGBOT_V2_DINO_CKPT}" \
    "${LINGBOT_V2_DINO_CONFIG}" \
    "${qwen_root}/config.json" \
    "${qwen_root}/tokenizer_config.json"; do
    [[ -f "${required_asset}" ]] || {
      echo "LingBot-v2 pretrained asset is missing: ${required_asset}" >&2
      exit 5
    }
  done
  if [[ "${RESUME_MODE}" == "hf" ]]; then
    run_dir="${output_base}/$(run_directory_name "${RESUME_RUN_ID}")"
    download_resume "${run_dir}"
  fi
  PIPELINE_PHASE="compute LingBot-v2 normalization"
  norm_file="${run_dir}/norm_stats.json"
  mkdir -p "${run_dir}"
  export LINGBOT_V2_ROOT="${CODE_DIR}/third_party/lingbot-vla-v2"
  export LINGBOT_V2_NORM_STATS="${norm_file}"
  PYTHONPATH="${CODE_DIR}:${LINGBOT_V2_ROOT}:${CODE_DIR}/third_party/lerobot/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${CODE_DIR}/kuavo_train/lingbot_v2/compute_norm_stats.py" \
      "${CODE_DIR}/${LINGBOT_V2_NORM_CONFIG}" \
      --data.data_name "${LINGBOT_V2_DATA_NAME}" \
      --data.train_path "${DATASET_ROOT}" \
      --data.robot_config_root "${CODE_DIR}/configs/robot_configs" \
      --data.num_workers "${NORM_WORKERS:-4}" \
      --data.norm_path "${norm_file}" \
      --data.data_ratio_for_norm_compute 1 \
      --data.norm_merge_chunk_dim true
  PIPELINE_PHASE="train LingBot-v2"
  TRAIN_STARTED=1
  (cd "${CODE_DIR}" && "${PYTHON_BIN}" -m kuavo_train.train_lingbot_v2 \
    "root=${DATASET_ROOT}" \
    "repoid=${DATASET_REPO}" \
    "timestamp=${RUN_ID}" \
    "training.output_directory=${output_base}" \
    "training.batch_size=${TRAIN_BATCH_SIZE:-1}" \
    "training.accumulation_steps=${GRAD_ACCUM_STEPS:-4}" \
    "training.max_training_step=${TRAIN_MAX_STEPS:-10000}" \
    "training.max_epoch=${TRAIN_EPOCHS:-10}" \
    "training.resume=$([[ "${RESUME_MODE}" == "hf" ]] && echo true || echo false)" \
    "training.resume_timestamp=${RESUME_RUN_ID}" \
    "policy.config_path=${LINGBOT_V2_TRAIN_CONFIG}" \
    "policy.model_path=${model_root}" \
    "policy.tokenizer_path=${qwen_root}")
  if [[ -f "${LOG_DIR}/dataset_mix.resolved.json" ]]; then
    cp "${LOG_DIR}/dataset_mix.resolved.json" "${run_dir}/dataset_mix.json"
  fi
  upload_directory "${run_dir}"
}

DATASET_ROOT="${DATASET_ROOT:-${WORK_ROOT}/datasets/lerobot}"
require_command curl
notify "Kuavo ${MODEL_BACKEND} 云训练开始" "Run: ${RUN_ID}
Host: $(hostname)
Dataset: ${DATASET_REPO}"

case "${MODEL_BACKEND}" in
  dp|act)
    prepare_classic_environment
    validate_classic_cuda
    configure_classic_batch_size
    require_command hf
    run_classic
    ;;
  openpi)
    set_phase "OpenPI delegated pipeline"
    TRAIN_STARTED=1
    child_serverchan="${SERVERCHAN_SENDKEY}"
    SERVERCHAN_SENDKEY="" AUTO_STOP_INSTANCE=0 \
      DATASET_MIX_JSON="${DATASET_MIX_JSON}" \
      RESUME="$([[ "${RESUME_MODE}" == "hf" ]] && echo 1 || echo 0)" \
      RESUME_REPO="${RESUME_REPO:-${MODEL_REPO}}" \
      RUN_ID="${RESUME_RUN_ID:-${RUN_ID}}" \
      CODE_DIR="${CODE_DIR}/third_party/openpi-kuavo" \
      WORK_ROOT="${WORK_ROOT}" \
      bash "${CODE_DIR}/third_party/openpi-kuavo/scripts/vast/run_pi05_pipeline.sh"
    SERVERCHAN_SENDKEY="${child_serverchan}"
    if [[ "${AUTO_UPLOAD:-1}" == "1" ]]; then
      UPLOAD_STATUS="success"
    else
      UPLOAD_STATUS="disabled"
    fi
    ;;
  lingbot-v1)
    set_phase "LingBot-v1 delegated pipeline"
    TRAIN_STARTED=1
    child_serverchan="${SERVERCHAN_SENDKEY}"
    child_serverchan_url="${SERVERCHAN_URL}"
    SERVERCHAN_SENDKEY="" SERVERCHAN_URL="" AUTO_STOP_INSTANCE=0 \
      CODE_DIR="${CODE_DIR}" \
      LINGBOT_ROOT="${CODE_DIR}/third_party/lingbot-vla" \
      RESUME="$([[ "${RESUME_MODE}" == "hf" ]] && echo 1 || echo 0)" \
      RESUME_REPO="${RESUME_REPO}" \
      DATASET_MIX_JSON="${DATASET_MIX_JSON}" \
      TRAINING_TASK="${TRAINING_TASK}" \
      RUN_ID="${RESUME_RUN_ID:-${RUN_ID}}" \
      WORK_ROOT="${WORK_ROOT}" \
      bash "${CODE_DIR}/scripts/run_lingbot_v1_full_pipeline.sh"
    SERVERCHAN_SENDKEY="${child_serverchan}"
    SERVERCHAN_URL="${child_serverchan_url}"
    UPLOAD_STATUS="success"
    ;;
  lingbot-v2)
    prepare_lingbot_v2_environment
    validate_lingbot_v2_environment
    require_command hf
    run_lingbot_v2
    ;;
esac
