#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-}"
DRY_RUN="${DRY_RUN:-0}"
GPU_DEVICE="${GPU_DEVICE:-all}"
HOST_PORT="${HOST_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-}"
ASSET_DIR="${ASSET_DIR:-}"
POLICY_PATH="${POLICY_PATH:-/models/checkpoint}"
API_ENV_FILE="${API_ENV_FILE:-}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
IMAGE_NAME="${IMAGE_NAME:-}"
INTERACTIVE="${INTERACTIVE:-0}"

case "${BACKEND}" in
    act|diffusion)
        IMAGE_NAME="${IMAGE_NAME:-kuavo-classic:latest}"
        ;;
    openpi)
        IMAGE_NAME="${IMAGE_NAME:-kuavo-openpi-worker:latest}"
        ;;
    lingbot)
        IMAGE_NAME="${IMAGE_NAME:-kdc_real_task1_lingbot:latest}"
        ;;
    lingbot_v2)
        IMAGE_NAME="${IMAGE_NAME:-kuavo-lingbot-v2-worker:latest}"
        ;;
    *)
        echo "Set BACKEND to act, diffusion, openpi, lingbot, or lingbot_v2." >&2
        exit 2
        ;;
esac

CONTAINER_NAME="${CONTAINER_NAME:-kuavo-${BACKEND//_/-}-worker}"

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
        echo "Set MODEL_DIR to the checkpoint/asset directory to mount read-only." >&2
        exit 2
    fi
    if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "Container ${CONTAINER_NAME} already exists; stop/remove it explicitly or choose CONTAINER_NAME." >&2
        exit 3
    fi
fi

docker_args=(
    docker run
    --rm
    --init
    --name "${CONTAINER_NAME}"
    --network host
    --gpus "${GPU_DEVICE}"
)

if [[ "${INTERACTIVE}" == "1" ]]; then
    docker_args+=(-it)
fi
if [[ -n "${MODEL_DIR}" ]]; then
    docker_args+=(-v "${MODEL_DIR}:/models:ro")
fi
if [[ -n "${ASSET_DIR}" ]]; then
    if [[ "${DRY_RUN}" != "1" && ! -d "${ASSET_DIR}" ]]; then
        echo "ASSET_DIR does not exist: ${ASSET_DIR}" >&2
        exit 2
    fi
    docker_args+=(-v "${ASSET_DIR}:/assets:ro")
fi
if [[ -n "${API_ENV_FILE}" ]]; then
    if [[ "${DRY_RUN}" != "1" ]]; then
        if [[ ! -s "${API_ENV_FILE}" ]]; then
            echo "API_ENV_FILE is missing or empty." >&2
            exit 2
        fi
        mode="$(stat -c '%a' "${API_ENV_FILE}")"
        if [[ "${mode}" != "600" && "${mode}" != "400" ]]; then
            echo "API_ENV_FILE must have mode 600 or 400." >&2
            exit 3
        fi
    fi
    docker_args+=(--env-file "${API_ENV_FILE}")
fi

if [[ "${BACKEND}" == "openpi" ]]; then
    if [[ -z "${SERVER_ARGS:-}" ]]; then
        echo "Set SERVER_ARGS to the OpenPI serve_policy.py arguments." >&2
        exit 2
    fi
    docker_args+=(
        -e "SERVER_ARGS=${SERVER_ARGS}"
        -e OPENPI_DATA_HOME=/models
        -e IS_DOCKER=true
        "${IMAGE_NAME}"
    )
else
    worker_args=(
        tools/policy_worker.py
        --backend "${BACKEND}"
        --policy-path "${POLICY_PATH}"
        --host 0.0.0.0
        --port "${HOST_PORT}"
    )
    if [[ "${BACKEND}" == "lingbot" ]]; then
        worker_args+=(--lingbot-root /root/kuavo_data_challenge/third_party/lingbot-vla)
    elif [[ "${BACKEND}" == "lingbot_v2" ]]; then
        worker_args+=(--lingbot-root /opt/kuavo/third_party/lingbot-vla-v2)
    fi
    if [[ -n "${QWEN_PATH:-}" ]]; then
        worker_args+=(--qwen-path "${QWEN_PATH}")
    fi
    if [[ -n "${NORM_STATS_FILE:-}" ]]; then
        worker_args+=(--norm-stats-file "${NORM_STATS_FILE}")
    fi
    if [[ -n "${ROBOT_NAME:-}" ]]; then
        worker_args+=(--robot-name "${ROBOT_NAME}")
    fi
    if [[ -n "${TASK_PROMPT:-}" ]]; then
        worker_args+=(--task-prompt "${TASK_PROMPT}")
    fi
    docker_args+=(--entrypoint python "${IMAGE_NAME}" "${worker_args[@]}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${docker_args[@]}"
    printf '\n'
    exit 0
fi

exec "${docker_args[@]}"
