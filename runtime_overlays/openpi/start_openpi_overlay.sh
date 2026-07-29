#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 /absolute/or/repo-relative/runtime.env" >&2
}

[[ $# -eq 1 ]] || {
    usage
    exit 2
}

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$1"
if [[ "${ENV_FILE}" != /* ]]; then
    ENV_FILE="${ROOT}/${ENV_FILE}"
fi
[[ -f "${ENV_FILE}" ]] || {
    echo "Runtime env file not found: ${ENV_FILE}" >&2
    exit 2
}

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${IMAGE_NAME:?Set IMAGE_NAME in ${ENV_FILE}}"
: "${CONTAINER_NAME:?Set CONTAINER_NAME in ${ENV_FILE}}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR in ${ENV_FILE}}"
: "${TOKENIZER_FILE:?Set TOKENIZER_FILE in ${ENV_FILE}}"
: "${NORM_STATS_FILE:?Set NORM_STATS_FILE in ${ENV_FILE}}"
: "${OPENPI_ASSET_ID:?Set OPENPI_ASSET_ID in ${ENV_FILE}}"
: "${DEPLOY_CONFIG:?Set DEPLOY_CONFIG in ${ENV_FILE}}"
: "${EXECUTE_STEPS:?Set EXECUTE_STEPS in ${ENV_FILE}}"
: "${ROS_MASTER_URI:?Set ROS_MASTER_URI in ${ENV_FILE}}"
: "${ROS_IP:?Set ROS_IP in ${ENV_FILE}}"

[[ "${OPENPI_ASSET_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "OPENPI_ASSET_ID contains unsupported characters: ${OPENPI_ASSET_ID}" >&2
    exit 2
}
[[ "${EXECUTE_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "EXECUTE_STEPS must be a positive integer, got ${EXECUTE_STEPS}" >&2
    exit 2
}
(( EXECUTE_STEPS <= 50 )) || {
    echo "EXECUTE_STEPS cannot exceed the OpenPI action horizon 50" >&2
    exit 2
}

CHECKPOINT_DIR="$(realpath -e "${CHECKPOINT_DIR}")"
TOKENIZER_FILE="$(realpath -e "${TOKENIZER_FILE}")"
NORM_STATS_FILE="$(realpath -e "${NORM_STATS_FILE}")"
if [[ "${DEPLOY_CONFIG}" != /* ]]; then
    DEPLOY_CONFIG="${ROOT}/${DEPLOY_CONFIG}"
fi
DEPLOY_CONFIG="$(realpath -e "${DEPLOY_CONFIG}")"

[[ -f "${CHECKPOINT_DIR}/params/_METADATA" ]] || {
    echo "OpenPI checkpoint is missing params/_METADATA: ${CHECKPOINT_DIR}" >&2
    exit 3
}
[[ -f "${TOKENIZER_FILE}" ]] || {
    echo "Tokenizer is not a file: ${TOKENIZER_FILE}" >&2
    exit 3
}
[[ "$(basename -- "${TOKENIZER_FILE}")" == "tokenizer.model" ]] || {
    echo "OpenPI tokenizer must be named tokenizer.model: ${TOKENIZER_FILE}" >&2
    exit 3
}
TOKENIZER_DIR="$(dirname -- "${TOKENIZER_FILE}")"
[[ -f "${NORM_STATS_FILE}" ]] || {
    echo "Norm stats are not a file: ${NORM_STATS_FILE}" >&2
    exit 3
}
grep -q 'client_execute_steps: __EXECUTE_STEPS__' "${DEPLOY_CONFIG}" || {
    echo "Deploy template lacks the client_execute_steps placeholder: ${DEPLOY_CONFIG}" >&2
    exit 3
}
grep -q 'pretrained_path: "/models/checkpoint"' "${DEPLOY_CONFIG}" || {
    echo "Deploy template must use /models/checkpoint: ${DEPLOY_CONFIG}" >&2
    exit 3
}

RUNTIME_DIR="$(mktemp -d -t kuavo-openpi-overlay.XXXXXX)"
cleanup() {
    rm -rf -- "${RUNTIME_DIR}"
}
trap cleanup EXIT INT TERM

mkdir -p "${RUNTIME_DIR}/assets/${OPENPI_ASSET_ID}"
cp -- "${NORM_STATS_FILE}" "${RUNTIME_DIR}/assets/${OPENPI_ASSET_ID}/norm_stats.json"
sed "s/client_execute_steps: __EXECUTE_STEPS__/client_execute_steps: ${EXECUTE_STEPS}/" \
    "${DEPLOY_CONFIG}" >"${RUNTIME_DIR}/kuavo_env.yaml"

python3 - "${RUNTIME_DIR}/assets/${OPENPI_ASSET_ID}/norm_stats.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
stats = payload.get("norm_stats", payload)
if not isinstance(stats, dict) or not stats:
    raise SystemExit(f"Invalid or empty OpenPI norm stats: {path}")
print(f"Validated OpenPI norm stats: {path} ({len(stats)} entries)")
PY

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
docker image inspect "${IMAGE_NAME}" >/dev/null

echo "Image: ${IMAGE_NAME}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Norm: ${NORM_STATS_FILE}"
echo "OpenPI asset ID: ${OPENPI_ASSET_ID}"
echo "Tokenizer: ${TOKENIZER_FILE}"
echo "Execute steps: ${EXECUTE_STEPS}"
echo "Deploy YAML: ${DEPLOY_CONFIG}"

docker run --rm -it --init \
    --name "${CONTAINER_NAME}" \
    --network host \
    --gpus all \
    -e "ROS_MASTER_URI=${ROS_MASTER_URI}" \
    -e "ROS_IP=${ROS_IP}" \
    -e "KUAVO_MIX_ASSET_ID=${OPENPI_ASSET_ID}" \
    --mount "type=bind,src=${CHECKPOINT_DIR}/params,dst=/models/checkpoint/params,readonly" \
    --mount "type=bind,src=${RUNTIME_DIR}/assets,dst=/models/checkpoint/assets,readonly" \
    --mount "type=bind,src=${TOKENIZER_DIR},dst=/assets/tokenizer,readonly" \
    --mount "type=bind,src=${TOKENIZER_DIR},dst=/mnt/pqssd/pretrained/google/paligemma-3b-pt-224,readonly" \
    --mount "type=bind,src=${RUNTIME_DIR}/kuavo_env.yaml,dst=/root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml,readonly" \
    --mount "type=bind,src=${RUNTIME_DIR}/kuavo_env.yaml,dst=/root/kuavo_data_challenge/configs/deploy/kuavo_env.release.yaml,readonly" \
    --entrypoint bash \
    "${IMAGE_NAME}" \
    -lc '
set -e
test -f /models/checkpoint/params/_METADATA
test -f "/models/checkpoint/assets/${KUAVO_MIX_ASSET_ID}/norm_stats.json"
test -f /assets/tokenizer/tokenizer.model
test -f /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model
echo "OpenPI runtime overlay validation: OK"
echo "Checkpoint: /models/checkpoint"
echo "Norm: /models/checkpoint/assets/${KUAVO_MIX_ASSET_ID}/norm_stats.json"
echo "Tokenizer: $(readlink -f /mnt/pqssd/pretrained/google/paligemma-3b-pt-224/tokenizer.model)"
grep -E "pretrained_path:|client_execute_steps:|openpi_policy_config:" \
  /root/kuavo_data_challenge/configs/deploy/kuavo_env.yaml
exec bash
'
