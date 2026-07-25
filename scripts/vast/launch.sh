#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_BACKEND:?Set MODEL_BACKEND to dp, act, openpi, lingbot-v1, or lingbot-v2}"
: "${TRAINING_TASK:?Set TRAINING_TASK to task1, task2, or task3}"
: "${DRY_RUN:=0}"
: "${VAST_SSH_USER:=root}"
: "${REMOTE_ROOT:=/workspace/kuavo_unified_stack}"
: "${REMOTE_ENV_FILE:=${REMOTE_ROOT}/.secrets/${MODEL_BACKEND}.env}"
: "${SYNC_ONLY:=0}"
: "${DETACH:=1}"
: "${RESUME_MODE:=none}"
: "${RESUME_REPO:=}"
: "${RESUME_RUN_ID:=}"

case "${MODEL_BACKEND}" in
  dp|act|openpi|lingbot-v1|lingbot-v2) ;;
  *) echo "Unsupported MODEL_BACKEND=${MODEL_BACKEND}" >&2; exit 2 ;;
esac
source "${ROOT}/scripts/vast/job_profile.sh"
apply_vast_job_profile
for value in DRY_RUN SYNC_ONLY DETACH; do
  [[ "${!value}" == "0" || "${!value}" == "1" ]] || {
    echo "${value} must be 0 or 1, got ${!value}" >&2
    exit 2
  }
done
for remote_path in "${REMOTE_ROOT}" "${REMOTE_ENV_FILE}"; do
  [[ "${remote_path}" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "Unsafe remote path: ${remote_path}" >&2
    exit 2
  }
done
[[ "${RESUME_MODE}" == "none" || "${RESUME_MODE}" == "hf" ]] || {
  echo "RESUME_MODE must be none or hf" >&2
  exit 2
}
if [[ "${RESUME_MODE}" == "hf" ]]; then
  [[ "${RESUME_REPO}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || {
    echo "Unsafe or missing RESUME_REPO" >&2
    exit 2
  }
  [[ "${RESUME_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Unsafe or missing RESUME_RUN_ID" >&2
    exit 2
  }
fi

echo "Backend: ${MODEL_BACKEND}"
echo "Task: ${TRAINING_TASK}"
echo "Remote code: ${REMOTE_ROOT}"
echo "Remote private env: ${REMOTE_ENV_FILE}"
if [[ "${DRY_RUN}" == "1" ]]; then
  MODEL_BACKEND="${MODEL_BACKEND}" TRAINING_TASK="${TRAINING_TASK}" \
    DRY_RUN=1 CODE_DIR="${REMOTE_ROOT}" \
    bash "${ROOT}/scripts/vast/run_backend.sh"
  echo "Launcher dry run complete; SSH, rsync, secret upload, and remote execution were skipped."
  exit 0
fi

: "${VAST_SSH_HOST:?Set VAST_SSH_HOST}"
: "${VAST_SSH_PORT:?Set VAST_SSH_PORT}"
: "${VAST_ENV_FILE:?Set VAST_ENV_FILE to a private local env file}"
[[ -f "${VAST_ENV_FILE}" ]] || { echo "Private env file not found: ${VAST_ENV_FILE}" >&2; exit 3; }
env_mode="$(stat -c '%a' "${VAST_ENV_FILE}")"
[[ "${env_mode}" == "600" || "${env_mode}" == "400" ]] || {
  echo "VAST_ENV_FILE must have mode 600 or 400, got ${env_mode}" >&2
  exit 3
}

ssh_args=(-p "${VAST_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
scp_args=(-P "${VAST_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
remote="${VAST_SSH_USER}@${VAST_SSH_HOST}"
ssh "${ssh_args[@]}" "${remote}" "mkdir -p '${REMOTE_ROOT}' '$(dirname "${REMOTE_ENV_FILE}")' '${REMOTE_ROOT}/logs'"
rsync -az --info=progress2 \
  -e "ssh ${ssh_args[*]}" \
  --exclude '.git/' \
  --exclude '.secrets/' \
  --exclude '.venv/' \
  --exclude '*.env' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude 'checkpoints/' \
  --exclude 'wandb/' \
  --exclude '*.tar' \
  --exclude '*.tar.gz' \
  "${ROOT}/" "${remote}:${REMOTE_ROOT}/"
scp "${scp_args[@]}" "${VAST_ENV_FILE}" "${remote}:${REMOTE_ENV_FILE}"
ssh "${ssh_args[@]}" "${remote}" "chmod 600 '${REMOTE_ENV_FILE}'"
echo "Code and private environment synchronized to ${remote}"

[[ "${SYNC_ONLY}" == "0" ]] || exit 0
remote_log="${REMOTE_ROOT}/logs/${TRAINING_TASK}-${MODEL_BACKEND}-launcher.log"
remote_command="cd '${REMOTE_ROOT}' && set -a && source '${REMOTE_ENV_FILE}' && set +a && export MODEL_BACKEND='${MODEL_BACKEND}' TRAINING_TASK='${TRAINING_TASK}' CODE_DIR='${REMOTE_ROOT}' RESUME_MODE='${RESUME_MODE}' RESUME_REPO='${RESUME_REPO}' RESUME_RUN_ID='${RESUME_RUN_ID}'"
if [[ "${DETACH}" == "1" ]]; then
  ssh "${ssh_args[@]}" "${remote}" \
    "${remote_command} && nohup bash scripts/vast/run_backend.sh >>'${remote_log}' 2>&1 </dev/null & echo \$!"
  echo "Remote pipeline started in background. Log: ${remote_log}"
else
  ssh -t "${ssh_args[@]}" "${remote}" \
    "${remote_command} && exec bash scripts/vast/run_backend.sh"
fi
