#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAINING_TASK=""
MODEL_BACKEND=""
ENV_FILE=""
SSH_HOST="${VAST_SSH_HOST:-}"
SSH_PORT="${VAST_SSH_PORT:-}"
RESUME_MODE="none"
RESUME_REPO=""
RESUME_RUN_ID=""
DRY_RUN=0
SYNC_ONLY=0
DETACH=1

usage() {
  cat <<'EOF'
Usage:
  scripts/vast/launch_job.sh \
    --task task1|task2|task3 \
    --algorithm openpi|lingbot-v1|lingbot-v2|dp|act \
    --env-file /secure/job.env \
    --host VAST_HOST --port VAST_PORT
    [--resume-repo owner/full-checkpoint-repo --resume-run-id RUN_ID]
    [--foreground] [--sync-only] [--dry-run]

Each invocation binds one VastAI instance to exactly one task and algorithm.
The private env file supplies HF_TOKEN, DATASET_REPO, MODEL_REPO, W&B,
ServerChan and Vast credentials.
EOF
}

while (($#)); do
  case "$1" in
    --task) TRAINING_TASK="${2:?--task requires a value}"; shift 2 ;;
    --algorithm) MODEL_BACKEND="${2:?--algorithm requires a value}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file requires a path}"; shift 2 ;;
    --host) SSH_HOST="${2:?--host requires a value}"; shift 2 ;;
    --port) SSH_PORT="${2:?--port requires a value}"; shift 2 ;;
    --resume-repo) RESUME_REPO="${2:?--resume-repo requires a value}"; RESUME_MODE=hf; shift 2 ;;
    --resume-run-id) RESUME_RUN_ID="${2:?--resume-run-id requires a value}"; shift 2 ;;
    --foreground) DETACH=0; shift ;;
    --sync-only) SYNC_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${TRAINING_TASK}" && -n "${MODEL_BACKEND}" ]] || {
  usage >&2
  exit 2
}
if [[ "${RESUME_MODE}" == "hf" && -z "${RESUME_RUN_ID}" ]]; then
  echo "--resume-repo also requires --resume-run-id" >&2
  exit 2
fi
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -n "${ENV_FILE}" && -n "${SSH_HOST}" && -n "${SSH_PORT}" ]] || {
    echo "A real launch requires --env-file, --host and --port." >&2
    exit 2
  }
fi

export TRAINING_TASK MODEL_BACKEND
export VAST_ENV_FILE="${ENV_FILE}"
export VAST_SSH_HOST="${SSH_HOST}"
export VAST_SSH_PORT="${SSH_PORT}"
export RESUME_MODE RESUME_REPO RESUME_RUN_ID
export DRY_RUN SYNC_ONLY DETACH
exec "${ROOT}/scripts/vast/launch.sh"
