#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_BACKEND:?Set MODEL_BACKEND}"
: "${TRAINING_TASK:?Set TRAINING_TASK}"
: "${VAST_SSH_HOST:?Set VAST_SSH_HOST}"
: "${VAST_SSH_PORT:?Set VAST_SSH_PORT}"
: "${VAST_SSH_USER:=root}"
: "${WORK_ROOT:=/workspace/kuavo_runs/${MODEL_BACKEND}}"
: "${REMOTE_ROOT:=/workspace/kuavo_unified_stack}"
: "${TAIL_LINES:=30}"
: "${WATCH_SECONDS:=0}"

[[ "${TAIL_LINES}" =~ ^[1-9][0-9]*$ ]] || { echo "TAIL_LINES must be positive" >&2; exit 2; }
[[ "${WATCH_SECONDS}" =~ ^[0-9]+$ ]] || { echo "WATCH_SECONDS must be non-negative" >&2; exit 2; }
remote="${VAST_SSH_USER}@${VAST_SSH_HOST}"
ssh_args=(-p "${VAST_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

snapshot() {
  ssh "${ssh_args[@]}" "${remote}" bash -s -- \
    "${WORK_ROOT}" "${REMOTE_ROOT}" "${TRAINING_TASK}" "${MODEL_BACKEND}" "${TAIL_LINES}" <<'REMOTE'
set -euo pipefail
work_root="$1"
remote_root="$2"
task="$3"
backend="$4"
tail_lines="$5"
status_file="$(find "${work_root}/logs" -type f -name status.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -n "${status_file}" && -f "${status_file}" ]]; then
  echo "STATUS ${status_file}"
  cat "${status_file}"
  pipeline_log="${status_file%/status.json}/pipeline.log"
  if [[ -f "${pipeline_log}" ]]; then
    echo "PIPELINE_TAIL ${pipeline_log}"
    tail -n "${tail_lines}" "${pipeline_log}"
  fi
else
  launcher_log="${remote_root}/logs/${task}-${backend}-launcher.log"
  echo "STATUS pending/no status.json yet"
  if [[ -f "${launcher_log}" ]]; then
    echo "LAUNCHER_TAIL ${launcher_log}"
    tail -n "${tail_lines}" "${launcher_log}"
  fi
fi
echo "GPU"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null || true
REMOTE
}

while true; do
  snapshot
  (( WATCH_SECONDS > 0 )) || break
  sleep "${WATCH_SECONDS}"
done
