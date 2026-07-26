#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${VAST_BOOTSTRAP_ROOT:=/workspace/kuavo_bootstrap}"
: "${JOB_ENV:=${ROOT}/.secrets/interactive-job.env}"
: "${JOB_MANIFEST:=${ROOT}/logs/interactive-job.json}"
: "${CREDENTIALS_FILE:=${ROOT}/.secrets/vast-credentials.json}"
: "${PIP_INDEX_URL:=https://pypi.org/simple}"

command -v python3 >/dev/null || {
  echo "python3 is required in the VastAI base image." >&2
  exit 3
}

if [[ ! -x "${VAST_BOOTSTRAP_ROOT}/bin/python" ]]; then
  python3 -m venv "${VAST_BOOTSTRAP_ROOT}"
fi
"${VAST_BOOTSTRAP_ROOT}/bin/python" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --upgrade pip huggingface_hub

export VAST_BOOTSTRAP_PYTHON="${VAST_BOOTSTRAP_ROOT}/bin/python"
"${VAST_BOOTSTRAP_PYTHON}" "${ROOT}/tools/vast_job_wizard.py" \
  --output-env "${JOB_ENV}" \
  --output-manifest "${JOB_MANIFEST}" \
  --credentials-file "${CREDENTIALS_FILE}"

echo "即将使用以下无凭据配置："
"${VAST_BOOTSTRAP_PYTHON}" -m json.tool "${JOB_MANIFEST}"
read -r -p "确认恢复算法环境并启动训练？ [y/N] " confirm
[[ "${confirm}" == "y" || "${confirm}" == "Y" ]] || {
  echo "已保存配置但未启动。稍后可执行："
  echo "  set -a; source '${JOB_ENV}'; set +a; bash scripts/vast/run_backend.sh"
  exit 0
}

set -a
source "${JOB_ENV}"
set +a
export CODE_DIR="${ROOT}"
exec bash "${ROOT}/scripts/vast/run_backend.sh"
