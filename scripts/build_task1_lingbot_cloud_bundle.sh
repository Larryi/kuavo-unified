#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${ROOT}/kuavo_task1_lingbot_cloud.zip}"
LINGBOT_ROOT="${LINGBOT_ROOT:-${ROOT}/third_party/lingbot-vla}"
LINGBOT_OUTPUT="${2:-${ROOT}/lingbot-vla-cloud.zip}"
EXPECTED_LEROBOT_COMMIT="56b43cc88844cab4f231cf370a6c8eb8103bc9b8"
EXPECTED_LINGBOT_COMMIT="4eb34b7693a0565c67433f8fac9c59a2e67eb60b"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

require_commit() {
    local path="$1" expected="$2" name="$3" actual
    [[ -d "${path}" ]] || { echo "Missing ${name}: ${path}" >&2; exit 2; }
    actual="$(git -C "${path}" rev-parse HEAD)"
    [[ "${actual}" == "${expected}" ]] || {
        echo "${name} must be pinned at ${expected}; found ${actual}" >&2
        exit 2
    }
}

require_commit "${ROOT}/third_party/lerobot" "${EXPECTED_LEROBOT_COMMIT}" "LeRobot"
require_commit "${LINGBOT_ROOT}" "${EXPECTED_LINGBOT_COMMIT}" "LingBot-v1"
if git -C "${LINGBOT_ROOT}" submodule status --recursive | grep -q '^[+-]'; then
    echo "LingBot-v1 nested submodules are missing or at the wrong commit; run git submodule update --init --recursive." >&2
    exit 2
fi

DEST="${TMP}/kuavo_unified_stack"
mkdir -p "${DEST}/configs/policy" "${DEST}/configs/robot_configs" \
    "${DEST}/kuavo_train" "${DEST}/tools" "${DEST}/scripts"
cp -a "${ROOT}/configs/policy/lingbot" "${DEST}/configs/policy/"
cp -a "${ROOT}/configs/robot_configs/kuavo_v2_right_arm.yaml" \
    "${DEST}/configs/robot_configs/"
cp -a "${ROOT}/kuavo_train/lingbot" "${DEST}/kuavo_train/"
mkdir -p "${DEST}/third_party"
cp -a "${ROOT}/third_party/lerobot" "${DEST}/third_party/"
cp -a "${ROOT}/tools/export_lingbot_full_checkpoint.py" "${DEST}/tools/"
cp -a "${ROOT}/scripts/run_task1_lingbot_full_pipeline.sh" "${DEST}/scripts/"
cp -a "${ROOT}/requirements_lingbot_cloud.txt" "${DEST}/"
rm -rf "${DEST}/third_party/lerobot/tests"
find "${DEST}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${DEST}" -type d -name .git -prune -exec rm -rf {} +
find "${DEST}" -type f -name .git -delete
find "${DEST}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

OUTPUT="${OUTPUT}" TMP="${TMP}" python - <<'PY'
import os
import zipfile
from pathlib import Path
root = Path(os.environ["TMP"])
output = Path(os.environ["OUTPUT"]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root))
print(output)
PY

[[ -f "${LINGBOT_ROOT}/lingbotvla/models/auto.py" ]] || {
    echo "Invalid LINGBOT_ROOT: ${LINGBOT_ROOT}" >&2
    exit 2
}
LINGBOT_STAGE="${TMP}/lingbot_stage"
mkdir -p "${LINGBOT_STAGE}/lingbot-vla"
cp -a "${LINGBOT_ROOT}/." "${LINGBOT_STAGE}/lingbot-vla/"
rm -rf \
    "${LINGBOT_STAGE}/lingbot-vla/.git" \
    "${LINGBOT_STAGE}/lingbot-vla/.pytest_cache" \
    "${LINGBOT_STAGE}/lingbot-vla/build" \
    "${LINGBOT_STAGE}/lingbot-vla/dist"
find "${LINGBOT_STAGE}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${LINGBOT_STAGE}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "${LINGBOT_STAGE}" -type f \( \
    -name '.env' -o -name '.netrc' -o -name '*.pem' -o -name '*.key' \
    -o -name 'credentials*.json' \
    \) -delete

OUTPUT="${LINGBOT_OUTPUT}" STAGE="${LINGBOT_STAGE}" python - <<'PY'
import os
import zipfile
from pathlib import Path
root = Path(os.environ["STAGE"])
output = Path(os.environ["OUTPUT"]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root))
print(output)
PY
