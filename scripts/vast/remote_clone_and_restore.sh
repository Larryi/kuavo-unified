#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL=""
GIT_REF=""
REMOTE_ROOT="/workspace/kuavo_unified_stack"
LAUNCH=1

usage() {
  cat <<'EOF'
Usage:
  remote_clone_and_restore.sh \
    --repo-url https://github.com/OWNER/REPO.git \
    --git-ref BRANCH_OR_TAG \
    [--remote-root /workspace/kuavo_unified_stack] [--no-launch]

This script is intended to be copied to a new VastAI instance. It clones the
main repository, checks out the requested ref, initializes pinned submodules,
records the resolved commits, then optionally starts restore_and_launch.sh.
EOF
}

while (($#)); do
  case "$1" in
    --repo-url) REPO_URL="${2:?--repo-url requires a value}"; shift 2 ;;
    --git-ref) GIT_REF="${2:?--git-ref requires a value}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:?--remote-root requires a value}"; shift 2 ;;
    --no-launch) LAUNCH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${REPO_URL}" && -n "${GIT_REF}" ]] || { usage >&2; exit 2; }
[[ "${REMOTE_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
  echo "Unsafe remote root: ${REMOTE_ROOT}" >&2
  exit 2
}
command -v git >/dev/null || {
  echo "git is required in the VastAI base image." >&2
  exit 3
}

if [[ -e "${REMOTE_ROOT}" && ! -d "${REMOTE_ROOT}/.git" ]]; then
  echo "Remote root exists but is not a Git repository: ${REMOTE_ROOT}" >&2
  exit 3
fi

if [[ ! -d "${REMOTE_ROOT}/.git" ]]; then
  mkdir -p "$(dirname -- "${REMOTE_ROOT}")"
  git clone --branch "${GIT_REF}" --recurse-submodules \
    "${REPO_URL}" "${REMOTE_ROOT}"
else
  [[ -z "$(git -C "${REMOTE_ROOT}" status --short)" ]] || {
    echo "Existing remote repository is dirty; refusing to overwrite it." >&2
    exit 3
  }
  git -C "${REMOTE_ROOT}" fetch origin "${GIT_REF}"
  git -C "${REMOTE_ROOT}" checkout --detach FETCH_HEAD
fi

git -C "${REMOTE_ROOT}" submodule sync --recursive
git -C "${REMOTE_ROOT}" submodule update --init --recursive --jobs 8

main_commit="$(git -C "${REMOTE_ROOT}" rev-parse HEAD)"
submodule_status="$(git -C "${REMOTE_ROOT}" submodule status --recursive)"
if grep -Eq '^[+-U]' <<<"${submodule_status}"; then
  echo "One or more submodules did not resolve to the pinned commit:" >&2
  echo "${submodule_status}" >&2
  exit 4
fi

MAIN_COMMIT="${main_commit}" GIT_REF="${GIT_REF}" REPO_URL="${REPO_URL}" \
REMOTE_ROOT="${REMOTE_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ["REMOTE_ROOT"])
raw = subprocess.run(
    ["git", "-C", str(root), "submodule", "status", "--recursive"],
    text=True,
    capture_output=True,
    check=True,
).stdout
submodules = []
for line in raw.splitlines():
    fields = line[1:].split()
    submodules.append({"commit": fields[0], "path": fields[1]})
(root / ".vast_git_manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "sync_mode": "remote-git-clone",
            "repo_url": os.environ["REPO_URL"],
            "requested_ref": os.environ["GIT_REF"],
            "main_commit": os.environ["MAIN_COMMIT"],
            "submodules": submodules,
        },
        indent=2,
    )
    + "\n"
)
PY

echo "Git checkout ready: ${REPO_URL}@${main_commit}"
echo "${submodule_status}"
if [[ "${LAUNCH}" == "1" ]]; then
  exec bash "${REMOTE_ROOT}/scripts/vast/restore_and_launch.sh"
fi
