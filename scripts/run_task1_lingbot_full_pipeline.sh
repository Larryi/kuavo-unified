#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAINING_TASK="${TRAINING_TASK:-task1}"
exec bash "${ROOT}/scripts/run_lingbot_v1_full_pipeline.sh" "$@"
