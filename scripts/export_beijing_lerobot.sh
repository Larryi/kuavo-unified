#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
: "${OUTPUT_BASE:?Set OUTPUT_BASE to the parent directory for exported datasets}"
OVERWRITE="${OVERWRITE:-0}"
LOG_DIR="${LOG_DIR:-$OUTPUT_BASE/logs}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "/home/larry/miniconda3/envs/kdc_dev/bin/python" ]]; then
        PYTHON_BIN="/home/larry/miniconda3/envs/kdc_dev/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

cd "$REPO_ROOT"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/beijing-lerobot-hf-cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

"$PYTHON_BIN" -c \
    "import cv2, lerobot, numpy, pyarrow, rosbag; print('Python dependency check passed')"

extra_args=()
if [[ "$OVERWRITE" == "1" ]]; then
    extra_args+=(--overwrite)
fi

run_export() {
    local task="$1"
    local count="$2"
    local output_name="$3"

    "$PYTHON_BIN" -u \
        tools/export_beijing_header_aligned_lerobot.py \
        --task "$task" \
        --whitelist "outputs/audit/beijing_rosbags/${task}_causal_whitelist.txt" \
        --output-root "$OUTPUT_BASE/$output_name" \
        --repo-id "local/beijing-${task}-${count}" \
        "${extra_args[@]}"
}

launch_export() {
    local task="$1"
    local count="$2"
    local output_name="$3"
    local log_file="$LOG_DIR/${task}.log"

    (
        set -o pipefail
        run_export "$task" "$count" "$output_name" 2>&1 \
            | tee "$log_file" \
            | sed -u "s/^/[${task}] /"
    ) &
    pids+=("$!")
    task_names+=("$task")
}

pids=()
task_names=()
launch_export task1 220 lerobot_task1_220
launch_export task2 208 lerobot_task2_208
launch_export task3 234 lerobot_task3_234

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "${task_names[$index]} completed successfully"
    else
        echo "${task_names[$index]} failed; see $LOG_DIR/${task_names[$index]}.log" >&2
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    echo "One or more Beijing exports failed." >&2
    exit "$status"
fi

echo "All Beijing LeRobot datasets completed under: $OUTPUT_BASE"
