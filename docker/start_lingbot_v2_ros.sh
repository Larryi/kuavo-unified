#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/root/kuavo_data_challenge"
LINGBOT_V2_POLICY_DIR="${LINGBOT_V2_POLICY_DIR:-/models/checkpoint}"
LINGBOT_V2_QWEN_DIR="${LINGBOT_V2_QWEN_DIR:-/assets/qwen}"
LINGBOT_V2_NORM_STATS="${LINGBOT_V2_NORM_STATS:-/assets/norm_stats/norm_stats.json}"
LINGBOT_V2_ROBOT_NAME="${LINGBOT_V2_ROBOT_NAME:-kuavo_v2_right_arm}"
LINGBOT_V2_TASK_PROMPT="${LINGBOT_V2_TASK_PROMPT:-Pick and place the target object}"
LINGBOT_V2_PORT="${LINGBOT_V2_PORT:-8000}"
SERVER_LOG="${LINGBOT_V2_SERVER_LOG:-/tmp/lingbot-v2-policy-server.log}"
STARTUP_TIMEOUT_S="${LINGBOT_V2_STARTUP_TIMEOUT_S:-300}"

for required in "${LINGBOT_V2_POLICY_DIR}" "${LINGBOT_V2_QWEN_DIR}" "${LINGBOT_V2_NORM_STATS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required LingBot-v2 asset does not exist: ${required}" >&2
        exit 2
    fi
done

server_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
/opt/kuavo-env/bin/python tools/policy_worker.py \
    --backend lingbot_v2 \
    --policy-path "${LINGBOT_V2_POLICY_DIR}" \
    --lingbot-root "${REPO_ROOT}/third_party/lingbot-vla-v2" \
    --qwen-path "${LINGBOT_V2_QWEN_DIR}" \
    --norm-stats-file "${LINGBOT_V2_NORM_STATS}" \
    --robot-name "${LINGBOT_V2_ROBOT_NAME}" \
    --task-prompt "${LINGBOT_V2_TASK_PROMPT}" \
    --host 127.0.0.1 \
    --port "${LINGBOT_V2_PORT}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!
echo "LingBot-v2 policy server PID ${server_pid}; log: ${SERVER_LOG}"

if [[ "$#" -eq 0 ]]; then
    wait "${server_pid}"
    exit $?
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until bash -c "</dev/tcp/127.0.0.1/${LINGBOT_V2_PORT}" 2>/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "LingBot-v2 policy server exited before accepting connections." >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for LingBot-v2 server on port ${LINGBOT_V2_PORT}." >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 1
done

set +u
source /opt/ros/noetic/setup.bash
source "${REPO_ROOT}/myenv/bin/activate"
set -u
echo "LingBot-v2 server is listening; starting ROS command: $*"
"$@"
