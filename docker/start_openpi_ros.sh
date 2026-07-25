#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/root/kuavo_data_challenge"
SERVER_ARGS="${SERVER_ARGS:-}"
OPENPI_POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-}"
OPENPI_POLICY_DIR="${OPENPI_POLICY_DIR:-}"
OPENPI_PORT="${OPENPI_PORT:-8000}"
SERVER_LOG="${OPENPI_SERVER_LOG:-/tmp/openpi-policy-server.log}"
STARTUP_TIMEOUT_S="${OPENPI_STARTUP_TIMEOUT_S:-300}"

if [[ -z "${SERVER_ARGS}" && ( -z "${OPENPI_POLICY_CONFIG}" || -z "${OPENPI_POLICY_DIR}" ) ]]; then
    echo "Set OPENPI_POLICY_CONFIG and OPENPI_POLICY_DIR (recommended), or SERVER_ARGS." >&2
    exit 2
fi

server_pid=""
client_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${client_pid}" ]]; then
        kill "${client_pid}" 2>/dev/null || true
        wait "${client_pid}" 2>/dev/null || true
    fi
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
source /opt/ros/noetic/setup.bash
source "${REPO_ROOT}/myenv/bin/activate"
# Prefer structured variables so shell quoting cannot silently drop required
# Tyro options. Keep SERVER_ARGS for backwards compatibility.
if [[ -n "${SERVER_ARGS}" ]]; then
    mapfile -d '' -t server_argv < <(
        python -c \
            'import os, shlex, sys; [sys.stdout.buffer.write(arg.encode() + b"\0") for arg in shlex.split(os.environ["SERVER_ARGS"])]'
    )
else
    server_argv=(
        policy:checkpoint
        "--policy.config=${OPENPI_POLICY_CONFIG}"
        "--policy.dir=${OPENPI_POLICY_DIR}"
        "--port=${OPENPI_PORT}"
    )
fi
scripts/kuavo_openpi serve "${server_argv[@]}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!
echo "OpenPI policy server PID ${server_pid}; log: ${SERVER_LOG}"

if [[ "$#" -eq 0 ]]; then
    wait "${server_pid}"
    exit $?
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until bash -c "</dev/tcp/127.0.0.1/${OPENPI_PORT}" 2>/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "OpenPI policy server exited before accepting connections." >&2
        tail -n 40 "${SERVER_LOG}" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for OpenPI server on port ${OPENPI_PORT}." >&2
        exit 1
    fi
    sleep 1
done

echo "OpenPI server is listening; starting ROS command: $*"
"$@" &
client_pid=$!
wait "${client_pid}"
