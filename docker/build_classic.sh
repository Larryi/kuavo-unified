#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_LEROBOT_COMMIT="56b43cc88844cab4f231cf370a6c8eb8103bc9b8"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-kuavo-classic}"
ENV_ARCHIVE="${CLASSIC_ENV_ARCHIVE:-}"
BUILD_LOG="${BUILD_LOG:-/tmp/${IMAGE_NAME}.build.log}"
DRY_RUN="${DRY_RUN:-0}"

actual_lerobot_commit="$(git -C "${REPO_ROOT}/third_party/lerobot" rev-parse HEAD)"
if [[ "${actual_lerobot_commit}" != "${EXPECTED_LEROBOT_COMMIT}" ]]; then
    echo "LeRobot pin mismatch: expected ${EXPECTED_LEROBOT_COMMIT}, got ${actual_lerobot_commit}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${ENV_ARCHIVE}" || ! -s "${ENV_ARCHIVE}" ]]; then
        echo "Set CLASSIC_ENV_ARCHIVE to a non-empty myenv.tar.gz." >&2
        exit 2
    fi
    if [[ "$(basename "${ENV_ARCHIVE}")" != "myenv.tar.gz" ]]; then
        echo "CLASSIC_ENV_ARCHIVE must be named myenv.tar.gz." >&2
        exit 2
    fi
    ENV_CONTEXT="$(cd "$(dirname "${ENV_ARCHIVE}")" && pwd)"
else
    ENV_CONTEXT="/path/to/classic-env-context"
fi

build_command=(
    docker buildx build
    --load
    --progress=plain
    --build-context "classic_env=${ENV_CONTEXT}"
    -f "${REPO_ROOT}/Dockerfile"
    -t "${IMAGE_NAME}:latest"
    "${REPO_ROOT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${build_command[@]}"
    printf '\n'
    exit 0
fi

echo "Building ${IMAGE_NAME}; full output: ${BUILD_LOG}"
if ! "${build_command[@]}" >"${BUILD_LOG}" 2>&1; then
    echo "Build failed. Last 80 log lines:" >&2
    tail -n 80 "${BUILD_LOG}" >&2
    exit 1
fi

echo "Build complete: ${IMAGE_NAME}:latest"
