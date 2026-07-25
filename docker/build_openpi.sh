#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly OPENPI_ROOT="${REPO_ROOT}/third_party/openpi-kuavo"
readonly EXPECTED_OPENPI_COMMIT="8e9c6c6cca8d0cef48a4084e3a3071aa98011a48"

IMAGE_NAME="${IMAGE_NAME:-kuavo-openpi}"
CLASSIC_BASE_IMAGE="${CLASSIC_BASE_IMAGE:-kuavo-classic:latest}"
BUILD_LOG="${BUILD_LOG:-/tmp/${IMAGE_NAME}.build.log}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${OPENPI_ROOT}/uv.lock" ]]; then
    echo "OpenPI submodule is not initialized: ${OPENPI_ROOT}" >&2
    exit 2
fi
actual_openpi_commit="$(git -C "${OPENPI_ROOT}" rev-parse HEAD)"
if [[ "${actual_openpi_commit}" != "${EXPECTED_OPENPI_COMMIT}" ]]; then
    echo "OpenPI pin mismatch: expected ${EXPECTED_OPENPI_COMMIT}, got ${actual_openpi_commit}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" ]] && ! docker image inspect "${CLASSIC_BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Classic ROS base image is missing: ${CLASSIC_BASE_IMAGE}" >&2
    echo "Build it first with CLASSIC_ENV_ARCHIVE=... docker/build_classic.sh" >&2
    exit 2
fi

build_command=(
    docker buildx build
    --load
    --progress=plain
    --build-arg "CLASSIC_BASE_IMAGE=${CLASSIC_BASE_IMAGE}"
    --build-context "openpi_src=${OPENPI_ROOT}"
    -f "${REPO_ROOT}/Dockerfile.openpi"
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
