#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPECTED_LINGBOT_V2_COMMIT="0f48206a34d3bd24454f0cc2671397a1c9065d4b"
IMAGE_NAME="${IMAGE_NAME:-kuavo-lingbot-v2}"
CLASSIC_BASE_IMAGE="${CLASSIC_BASE_IMAGE:-kuavo-classic:latest}"
ENV_ARCHIVE="${LINGBOT_V2_ENV_ARCHIVE:-}"
CUDA_BUILDER_IMAGE="${LINGBOT_V2_CUDA_BUILDER_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu20.04}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
BUILD_LOG="${BUILD_LOG:-/tmp/${IMAGE_NAME}.build.log}"
DRY_RUN="${DRY_RUN:-0}"

actual_lingbot_v2_commit="$(git -C "${REPO_ROOT}/third_party/lingbot-vla-v2" rev-parse HEAD)"
if [[ "${actual_lingbot_v2_commit}" != "${EXPECTED_LINGBOT_V2_COMMIT}" ]]; then
    echo "LingBot-v2 pin mismatch: expected ${EXPECTED_LINGBOT_V2_COMMIT}, got ${actual_lingbot_v2_commit}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    if ! docker image inspect "${CLASSIC_BASE_IMAGE}" >/dev/null 2>&1; then
        echo "Classic ROS base image not found: ${CLASSIC_BASE_IMAGE}" >&2
        echo "Build it first with docker/build_classic.sh." >&2
        exit 2
    fi
    if [[ -z "${ENV_ARCHIVE}" || ! -s "${ENV_ARCHIVE}" ]]; then
        echo "Set LINGBOT_V2_ENV_ARCHIVE to a non-empty myenv.tar.gz." >&2
        exit 2
    fi
    if [[ "$(basename "${ENV_ARCHIVE}")" != "myenv.tar.gz" ]]; then
        echo "LINGBOT_V2_ENV_ARCHIVE must be named myenv.tar.gz." >&2
        exit 2
    fi
    ENV_CONTEXT="$(cd "$(dirname "${ENV_ARCHIVE}")" && pwd)"
else
    ENV_CONTEXT="/path/to/lingbot-v2-env-context"
fi

build_command=(
    docker buildx build
    --load
    --progress=plain
    --build-arg "CLASSIC_BASE_IMAGE=${CLASSIC_BASE_IMAGE}"
    --build-arg "LINGBOT_V2_CUDA_BUILDER_IMAGE=${CUDA_BUILDER_IMAGE}"
    --build-arg "FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION}"
    --build-arg "FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}"
    --build-context "lingbot_v2_env=${ENV_CONTEXT}"
    --build-context "lingbot_v2_source=${REPO_ROOT}/third_party/lingbot-vla-v2"
    -f "${REPO_ROOT}/Dockerfile.lingbot_v2"
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
