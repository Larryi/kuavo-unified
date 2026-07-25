#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_LEROBOT_COMMIT="56b43cc88844cab4f231cf370a6c8eb8103bc9b8"
readonly RESNET18_FILENAME="resnet18-f37072fd.pth"
readonly RESNET18_SHA256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
readonly RESNET18_URL="https://download.pytorch.org/models/${RESNET18_FILENAME}"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-kuavo-classic}"
ENV_ARCHIVE="${CLASSIC_ENV_ARCHIVE:-}"
RESNET18_CHECKPOINT="${RESNET18_CHECKPOINT:-${XDG_CACHE_HOME:-${HOME}/.cache}/torch/hub/checkpoints/${RESNET18_FILENAME}}"
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

    if [[ ! -s "${RESNET18_CHECKPOINT}" ]]; then
        echo "Downloading ${RESNET18_FILENAME} once into the host Torch cache."
        mkdir -p "$(dirname "${RESNET18_CHECKPOINT}")"
        checkpoint_tmp="${RESNET18_CHECKPOINT}.part"
        curl --fail --location \
            --retry 10 --retry-delay 3 --retry-all-errors \
            --continue-at - \
            --output "${checkpoint_tmp}" \
            "${RESNET18_URL}"
        mv "${checkpoint_tmp}" "${RESNET18_CHECKPOINT}"
    fi
    echo "${RESNET18_SHA256}  ${RESNET18_CHECKPOINT}" | sha256sum --check -

    TORCH_CHECKPOINT_CONTEXT="$(mktemp -d -t kuavo-resnet18-XXXXXX)"
    trap 'rm -rf "${TORCH_CHECKPOINT_CONTEXT}"' EXIT
    cp "${RESNET18_CHECKPOINT}" "${TORCH_CHECKPOINT_CONTEXT}/${RESNET18_FILENAME}"
else
    ENV_CONTEXT="/path/to/classic-env-context"
    TORCH_CHECKPOINT_CONTEXT="/path/to/torch-checkpoint-context"
fi

build_command=(
    docker buildx build
    --load
    --progress=plain
    --build-context "classic_env=${ENV_CONTEXT}"
    --build-context "torch_checkpoints=${TORCH_CHECKPOINT_CONTEXT}"
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
