#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-kdc_real_task1_lingbot}"
SECRET_FILE="${KUAVO_DIAG_ENV:-/home/larry/kuavo_diag.env}"
BUILD_LOG="${BUILD_LOG:-/tmp/${IMAGE_NAME}.build.log}"

if [[ ! -s "${SECRET_FILE}" ]]; then
    echo "Diagnostics secret is missing or empty: ${SECRET_FILE}" >&2
    exit 2
fi

echo "Building ${IMAGE_NAME}; full output: ${BUILD_LOG}"
if ! docker buildx build \
    --load \
    --progress=plain \
    --secret "id=kuavo_diag_env,src=${SECRET_FILE}" \
    -f Dockerfile.lingbot \
    -t "${IMAGE_NAME}:latest" \
    . >"${BUILD_LOG}" 2>&1; then
    echo "Build failed. Last 80 log lines:" >&2
    tail -n 80 "${BUILD_LOG}" >&2
    exit 1
fi

echo "Build complete: ${IMAGE_NAME}:latest"
