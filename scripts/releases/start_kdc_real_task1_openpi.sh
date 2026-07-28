#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="kdc_real_task1_openpi"
CONTAINER_NAME="kdc_real_task1_openpi"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAR="${SCRIPT_DIR}/${IMAGE_NAME}.tar"

if [[ -n "$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")" ]]; then
    echo "Container exists. Removing..."
    docker rm -f "${CONTAINER_NAME}"
fi

EXISTING_IMAGE="$(docker images -q "${IMAGE_NAME}:latest")"
if [[ -n "${EXISTING_IMAGE}" ]]; then
    echo "Image ${IMAGE_NAME}:latest already exists. Removing..."
    docker rmi -f "${IMAGE_NAME}:latest"
fi

if [[ -f "${IMAGE_TAR}" ]]; then
    echo "Loading image from ${IMAGE_TAR}..."
    docker load -i "${IMAGE_TAR}"
else
    echo "Error: ${IMAGE_TAR} not found!"
    exit 1
fi

docker run --gpus all -it \
    --net=host \
    -e ROS_MASTER_URI=http://kuavo_master:11311 \
    -e ROS_IP=192.168.26.10 \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:latest" bash
