#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-kdc_v0}"
CONTAINER_NAME="${CONTAINER_NAME:-kdc_v0}"
IMAGE_TAR="${IMAGE_TAR:-${IMAGE_NAME}.tar}"
ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
ROS_IP="${ROS_IP:-127.0.0.1}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Starting existing container ${CONTAINER_NAME}; no image or container is deleted."
    exec docker start -ai "${CONTAINER_NAME}"
fi

if ! docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1; then
    if [[ ! -s "${IMAGE_TAR}" ]]; then
        echo "Image ${IMAGE_NAME}:latest is absent and ${IMAGE_TAR} was not found." >&2
        exit 2
    fi
    docker load -i "${IMAGE_TAR}"
fi

exec docker run --gpus all -it \
    --network host \
    -e "ROS_MASTER_URI=${ROS_MASTER_URI}" \
    -e "ROS_IP=${ROS_IP}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}:latest" bash
