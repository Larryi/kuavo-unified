#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LINGBOT_V2_ENV_NAME:-lingbotvla_v2}"
WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-/mnt/pqssd/wheelhouse/flash-attn-v2.8.3}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "require host glibc >= 2.32 for the official flash-attn 2.8.3 wheel"
    echo "conda create -n ${ENV_NAME} python=3.12 pip -y"
    echo "conda run -n ${ENV_NAME} python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0"
    echo "conda run -n ${ENV_NAME} python docker/flash_attn_wheel.py --version 2.8.3 --download-dir ${WHEEL_DIR}"
    echo "bash third_party/lingbot-vla-v2/tools/create_train_env.sh --env-name ${ENV_NAME} --resume --flash-attn-wheel <resolved-wheel>"
    exit 0
fi

# The official v2.8.3 Linux wheel references GLIBC_2.32. The current ROS
# Noetic/Ubuntu 20.04 host has glibc 2.31, even though its NVIDIA driver is
# sufficiently new. Fail before creating a partially configured environment.
host_glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if [[ "$(printf '%s\n' "2.32" "${host_glibc}" | sort -V | head -n1)" != "2.32" ]]; then
    echo "LingBot-v2 setup requires glibc >= 2.32; host has ${host_glibc}." >&2
    echo "Run this script on Ubuntu 22.04/cloud, then package the environment." >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.12 pip -y
fi

# Install the exact Torch stack first, so wheel resolution reads Torch's CUDA
# runtime and ABI instead of guessing from the host driver's nvidia-smi output.
conda run -n "${ENV_NAME}" python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    torchdata==0.11.0 torchcodec==0.6.0

resolver_output="$(
    conda run -n "${ENV_NAME}" python \
        "${REPO_ROOT}/docker/flash_attn_wheel.py" \
        --version 2.8.3 \
        --download-dir "${WHEEL_DIR}"
)"
printf '%s\n' "${resolver_output}"
wheel_path="$(
    python -c 'import json,sys; print(json.load(sys.stdin)["path"])' \
        <<<"${resolver_output}"
)"

bash "${REPO_ROOT}/third_party/lingbot-vla-v2/tools/create_train_env.sh" \
    --env-name "${ENV_NAME}" \
    --resume \
    --flash-attn-wheel "${wheel_path}"
