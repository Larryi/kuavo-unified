#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LINGBOT_V2_ENV_NAME:-lingbotvla_v2}"
WHEEL_DIR="${FLASH_ATTN_WHEEL_DIR:-/mnt/pqssd/wheelhouse/flash-attn-v2.8.3}"
DRY_RUN="${DRY_RUN:-0}"
FLASH_ATTN_INSTALL_MODE="${FLASH_ATTN_INSTALL_MODE:-auto}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "FLASH_ATTN_INSTALL_MODE=auto: official wheel on glibc >= 2.32, source build otherwise"
    echo "FLASH_ATTN_INSTALL_MODE=defer: omit flash-attn only for an archive consumed by Dockerfile.lingbot_v2"
    echo "conda create -n ${ENV_NAME} python=3.12 pip -y"
    echo "conda run -n ${ENV_NAME} python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0"
    echo "when compatible: conda run -n ${ENV_NAME} python docker/flash_attn_wheel.py --version 2.8.3 --download-dir ${WHEEL_DIR}"
    echo "bash third_party/lingbot-vla-v2/tools/create_train_env.sh --env-name ${ENV_NAME} --resume [--flash-attn-wheel <resolved-wheel>]"
    exit 0
fi

host_glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
use_official_wheel=0
defer_flash_attn=0
case "${FLASH_ATTN_INSTALL_MODE}" in
    auto)
        if [[ "$(printf '%s\n' "2.32" "${host_glibc}" | sort -V | head -n1)" == "2.32" ]]; then
            use_official_wheel=1
        fi
        ;;
    wheel) use_official_wheel=1 ;;
    source) use_official_wheel=0 ;;
    defer) defer_flash_attn=1 ;;
    *)
        echo "FLASH_ATTN_INSTALL_MODE must be auto, wheel, source, or defer." >&2
        exit 2
        ;;
esac

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.12 pip -y
fi

# Install the exact Torch stack first, so wheel resolution reads Torch's CUDA
# runtime and ABI instead of guessing from the host driver's nvidia-smi output.
conda run -n "${ENV_NAME}" python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    torchdata==0.11.0 torchcodec==0.6.0

setup_args=(--env-name "${ENV_NAME}" --resume)
if [[ "${defer_flash_attn}" == "1" ]]; then
    echo "Deferring flash-attn installation to Dockerfile.lingbot_v2."
    setup_args+=(--skip-flash-attn)
elif [[ "${use_official_wheel}" == "1" ]]; then
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
    setup_args+=(--flash-attn-wheel "${wheel_path}")
else
    command -v nvcc >/dev/null || {
        echo "Source flash-attn build requires nvcc (CUDA toolkit), not only an NVIDIA driver." >&2
        exit 2
    }
    echo "Host glibc ${host_glibc}: building flash-attn 2.8.3 from source for Classic compatibility."
fi

bash "${REPO_ROOT}/third_party/lingbot-vla-v2/tools/create_train_env.sh" "${setup_args[@]}"
