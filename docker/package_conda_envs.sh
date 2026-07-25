#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
OUTPUT_ROOT="${DOCKER_ENV_ROOT:-/mnt/pqssd/docker_envs}"
PACK_TMPDIR="${CONDA_PACK_TMPDIR:-${OUTPUT_ROOT}/.tmp}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    echo "Usage: $0 [classic|lingbot-v1|lingbot-v2|all]" >&2
}

pack_env() {
    local env_name="$1"
    local target_dir="$2"
    local archive="${target_dir}/myenv.tar.gz"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'mkdir -p %q\n' "${PACK_TMPDIR}"
        printf 'mkdir -p %q\n' "${target_dir}"
        printf 'TMPDIR=%q conda-pack -n %q --ignore-editable-packages -o %q\n' \
            "${PACK_TMPDIR}" "${env_name}" "${archive}"
        return
    fi
    if ! conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
        echo "Conda environment does not exist: ${env_name}" >&2
        exit 2
    fi
    mkdir -p "${PACK_TMPDIR}" "${target_dir}"
    if [[ -e "${archive}" ]]; then
        echo "Refusing to overwrite existing archive: ${archive}" >&2
        exit 3
    fi
    TMPDIR="${PACK_TMPDIR}" conda-pack \
        -n "${env_name}" \
        --ignore-editable-packages \
        -o "${archive}"
    sha256sum "${archive}" >"${archive}.sha256"
    echo "Packed ${env_name}: ${archive}"
}

verify_lingbot_v1() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "conda run -n kdc_vla python -c <verify flash_attn import and Torch runtime>"
        return
    fi
    conda run -n kdc_vla python -c \
        'import flash_attn,torch; print({"flash_attn": flash_attn.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)})'
}

case "${TARGET}" in
    classic)
        pack_env kdc_dev "${OUTPUT_ROOT}/classic"
        ;;
    lingbot-v1)
        verify_lingbot_v1
        pack_env kdc_vla "${OUTPUT_ROOT}/lingbot-v1"
        ;;
    lingbot-v2)
        if ! conda env list | awk '{print $1}' | grep -Fxq lingbotvla_v2; then
            echo "Create lingbotvla_v2 first; see docker/readme.md." >&2
            exit 2
        fi
        conda run -n lingbotvla_v2 python "${REPO_ROOT}/docker/flash_attn_wheel.py" --dry-run
        pack_env lingbotvla_v2 "${OUTPUT_ROOT}/lingbot-v2"
        ;;
    all)
        pack_env kdc_dev "${OUTPUT_ROOT}/classic"
        verify_lingbot_v1
        pack_env kdc_vla "${OUTPUT_ROOT}/lingbot-v1"
        if conda env list | awk '{print $1}' | grep -Fxq lingbotvla_v2; then
            conda run -n lingbotvla_v2 python "${REPO_ROOT}/docker/flash_attn_wheel.py" --dry-run
            pack_env lingbotvla_v2 "${OUTPUT_ROOT}/lingbot-v2"
        else
            echo "Skipping missing lingbotvla_v2; create it as documented." >&2
        fi
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
