#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:?usage: $0 DATASET_ROOT [OUTPUT_JSON]}"
OUTPUT_JSON="${2:-assets/norm_stats/kuavo_v2_right_arm_meanstd.json}"
V2_ROOT="${LINGBOT_V2_ROOT:-/home/larry/lingbot-vla-v2}"
CONFIG="${KUAVO_V2_CONFIG:-configs/policy/lingbot_v2/kuavo_norm.yaml}"
DATA_NAME="${KUAVO_V2_DATA_NAME:-kuavo_v2_right_arm}"
NUM_WORKERS="${KUAVO_V2_NORM_NUM_WORKERS:-2}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"
export PYTHONPATH="${PWD}:${V2_ROOT}:${PWD}/third_party/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${PWD}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python kuavo_train/lingbot_v2/compute_norm_stats.py "${CONFIG}" \
  --data.data_name "${DATA_NAME}" \
  --data.train_path "${DATASET_ROOT}" \
  --data.robot_config_root configs/robot_configs \
  --data.num_workers "${NUM_WORKERS}" \
  --data.norm_path "${OUTPUT_JSON}" \
  --data.data_ratio_for_norm_compute 1 \
  --data.norm_merge_chunk_dim true
