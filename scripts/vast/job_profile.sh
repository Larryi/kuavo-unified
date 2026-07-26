#!/usr/bin/env bash
# Sourced by launch.sh and run_backend.sh. It must not print secrets.

apply_vast_job_profile() {
  : "${TRAINING_TASK:?Set TRAINING_TASK to task1, task2, or task3}"
  case "${TRAINING_TASK}:${MODEL_BACKEND}" in
    task1:openpi)
      export ROBOT_TASK="${ROBOT_TASK:-task1}"
      ;;
    task1:lingbot-v1)
      ;;
    task2:dp)
      export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-dp_r2_h100.yaml}"
      export TASK_NAME="${TASK_NAME:-r2}"
      export METHOD_NAME="${METHOD_NAME:-dp_vast}"
      ;;
    task2:lingbot-v2)
      export TASK_NAME="${TASK_NAME:-r2}"
      export METHOD_NAME="${METHOD_NAME:-lingbot_v2_bimanual_vast}"
      export LINGBOT_V2_TRAIN_CONFIG="${LINGBOT_V2_TRAIN_CONFIG:-configs/policy/lingbot_v2/kuavo_lora_task2_bimanual.yaml}"
      export LINGBOT_V2_NORM_CONFIG="${LINGBOT_V2_NORM_CONFIG:-configs/policy/lingbot_v2/kuavo_norm_task2_bimanual.yaml}"
      export LINGBOT_V2_DATA_NAME="${LINGBOT_V2_DATA_NAME:-kuavo_v2_bimanual}"
      ;;
    task3:act)
      export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-act_config.yaml}"
      export TASK_NAME="${TASK_NAME:-task3}"
      export METHOD_NAME="${METHOD_NAME:-act_vast}"
      ;;
    *)
      echo "Unsupported VastAI task/backend pair: ${TRAINING_TASK} + ${MODEL_BACKEND}" >&2
      return 2
      ;;
  esac
}
