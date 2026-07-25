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
    task3:act)
      export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-act_config.yaml}"
      export TASK_NAME="${TASK_NAME:-task3}"
      export METHOD_NAME="${METHOD_NAME:-act_vast}"
      ;;
    *:lingbot-v2)
      echo "LingBot-v2 cloud training integration is paused." >&2
      return 2
      ;;
    *)
      echo "Unsupported VastAI task/backend pair: ${TRAINING_TASK} + ${MODEL_BACKEND}" >&2
      return 2
      ;;
  esac
}
