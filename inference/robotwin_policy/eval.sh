#!/usr/bin/env bash
# ============================================================
#  Run RoboTwin evaluation with the KineWorld flow-action policy.
#
#  Prerequisites:
#    1. Start the inference server first (in the server env):
#         cd inference/ && CHECKPOINT=... bash start_server.sh
#    2. Run this script in the RoboTwin conda environment.
#
#  This symlinks the policy package into ${ROBOTWIN_ROOT}/policy/kineworld
#  and calls RoboTwin's stock script/eval_policy.py — no RoboTwin source
#  changes are required.
#
#  Usage:
#    ROBOTWIN_ROOT=/path/to/RoboTwin bash eval.sh <task_name> <task_config> <seed> <gpu_id>
#  Example:
#    ROBOTWIN_ROOT=/path/to/RoboTwin bash eval.sh place_dual_shoes demo_clean 0 0
# ============================================================
set -uo pipefail

POLICY_NAME=kineworld
TASK_NAME=${1:?"arg 1: task_name"}
TASK_CONFIG=${2:?"arg 2: task_config (e.g. demo_clean)"}
SEED=${3:-0}
GPU_ID=${4:-0}

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT to your RoboTwin checkout}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Server connection (must match start_server.sh).
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
# Execute EXECUTE_WINDOW actions per replan; chunk = 1 anchor + EXECUTE_WINDOW.
EXECUTE_WINDOW="${EXECUTE_WINDOW:-25}"
ACTION_CHUNK_SIZE=$((1 + EXECUTE_WINDOW))

# SAPIEN/glvnd EGL compatibility on systems without /etc/glvnd/egl_vendor.d.
if [ -d "/usr/share/glvnd/egl_vendor.d" ]; then
    export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"
fi
# SAPIEN renders headless via Vulkan; point it at the NVIDIA ICD if present.
if [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
# Reach the local inference server directly even when an http(s) proxy is set.
export no_proxy="${SERVER_HOST},127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="${no_proxy}"

# Register the policy package with RoboTwin (additive; no source changes).
POLICY_LINK="${ROBOTWIN_ROOT}/policy/${POLICY_NAME}"
if [ ! -e "${POLICY_LINK}" ]; then
    ln -sfn "${SCRIPT_DIR}" "${POLICY_LINK}"
    echo "Symlinked ${POLICY_LINK} -> ${SCRIPT_DIR}"
fi

cd "${ROBOTWIN_ROOT}"

# Use xvfb-run only if installed; otherwise run directly (SAPIEN renders
# headless via Vulkan/EGL and needs no virtual display).
RUN_PREFIX=""
if command -v xvfb-run >/dev/null 2>&1; then
    RUN_PREFIX="xvfb-run -a"
fi

PYTHONWARNINGS=ignore::UserWarning \
${RUN_PREFIX} "${PYTHON}" script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${POLICY_NAME}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --server_host "${SERVER_HOST}" \
    --server_port "${SERVER_PORT}" \
    --action_chunk_size "${ACTION_CHUNK_SIZE}"

echo ""
echo "========== Evaluation complete: ${TASK_NAME} / ${TASK_CONFIG} =========="
echo "  Results: ${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/${POLICY_NAME}/${TASK_CONFIG}/"
