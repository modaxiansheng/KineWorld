#!/usr/bin/env bash
# ============================================================
#  Start the KineWorld flow-action inference server.
#
#  Base Wan2.2-TI2V-5B weights are expected under inference/models/Wan-AI
#  (see README "Model download"). Defaults below match the released
#  33-frame (9 video frame, 4x downsample) checkpoint recipe.
#
#  Usage:
#    CHECKPOINT=/path/to/step-XXXX.safetensors bash start_server.sh
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a trained .safetensors}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-$(dirname "${CHECKPOINT}")/action_norm_stats.npz}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-${SCRIPT_DIR}/models}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-full}"   # "full" or "lora"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"

# ---- Must match the training config of the checkpoint ----
ACTION_DIM=14
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
NUM_ACTION_LAYERS="${NUM_ACTION_LAYERS:-30}"
ACTION_PRED_TARGET="${ACTION_PRED_TARGET:-velocity}"
ACTION_POS_MODE="${ACTION_POS_MODE:-rope}"
PROPRIO_MODE="${PROPRIO_MODE:-text}"
COND_LAYER_STRIDE="${COND_LAYER_STRIDE:-1}"
ACTION_SNR_SHIFT="${ACTION_SNR_SHIFT:-5.0}"
TEXT_CONTEXT_DIM=4096
CAMERAS="head_camera left_camera right_camera"
SIZE_W=320
SIZE_H=256

# ---- Decoding ----
VIDEO_INFERENCE_STEPS="${VIDEO_INFERENCE_STEPS:-25}"
SIGMA_SHIFT="${SIGMA_SHIFT:-5.0}"
ACTION_INFERENCE_STEPS="${ACTION_INFERENCE_STEPS:-50}"
# Noise-level label for the action expert's visual condition. 0.0 treats the
# generated latents as clean.
ACTION_COND_SIGMA="${ACTION_COND_SIGMA:-0.0}"
# The server returns the full predicted chunk; the client picks EXECUTE_WINDOW.
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-${NUM_FRAMES}}"

echo "========== KineWorld Flow-Action Inference Server =========="
echo "  checkpoint:        ${CHECKPOINT}   (${CHECKPOINT_MODE})"
echo "  host:port:         ${HOST}:${PORT}   device: ${DEVICE}"
echo "  frames:            action=${NUM_FRAMES}  video=${NUM_VIDEO_FRAMES}  action_layers=${NUM_ACTION_LAYERS}"
echo "  video/action steps:${VIDEO_INFERENCE_STEPS} / ${ACTION_INFERENCE_STEPS}   snr_shift=${ACTION_SNR_SHIFT}"
echo "=========================================================="

"${PYTHON}" "${SCRIPT_DIR}/flow_action_server.py" \
  --checkpoint "${CHECKPOINT}" \
  --action_norm_path "${ACTION_NORM_PATH}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --checkpoint_mode "${CHECKPOINT_MODE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --action_dim "${ACTION_DIM}" \
  --num_frames "${NUM_FRAMES}" \
  --num_video_frames "${NUM_VIDEO_FRAMES}" \
  --num_action_layers "${NUM_ACTION_LAYERS}" \
  --action_pred_target "${ACTION_PRED_TARGET}" \
  --action_pos_mode "${ACTION_POS_MODE}" \
  --proprio_mode "${PROPRIO_MODE}" \
  --cond_layer_stride "${COND_LAYER_STRIDE}" \
  --action_snr_shift "${ACTION_SNR_SHIFT}" \
  --cameras ${CAMERAS} \
  --text_context_dim "${TEXT_CONTEXT_DIM}" \
  --size "${SIZE_W}" "${SIZE_H}" \
  --video_inference_steps "${VIDEO_INFERENCE_STEPS}" \
  --sigma_shift "${SIGMA_SHIFT}" \
  --action_inference_steps "${ACTION_INFERENCE_STEPS}" \
  --action_cond_sigma "${ACTION_COND_SIGMA}" \
  --action_chunk_size "${ACTION_CHUNK_SIZE}"
