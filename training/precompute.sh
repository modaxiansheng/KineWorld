#!/usr/bin/env bash
# =====================================================================
# Optional: pre-encode T5 text, VAE latents, and RAFT flow into a latent
# cache so training can skip the encoders (set LOAD_FROM_CACHE=true in
# train.sh with the same CACHE_ROOT and temporal config).
#
# Single node (all visible GPUs):
#   DATASET_BASE_PATH=/path/to/robotwin_data \
#   CACHE_ROOT=/path/to/cache bash precompute.sh
#
# Sharding is env-driven (RANK / WORLD_SIZE); torchrun fills these in.
# =====================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:?set DATASET_BASE_PATH to the RoboTwin data root}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT to the output cache directory}"

# Base-model files (under training/models/Wan-AI by default).
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/models/Wan-AI/Wan2.2-TI2V-5B}"
VAE_PATH="${VAE_PATH:-${MODELS_DIR}/Wan2.2_VAE.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${MODELS_DIR}/models_t5_umt5-xxl-enc-bf16.pth}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${SCRIPT_DIR}/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"

# Temporal config MUST match train.sh.
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
VISUAL_STRIDE="${VISUAL_STRIDE:-4}"
FLOW_MODE="${FLOW_MODE:-robot_only}"
FLOW_MAX_MAGNITUDE="${FLOW_MAX_MAGNITUDE:-25.0}"
VARIANTS="${VARIANTS:-aloha-agilex_clean_50 aloha-agilex_randomized_500}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

"${PYTHON}" -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/precompute_latents.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --cache_root "${CACHE_ROOT}" \
  --vae_path "${VAE_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --tokenizer_dir "${TOKENIZER_DIR}" \
  --variants ${VARIANTS} \
  --num_frames ${NUM_FRAMES} \
  --num_video_frames ${NUM_VIDEO_FRAMES} \
  --visual_stride ${VISUAL_STRIDE} \
  --flow_mode ${FLOW_MODE} \
  --flow_max_magnitude ${FLOW_MAX_MAGNITUDE}
