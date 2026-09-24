#!/usr/bin/env bash
# Train KineWorld on RoboTwin 2.0 Aloha-AgileX demonstrations.
#
# This launcher is intentionally environment-driven so the same command can be
# used by a one-device canary and by the HCU multi-node wrappers in ../track1.
# It defaults to the head-only WorldArena distribution and requires an official
# compatible warm-start checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

# ---- Distributed config ----
NUM_MACHINES="${NUM_MACHINES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MACHINE_RANK="${MACHINE_RANK:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine_rank) MACHINE_RANK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L | wc -l)"
  else
    echo "NUM_GPUS is required when nvidia-smi is unavailable (for example on HCU)" >&2
    exit 2
  fi
fi
require_positive_integer NUM_MACHINES "$NUM_MACHINES"
require_positive_integer NUM_GPUS "$NUM_GPUS"
require_positive_integer MASTER_PORT "$MASTER_PORT"
if [[ ! "$MACHINE_RANK" =~ ^[0-9]+$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "MACHINE_RANK must be in [0, NUM_MACHINES), got $MACHINE_RANK/$NUM_MACHINES" >&2
  exit 2
fi

DEVICE_LIST="${DEVICE_LIST:-$(seq -s ',' 0 $((NUM_GPUS - 1)))}"
IFS=',' read -r -a _visible_devices <<<"$DEVICE_LIST"
if (( ${#_visible_devices[@]} != NUM_GPUS )); then
  echo "DEVICE_LIST count (${#_visible_devices[@]}) must equal NUM_GPUS ($NUM_GPUS)" >&2
  exit 2
fi
# HCU's PyTorch runtime consults all three variables. Keep them identical.
export HCU_VISIBLE_DEVICES="$DEVICE_LIST"
export HIP_VISIBLE_DEVICES="$DEVICE_LIST"
export CUDA_VISIBLE_DEVICES="$DEVICE_LIST"

# ---- Data contract ----
DATASET_BASE_PATH="${DATASET_BASE_PATH:?set DATASET_BASE_PATH to the RoboTwin training root}"
if [[ ! -d "$DATASET_BASE_PATH" ]]; then
  echo "RoboTwin training root does not exist: $DATASET_BASE_PATH" >&2
  exit 3
fi
_dataset_lower="$(printf '%s' "$DATASET_BASE_PATH" | tr '[:upper:]' '[:lower:]')"
case "$_dataset_lower" in
  *dataset_track1*|*current_track1*|*evaluation_inputs*|*/track1_data*|*/track1/test*)
    echo "refusing to train from a path that looks like the official Track 1 test set: $DATASET_BASE_PATH" >&2
    exit 3
    ;;
esac
if [[ -d "$DATASET_BASE_PATH/data/fixed_scene_task" || \
      -d "$DATASET_BASE_PATH/data/random_scene_task" || \
      -d "$DATASET_BASE_PATH/first_frame/fixed_scene_task" || \
      -d "$DATASET_BASE_PATH/first_frame/random_scene_task" || \
      -d "$DATASET_BASE_PATH/instructions/fixed_scene_task" || \
      -d "$DATASET_BASE_PATH/instructions/random_scene_task" ]]; then
  echo "refusing official Track 1 test layout as training data: $DATASET_BASE_PATH" >&2
  exit 3
fi

VARIANTS="${VARIANTS:-aloha-agilex_clean_50}"
CAMERAS="${CAMERAS:-head_camera}"
# Empty is deliberate: the upstream CAMERA_PREFIX describes a T-shape
# head+wrist composite and is out-of-distribution for Track 1 head-only video.
CAMERA_PREFIX="${CAMERA_PREFIX:-}"
TRACK1_PROMPT_TEMPLATE="${TRACK1_PROMPT_TEMPLATE:-true}"
read -r -a VARIANT_ARGS <<<"$VARIANTS"
read -r -a CAMERA_ARGS <<<"$CAMERAS"
if (( ${#VARIANT_ARGS[@]} == 0 || ${#CAMERA_ARGS[@]} == 0 )); then
  echo "VARIANTS and CAMERAS must each contain at least one value" >&2
  exit 3
fi
if (( ${#VARIANT_ARGS[@]} != 1 )) || [[ "${VARIANT_ARGS[0]}" != "aloha-agilex_clean_50" ]]; then
  echo "the audited Track1 server bundle contains only aloha-agilex_clean_50" >&2
  exit 3
fi
TRAINING_MANIFEST="${TRAINING_MANIFEST:?set TRAINING_MANIFEST to the training JSONL manifest}"
TRAINING_MANIFEST_SHA256="${TRAINING_MANIFEST_SHA256:?set TRAINING_MANIFEST_SHA256 to its SHA-256 digest}"
if [[ ! "$TRAINING_MANIFEST_SHA256" =~ ^[a-fA-F0-9]{64}$ ]]; then
  echo "TRAINING_MANIFEST_SHA256 must be a 64-character SHA-256 digest" >&2
  exit 3
fi
if [[ ! -f "$TRAINING_MANIFEST" ]]; then
  echo "training manifest is missing: $TRAINING_MANIFEST" >&2
  exit 3
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required to validate the training manifest" >&2
  exit 3
fi
_actual_manifest_sha256="$(sha256sum "$TRAINING_MANIFEST" | awk '{print $1}')"
if [[ "$_actual_manifest_sha256" != "$TRAINING_MANIFEST_SHA256" ]]; then
  echo "training manifest SHA-256 mismatch: $_actual_manifest_sha256 != $TRAINING_MANIFEST_SHA256" >&2
  exit 3
fi
if (( ${#CAMERA_ARGS[@]} == 1 )) && [[ "${CAMERA_ARGS[0]}" == "head_camera" ]]; then
  _prefix_lower="$(printf '%s' "$CAMERA_PREFIX" | tr '[:upper:]' '[:lower:]')"
  if [[ "$_prefix_lower" == *"bottom-left"* || "$_prefix_lower" == *"bottom-right"* || \
        "$_prefix_lower" == *"top half"* ]]; then
    echo "refusing a T-shape multi-camera CAMERA_PREFIX for head-only training" >&2
    exit 3
  fi
fi
if is_true "$TRACK1_PROMPT_TEMPLATE" && [[ -n "$CAMERA_PREFIX" ]]; then
  echo "TRACK1_PROMPT_TEMPLATE=true requires an empty CAMERA_PREFIX" >&2
  echo "the Track1 semantic template is applied separately and idempotently" >&2
  exit 3
fi

# ---- Base models and official warm-start ----
MODEL_PATHS_DIT="${MODEL_PATHS_DIT:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors}"
MODEL_PATHS_T5="${MODEL_PATHS_T5:-Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth}"
MODEL_PATHS_VAE="${MODEL_PATHS_VAE:-Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}"
MODEL_PATHS_JSON="${MODEL_PATHS_JSON:-}"
TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to a compatible warm-start .safetensors file}"
RESUME_CHECKPOINT_SHA256="${RESUME_CHECKPOINT_SHA256:?set RESUME_CHECKPOINT_SHA256 to its SHA-256 digest}"
ALLOW_CUSTOM_WARM_START="${ALLOW_CUSTOM_WARM_START:-false}"
if [[ ! "$RESUME_CHECKPOINT_SHA256" =~ ^[a-fA-F0-9]{64}$ ]]; then
  echo "RESUME_CHECKPOINT_SHA256 must be a 64-character SHA-256 digest" >&2
  exit 4
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "required warm-start checkpoint is missing: $RESUME_CHECKPOINT" >&2
  exit 4
fi
_resume_checkpoint_sha256_lower="$(printf '%s' "$RESUME_CHECKPOINT_SHA256" | tr '[:upper:]' '[:lower:]')"
case "$_resume_checkpoint_sha256_lower" in
  21940bd45e95dc39bc3b166c6ecd2a777efa01c8068124958778173cd01ddf66)
    KINEWORLD_WARM_START_PROFILE="robotwin_pretrained"
    _expected_checkpoint_bytes=13249920488
    echo "WARNING: this warm-start was trained with a T-shape multi-camera distribution;" >&2
    echo "         this run explicitly re-adapts it to Track1 head-only prompts/camera data." >&2
    ;;
  e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4)
    KINEWORLD_WARM_START_PROFILE="worldarena_stage1"
    _expected_checkpoint_bytes=10137267208
    echo "WARNING: Stage1 is an explicit A/B alternative and has no pretrained action expert;" >&2
    echo "         its action head starts from random initialization." >&2
    ;;
  *)
    if ! is_true "$ALLOW_CUSTOM_WARM_START"; then
      echo "warm-start must match a recognized compatible checkpoint digest" >&2
      echo "set ALLOW_CUSTOM_WARM_START=true only for an intentionally verified derivative" >&2
      exit 4
    fi
    KINEWORLD_WARM_START_PROFILE="custom"
    _expected_checkpoint_bytes=0
    ;;
esac
export KINEWORLD_WARM_START_PROFILE
_checkpoint_bytes="$(stat -c %s "$RESUME_CHECKPOINT")"
if (( _expected_checkpoint_bytes > 0 && _checkpoint_bytes != _expected_checkpoint_bytes )); then
  echo "checkpoint byte-size mismatch: $_checkpoint_bytes != $_expected_checkpoint_bytes" >&2
  exit 4
fi
_actual_checkpoint_sha256="$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')"
if [[ "$_actual_checkpoint_sha256" != "$_resume_checkpoint_sha256_lower" ]]; then
  echo "checkpoint SHA-256 mismatch: $_actual_checkpoint_sha256 != $_resume_checkpoint_sha256_lower" >&2
  exit 4
fi

# ---- Temporal and spatial config ----
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
VISUAL_STRIDE="${VISUAL_STRIDE:-4}"
SIZE_W="${SIZE_W:-320}"
SIZE_H="${SIZE_H:-240}"

# ---- Flow ----
FLOW_METHOD="${FLOW_METHOD:-raft}"
FLOW_DEVICE="${FLOW_DEVICE:-cuda}"
FLOW_MODE="${FLOW_MODE:-robot_only}"
VIDEO_OBJECTIVE="${VIDEO_OBJECTIVE:-track1_conditional_rgb}"
FLOW_MAX_MAGNITUDE="${FLOW_MAX_MAGNITUDE:-25.0}"
FLOW_MOTION_BOOST="${FLOW_MOTION_BOOST:-2.0}"

# ---- Optimization ----
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-1.0}"
if [[ -z "${FLOW_LOSS_WEIGHT+x}" ]]; then
  if [[ "$VIDEO_OBJECTIVE" == "track1_conditional_rgb" ]]; then
    FLOW_LOSS_WEIGHT=0.0
  else
    FLOW_LOSS_WEIGHT=0.1
  fi
fi
case "$VIDEO_OBJECTIVE" in
  track1_conditional_rgb)
    if ! awk -v value="$FLOW_LOSS_WEIGHT" 'BEGIN { exit !(value == 0) }'; then
      echo "track1_conditional_rgb requires FLOW_LOSS_WEIGHT=0 (flow is clean conditioning only)" >&2
      exit 2
    fi
    ;;
  joint_dual_stream) ;;
  *)
    echo "VIDEO_OBJECTIVE must be track1_conditional_rgb or joint_dual_stream" >&2
    exit 2
    ;;
esac
ACTION_DIM="${ACTION_DIM:-14}"
NUM_ACTION_LAYERS="${NUM_ACTION_LAYERS:-30}"
ACTION_SNR_SHIFT="${ACTION_SNR_SHIFT:-5.0}"
TEXT_CONTEXT_DIM="${TEXT_CONTEXT_DIM:-4096}"
LOSS_TIMESTEP_WEIGHTING="${LOSS_TIMESTEP_WEIGHTING:-on}"
REF_AUG_STRENGTH="${REF_AUG_STRENGTH:-0.1}"
FP32_MODULATION="${FP32_MODULATION:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
DATA_SHUFFLE_SEED="${DATA_SHUFFLE_SEED:-42}"
if [[ ! "$DATA_SHUFFLE_SEED" =~ ^[0-9]+$ ]]; then
  echo "DATA_SHUFFLE_SEED must be a non-negative integer" >&2
  exit 2
fi

# ---- IDM conditioning ----
COND_NOISE_PROB="${COND_NOISE_PROB:-0.5}"
COND_DETACH="${COND_DETACH:-false}"
COND_LAYER_STRIDE="${COND_LAYER_STRIDE:-1}"
ACTION_PRED_TARGET="${ACTION_PRED_TARGET:-velocity}"
ACTION_POS_MODE="${ACTION_POS_MODE:-rope}"
PROPRIO_MODE="${PROPRIO_MODE:-text}"

# ---- LR schedule, checkpointing, and exact resume ----
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw}"
case "$OPTIMIZER_TYPE" in
  adamw|zro_adamw|hybrid_adamw) ;;
  *) echo "OPTIMIZER_TYPE must be adamw, zro_adamw, or hybrid_adamw" >&2; exit 2 ;;
esac
HYBRID_OPTIMIZER_REPLICATED_FRACTION="${HYBRID_OPTIMIZER_REPLICATED_FRACTION:-0.70}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
LR_MAX_STEPS="${LR_MAX_STEPS:-0}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-100}"
FULL_STATE_KEEP="${FULL_STATE_KEEP:-2}"
RESUME_STATE_DIR="${RESUME_STATE_DIR:-}"
if [[ "$OPTIMIZER_TYPE" != adamw && "$FULL_STATE_KEEP" != 0 ]]; then
  echo "OPTIMIZER_TYPE=$OPTIMIZER_TYPE requires FULL_STATE_KEEP=0" >&2
  exit 2
fi
if [[ "$OPTIMIZER_TYPE" != adamw && -n "$RESUME_STATE_DIR" ]]; then
  echo "OPTIMIZER_TYPE=$OPTIMIZER_TYPE resumes from safetensors weights, not RESUME_STATE_DIR" >&2
  exit 2
fi
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/models/train/kineworld_idm}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-${OUTPUT_PATH}/action_norm_stats.npz}"

# ---- Per-process VRAM fail-closed gate ----
VRAM_PROBE_STEP="${VRAM_PROBE_STEP:-3}"
MIN_ALLOCATED_GIB="${MIN_ALLOCATED_GIB:-54}"
MIN_VRAM_GIB="${MIN_VRAM_GIB:-54}"
MAX_VRAM_GIB="${MAX_VRAM_GIB:-63.0}"
STOP_AFTER_VRAM_PROBE="${STOP_AFTER_VRAM_PROBE:-false}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-0}"
require_positive_integer VRAM_PROBE_STEP "$VRAM_PROBE_STEP"
if [[ ! "$MIN_ALLOCATED_GIB" =~ ^[0-9]+([.][0-9]+)?$ || \
      ! "$MIN_VRAM_GIB" =~ ^[0-9]+([.][0-9]+)?$ || \
      ! "$MAX_VRAM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "MIN_ALLOCATED_GIB, MIN_VRAM_GIB, and MAX_VRAM_GIB must be non-negative decimal numbers" >&2
  exit 2
fi
if ! awk -v allocated="$MIN_ALLOCATED_GIB" -v reserved="$MIN_VRAM_GIB" -v max="$MAX_VRAM_GIB" \
    'BEGIN { exit !((allocated >= 54) && (reserved >= 54) && (max >= reserved) && (max <= 63)) }'; then
  echo "training requires allocated/reserved floors >=54 GiB and MIN_VRAM_GIB<=MAX_VRAM_GIB<=63" >&2
  exit 2
fi
if [[ ! "$MAX_OPTIMIZER_STEPS" =~ ^[0-9]+$ ]]; then
  echo "MAX_OPTIMIZER_STEPS must be a non-negative integer" >&2
  exit 2
fi

# ---- Online versus cached data ----
LOAD_FROM_CACHE="${LOAD_FROM_CACHE:-false}"
CACHE_ROOT="${CACHE_ROOT:-}"
CACHE_CHUNKS_PER_EPISODE="${CACHE_CHUNKS_PER_EPISODE:-4}"
CACHE_EPISODE_LRU_SIZE="${CACHE_EPISODE_LRU_SIZE:-4}"
if is_true "$LOAD_FROM_CACHE" && ! is_true "$TRACK1_PROMPT_TEMPLATE"; then
  if [[ -z "$CACHE_ROOT" || ! -d "$CACHE_ROOT" ]]; then
    echo "LOAD_FROM_CACHE=true requires an existing CACHE_ROOT" >&2
    exit 3
  fi
  MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_PATHS_DIT},${MODEL_PATHS_VAE}"
else
  if is_true "$LOAD_FROM_CACHE" && [[ -z "$CACHE_ROOT" || ! -d "$CACHE_ROOT" ]]; then
    echo "LOAD_FROM_CACHE=true requires an existing CACHE_ROOT" >&2
    exit 3
  fi
  MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_PATHS_DIT},${MODEL_PATHS_T5},${MODEL_PATHS_VAE}"
fi

mkdir -p "$OUTPUT_PATH"
cd "$SCRIPT_DIR"

ACCELERATE_ARGS=(
  --num_processes "$((NUM_GPUS * NUM_MACHINES))"
  --num_machines "$NUM_MACHINES"
)
if (( NUM_MACHINES > 1 )); then
  ACCELERATE_ARGS+=(
    --machine_rank "$MACHINE_RANK"
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
  )
fi

MODEL_ARGS=()
if [[ -n "$MODEL_PATHS_JSON" ]]; then
  MODEL_ARGS+=(--model_paths "$MODEL_PATHS_JSON")
else
  MODEL_ARGS+=(--model_id_with_origin_paths "$MODEL_ID_WITH_ORIGIN_PATHS")
fi

TRAIN_ARGS=(
  --dataset_base_path "$DATASET_BASE_PATH"
  --num_frames "$NUM_FRAMES"
  --num_video_frames "$NUM_VIDEO_FRAMES"
  --visual_stride "$VISUAL_STRIDE"
  --tokenizer_model_id "$TOKENIZER_MODEL_ID"
  "${MODEL_ARGS[@]}"
  --learning_rate "$LEARNING_RATE"
  --num_epochs "$NUM_EPOCHS"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "$OUTPUT_PATH"
  --size "$SIZE_W" "$SIZE_H"
  --variants "${VARIANT_ARGS[@]}"
  --training_manifest "$TRAINING_MANIFEST"
  --training_manifest_sha256 "$TRAINING_MANIFEST_SHA256"
  --cameras "${CAMERA_ARGS[@]}"
  --camera_prefix "$CAMERA_PREFIX"
  --trainable_models "$TRAINABLE_MODELS"
  --flow_method "$FLOW_METHOD"
  --flow_device "$FLOW_DEVICE"
  --flow_mode "$FLOW_MODE"
  --video_objective "$VIDEO_OBJECTIVE"
  --flow_max_magnitude "$FLOW_MAX_MAGNITUDE"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --data_shuffle_seed "$DATA_SHUFFLE_SEED"
  --flow_loss_weight "$FLOW_LOSS_WEIGHT"
  --action_loss_weight "$ACTION_LOSS_WEIGHT"
  --action_dim "$ACTION_DIM"
  --num_action_layers "$NUM_ACTION_LAYERS"
  --action_norm_path "$ACTION_NORM_PATH"
  --action_snr_shift "$ACTION_SNR_SHIFT"
  --cond_noise_prob "$COND_NOISE_PROB"
  --cond_layer_stride "$COND_LAYER_STRIDE"
  --action_pred_target "$ACTION_PRED_TARGET"
  --action_pos_mode "$ACTION_POS_MODE"
  --proprio_mode "$PROPRIO_MODE"
  --loss_timestep_weighting "$LOSS_TIMESTEP_WEIGHTING"
  --lr_scheduler_type "$LR_SCHEDULER_TYPE"
  --optimizer_type "$OPTIMIZER_TYPE"
  --hybrid_optimizer_replicated_fraction "$HYBRID_OPTIMIZER_REPLICATED_FRACTION"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --lr_max_steps "$LR_MAX_STEPS"
  --full_state_keep "$FULL_STATE_KEEP"
  --flow_motion_boost "$FLOW_MOTION_BOOST"
  --text_context_dim "$TEXT_CONTEXT_DIM"
  --batch_size "$BATCH_SIZE"
  --extra_inputs "input_image"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --ref_aug_strength "$REF_AUG_STRENGTH"
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
  --save_steps "$SAVE_STEPS"
  --resume_checkpoint "$RESUME_CHECKPOINT"
  --vram_probe_step "$VRAM_PROBE_STEP"
  --min_allocated_gib "$MIN_ALLOCATED_GIB"
  --min_vram_gib "$MIN_VRAM_GIB"
  --max_vram_gib "$MAX_VRAM_GIB"
  --max_optimizer_steps "$MAX_OPTIMIZER_STEPS"
)
if [[ -n "$RESUME_STATE_DIR" ]]; then
  TRAIN_ARGS+=(--resume_state_dir "$RESUME_STATE_DIR")
fi
if is_true "$COND_DETACH"; then
  TRAIN_ARGS+=(--cond_detach)
fi
if is_true "$FP32_MODULATION"; then
  TRAIN_ARGS+=(--fp32_modulation)
fi
if is_true "$USE_GRADIENT_CHECKPOINTING"; then
  TRAIN_ARGS+=(--use_gradient_checkpointing)
fi
if is_true "$TRACK1_PROMPT_TEMPLATE"; then
  TRAIN_ARGS+=(--track1_prompt_template)
else
  TRAIN_ARGS+=(--no_track1_prompt_template)
fi
if is_true "$STOP_AFTER_VRAM_PROBE"; then
  TRAIN_ARGS+=(--stop_after_vram_probe)
fi
if is_true "$LOAD_FROM_CACHE"; then
  TRAIN_ARGS+=(
    --load_from_cache
    --cache_root "$CACHE_ROOT"
    --cache_episode_batching
    --cache_chunks_per_episode "$CACHE_CHUNKS_PER_EPISODE"
    --cache_episode_lru_size "$CACHE_EPISODE_LRU_SIZE"
  )
fi

echo "========== KineWorld Aloha-AgileX Training =========="
echo "  machines/devices: ${NUM_MACHINES} x ${NUM_GPUS} (${DEVICE_LIST})"
echo "  dataset:          ${DATASET_BASE_PATH}"
echo "  variants:         ${VARIANTS}"
echo "  manifest:         ${TRAINING_MANIFEST} (${TRAINING_MANIFEST_SHA256})"
echo "  cameras/prefix:   ${CAMERAS} / $([[ -n "$CAMERA_PREFIX" ]] && echo custom || echo '<plain>')"
echo "  Track1 template:  ${TRACK1_PROMPT_TEMPLATE}"
echo "  size/frames:      ${SIZE_W}x${SIZE_H}; action=${NUM_FRAMES} video=${NUM_VIDEO_FRAMES} stride=${VISUAL_STRIDE}"
echo "  batch/lr:         ${BATCH_SIZE} / ${LEARNING_RATE}"
echo "  warm-start:       ${RESUME_CHECKPOINT}"
echo "  VRAM gate:        optimizer_step=${VRAM_PROBE_STEP}, current_allocated>=${MIN_ALLOCATED_GIB} GiB, current_reserved>=${MIN_VRAM_GIB} GiB, max_reserved<=${MAX_VRAM_GIB} GiB (hard ceiling)"
echo "  data mode:        $(is_true "$LOAD_FROM_CACHE" && echo "cache: ${CACHE_ROOT}" || echo "online ${FLOW_METHOD}/${FLOW_MODE}")"
echo "  video objective:  ${VIDEO_OBJECTIVE} (flow supervised=$([[ "$VIDEO_OBJECTIVE" == joint_dual_stream ]] && echo yes || echo no))"
echo "  output:           ${OUTPUT_PATH}"
echo "===================================================="

exec "$PYTHON" -m accelerate.commands.launch \
  "${ACCELERATE_ARGS[@]}" \
  "${SCRIPT_DIR}/flow_action_train.py" \
  "${TRAIN_ARGS[@]}"
