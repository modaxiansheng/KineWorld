#!/usr/bin/env bash
# ============================================================
#  Run KineWorld evaluation on all 50 RoboTwin tasks, optionally
#  sharded across GPUs (one inference server per GPU).
#
#  Prerequisite — start one server per GPU first (server env):
#    1 GPU:   CHECKPOINT=... bash inference/start_server.sh
#    N GPUs:  for g in $(seq 0 $((N-1))); do \
#               CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) CHECKPOINT=... \
#               bash inference/start_server.sh & done
#
#  Then, in the RoboTwin env:
#    ROBOTWIN_ROOT=/path/to/RoboTwin bash eval_all.sh [task_config] [seed]
#
#  Env:
#    NUM_GPUS   number of GPUs / servers (default 1)
#    PORT_BASE  first server port (default 8000); GPU g uses PORT_BASE + g
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_CONFIG="${1:-demo_clean}"
SEED="${2:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
PORT_BASE="${PORT_BASE:-8000}"

: "${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT to your RoboTwin checkout}"

# The 50 standard RoboTwin tasks (matches ROBOTWIN_ALL_TASKS in
# training/dataset_action_robotwin.py).
TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
  handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
  move_playingcard_away move_stapler_pad open_laptop open_microwave
  pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
  place_bread_basket place_bread_skillet place_burger_fries place_can_basket
  place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
  place_fan place_mouse_pad place_object_basket place_object_scale
  place_object_stand place_phone_stand place_shoe press_stapler
  put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
  shake_bottle_horizontally stack_blocks_three stack_blocks_two
  stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

echo "Evaluating ${#TASKS[@]} tasks | config=${TASK_CONFIG} seed=${SEED} gpus=${NUM_GPUS}"

for i in "${!TASKS[@]}"; do
  gpu=$(( i % NUM_GPUS ))
  port=$(( PORT_BASE + gpu ))
  SERVER_PORT="${port}" bash "${SCRIPT_DIR}/eval.sh" \
    "${TASKS[$i]}" "${TASK_CONFIG}" "${SEED}" "${gpu}" &
  # Keep at most NUM_GPUS tasks in flight (one per GPU / server).
  while (( $(jobs -rp | wc -l) >= NUM_GPUS )); do wait -n; done
done
wait

echo "Done. Results: ${ROBOTWIN_ROOT}/eval_result/<task>/kineworld/${TASK_CONFIG}/"
