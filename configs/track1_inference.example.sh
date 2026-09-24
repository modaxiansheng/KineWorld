#!/usr/bin/env bash
# Public inference example. Requires a compatible checkpoint and prepared flow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH}"
PRECOMPUTED_FLOW_ROOT="${PRECOMPUTED_FLOW_ROOT:?set PRECOMPUTED_FLOW_ROOT}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON}" "${REPO_ROOT}/track1/infer_track1.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --conditioning-mode action_flow \
  --action-flow-provider precomputed \
  --precomputed-flow-root "${PRECOMPUTED_FLOW_ROOT}" \
  --device "${DEVICE}"
