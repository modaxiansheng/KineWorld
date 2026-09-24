# Public configuration examples

These templates show the inputs accepted by the public source subset. They are
not the full V4 training recipe and do not reproduce V4 manuscript results.
They contain no dataset, model weight, server path, or credential.

- `train_public.example.env`: set paths and hashes before invoking
  `training/train.sh`. Source it with Bash, then run the training script.
- `track1_inference.example.sh`: run the public Track 1 inference entry point
  with a compatible checkpoint and precomputed action flow.

Checkpoint-specific metadata will be supplied with a validated model release.
