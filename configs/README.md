# Public configuration examples

These templates show the inputs accepted by the public source subset. They are
not the full V4 training recipe and do not reproduce V4 manuscript results.
They contain no dataset, model weight, server path, or credential.

- `train_public.example.env`: enables the opt-in public subset profile and
  continued fine-tuning from the released step-500. Set local paths and the
  generated manifest hash, then use `training/train.sh --dry-run` before a GPU run.
- `track1_inference.example.sh`: run the public Track 1 inference entry point
  with a compatible checkpoint and precomputed action flow.
- `robotwin_render.example.yml`: minimal legacy RoboTwin render configuration;
  set the actual absolute scene-data root before copying it into RoboTwin.

Complete instructions are in [中文使用指南](../README_zh-CN.md).
Checkpoint metadata is published in `pumpkin601/KineWorld` as
`training_config_public.json`; it does not certify manuscript reproduction.
