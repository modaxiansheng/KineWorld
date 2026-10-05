# Using the Hugging Face checkpoint

This guide uses [`pumpkin601/KineWorld`](https://huggingface.co/pumpkin601/KineWorld), revision `37e8f86c6c3cf45cde743162bf9b6de583cf1b73`, and its `step-500.safetensors` file. Commands use Bash on Linux and run from the KineWorld repository root.

For action-conditioned **video generation**, use `track1/infer_track1.py`. The **action-policy server** is a different interface, discussed separately below. The checkpoint is not a complete Diffusers repository and cannot be loaded through a generic `DiffusionPipeline.from_pretrained` call.

## 1. Install and download

Follow [Environment](../README.md#environment) and [Model download](../README.md#model-download) in the main README. Install a CUDA-compatible PyTorch/torchvision pair and make `ffmpeg` and `ffprobe` available on `PATH`.

After download, the relevant layout is:

```text
KineWorld/
  checkpoints/KineWorld/
    step-500.safetensors
    action_norm_stats.npz
    training_config_public.json
    SHA256SUMS
  models/Wan-AI/
    Wan2.2-TI2V-5B/
      diffusion_pytorch_model-*.safetensors
      models_t5_umt5-xxl-enc-bf16.pth
      Wan2.2_VAE.pth
    Wan2.1-T2V-1.3B/google/
      umt5-xxl/
        ... tokenizer files ...
```

The base weights are still required: the code first constructs the Wan pipeline, then loads KineWorld's fine-tuned DiT and flow-stream weights. The video-only loader deliberately skips `action_expert.*`; this is not a missing-checkpoint error. It does not need the action normalization file.

For offline use, complete all model/tokenizer downloads and prepare the action-flow files before disconnecting. Pass `--model-cache-dir models`, not `models/Wan-AI` or the checkpoint folder. The loader appends each model's repository ID to this root. A partially downloaded sharded model must be repaired before inference.

## 2. Prepare the inputs

The current public video adapter expects the WorldArena2 Track 1 layout:

```text
dataset_track1/
  data/fixed_scene_task/episode1.hdf5       ... episode1000.hdf5
  first_frame/fixed_scene_task/episode1.png ... episode1000.png
  instructions/fixed_scene_task/episode1.json ... episode1000.json
```

Each instruction JSON contains an `instruction` string. Each HDF5 contains `/joint_action/vector` with shape `[N, 14]`. The PNG is the initial observation. Obtain this input collection separately; the released 1,000-video archive contains generated outputs and is **not** a substitute for these inputs. Keep the original instruction text: the video adapter encodes it verbatim and does not add the training dataset's Track 1 prompt prefix. Custom short prompts are therefore not automatically equivalent to the training prompt format.

The adapter checks the complete set of 1,000 input filenames before selecting an episode range. Thus, `--episode-start 1 --episode-end 1` limits generation but does not enable a one-file input collection. Custom datasets need an adapter to this input contract, not just a different checkpoint path.

Prepare transport conditions for the same episodes using robot-only rendering and RAFT. A compatible RoboTwin asset checkout, SAPIEN/Vulkan runtime, and RAFT weights are required:

```bash
python track1/precompute_action_flow.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root /path/to/action_flow \
  --robotwin-assets-root /path/to/RoboTwin/assets \
  --render-width 640 --render-height 480 \
  --target-width 320 --target-height 240 \
  --episode-start 1 --episode-end 1 \
  --flow-device cuda:0
```

`--robotwin-assets-root` must contain `embodiments/aloha-agilex/`, not point to the checkout's parent directory. If you already have action-flow PNGs and chunk manifests produced by this preprocessor for the same inputs, reuse them with `--precomputed-flow-root`. Arbitrary optical-flow videos do not satisfy the per-chunk manifest contract. Do not substitute `zero_flow` to suppress missing-flow errors; that selects a different, text-driven baseline.

## 3. Generate a first video

Run the [one-episode inference command](../README.md#worldarena2-track-1-inference) with the downloaded `step-500.safetensors`. It uses 9 keyframes per autoregressive chunk, visual stride 4, and 25 RGB denoising steps. Flow remains a clean condition. The final video is linearly expanded to the action-defined frame count and exported at 640 x 480 and 24 fps. Playback fps is not generation throughput.

Expected outputs are:

```text
outputs/kineworld-step500-episode1/
  videos/episode1.mp4
  per_episode/episode1.json
  run_config.json
  status/
```

Read the logged `Checkpoint load report` and the per-episode record. This release has 825 DiT tensors, 3 flow-conditioning tensors, and 1,191 action-expert tensors. It does not contain `flow_head.*` weights. Those missing head keys are expected for the clean-flow video path, which never predicts flow; they are not evidence of a trained flow-generation head. Investigate other missing/unexpected weights rather than treating a partially loaded model as a successful run. The recorded checkpoint SHA-256 should be:

```text
86294739c54073c836a0dcb3f9114c6cf2bf83d1a8698423b71698e5f88460a3
```

To generate the full collection, precompute flow for `--episode-start 1 --episode-end 1000`, then use that same range in inference and a separate output directory such as `outputs/kineworld-step500-full`. Do not reuse output directories across checkpoint or sampler changes. Existing outputs with different provenance are rejected unless `--overwrite` is explicitly selected.

## 4. Check the output

```bash
python track1/validate_track1.py outputs \
  --dataset-root /path/to/dataset_track1 \
  --videos-dir outputs/kineworld-step500-episode1/videos \
  --records-dir outputs/kineworld-step500-episode1/per_episode \
  --run-config outputs/kineworld-step500-episode1/run_config.json \
  --episode-start 1 --episode-end 1 \
  --output-json outputs/kineworld-step500-episode1/validation_report.json
```

This checks structural properties such as decoding, frame count, resolution, fps, initial-frame preservation, and action-conditioning records. It is not a benchmark-quality score. Also watch the MP4 before scaling up generation.

## 5. Optional action-policy server

Only use this interface if you need predicted robot actions through the RoboTwin client. It is not necessary for Track 1 video generation.

The [released training configuration](https://huggingface.co/pumpkin601/KineWorld/blob/37e8f86c6c3cf45cde743162bf9b6de583cf1b73/training_config_public.json) records `head_camera`, 14-dimensional actions, 33 action frames, 9 video frames, 30 action-expert layers, `cond_layer_stride=2`, velocity prediction, RoPE, and text-mode proprioception. The generic shell wrapper does not select all of these settings.

Unlike the video adapter, the policy loader redirects the text encoder to the Wan2.1 directory and otherwise defaults to ModelScope for missing files. In addition to the main README's downloads, pre-download its text encoder from Hugging Face:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir models/Wan-AI/Wan2.1-T2V-1.3B
```

Then pass the checkpoint-specific settings explicitly:

```bash
python inference/flow_action_server.py \
  --checkpoint checkpoints/KineWorld/step-500.safetensors \
  --checkpoint_mode full \
  --action_norm_path checkpoints/KineWorld/action_norm_stats.npz \
  --local_model_path models \
  --host 127.0.0.1 --port 8000 --device cuda:0 \
  --cameras head_camera --size 320 240 \
  --action_dim 14 --num_frames 33 --num_video_frames 9 \
  --num_action_layers 30 --cond_layer_stride 2 \
  --action_pred_target velocity --action_pos_mode rope --proprio_mode text \
  --action_snr_shift 5.0 --action_cond_sigma 0.0 \
  --video_inference_steps 25 --sigma_shift 5.0 \
  --action_inference_steps 50 --action_chunk_size 33
```

The 320 x 240 source images are aligned internally to a 320 x 256 model grid. The sampler step counts above are explicit inference settings, not training hyperparameters. The companion normalization file must come from the same release. The WebSocket client supplies `images`, `instruction`, and `qpos`; the server returns `actions`. It does not export MP4s by default. See [`inference/robotwin_policy/`](../inference/robotwin_policy/) for simulator integration.

**Scope:** the public policy server jointly denoises RGB and flow and uses a three-camera-prefix prompt even with head-only input. Step-500 was trained with `track1_conditional_rgb`, clean flow conditions, and no flow supervision; it does not supply a trained flow-generation head. Selecting the correct architecture settings does not eliminate these differences. This command documents the action-expert loading interface, not a validated closed-loop policy recipe. Use the Track 1 path for clean-flow video generation; evaluate any policy adaptation in simulation before considering real hardware.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No such file` for the checkpoint | Run from the repository root or pass an absolute checkpoint path; verify `SHA256SUMS` |
| Base model/tokenizer download starts unexpectedly | Preserve `models/Wan-AI/<model-name>/`; provide all DiT shards and tokenizer files |
| `episode set mismatch` | Supply the complete official input layout, even when generating only one episode |
| Missing or mismatched action-flow manifest | Run the preprocessor for the selected episode range and the exact same input trajectories |
| CUDA out of memory | Use a suitable GPU, available host RAM, and one inference process per device; checkpoint disk size is not a VRAM requirement |
| Action server behaves differently from video inference | These are different samplers; check the action-policy scope above |

The download names, configuration, and CLI options have been checked against the public release and source. End-to-end GPU inference and policy quality are not established by this documentation update. The [model card](https://huggingface.co/pumpkin601/KineWorld) states the checkpoint's evaluation scope.
