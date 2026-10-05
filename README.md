# KineWorld

KineWorld is an action-conditioned world model built on the Wan2.2-TI2V-5B video backbone. The code includes an RGB/optical-flow dual-stream generator, an action-prediction module, RoboTwin data preparation and policy integration, and a WorldArena2 Track 1 video-generation pipeline.

[**Project page: method figures and playable videos ↗**](https://modaxiansheng.github.io/KineWorld/)

![KineWorld method overview](docs/assets/kineworld_framework_v4.png)

## Video samples

Three clips from the provided 1,000-video collection can be played on the [project page](https://modaxiansheng.github.io/KineWorld/#videos). Their source checkpoint is not specified here.

| Sample 001 | Sample 312 | Sample 1000 |
| --- | --- | --- |
| [![Video sample 001](docs/assets/sample_001.png)](https://modaxiansheng.github.io/KineWorld/#videos) | [![Video sample 312](docs/assets/sample_312.png)](https://modaxiansheng.github.io/KineWorld/#videos) | [![Video sample 1000](docs/assets/sample_1000.png)](https://modaxiansheng.github.io/KineWorld/#videos) |

## More manuscript figures

The [project page](https://modaxiansheng.github.io/KineWorld/) includes the framework and transport figures, analytical TAWD illustrations, 15 additional single-view rollout pairs, and three multi-view validation plates from the V4 manuscript. The transport-support panel is conceptual; the rollout panels are qualitative examples, not benchmark results.

![Conceptual KineWorld transport support](docs/assets/transport_illustration.png)

![Qualitative rollout comparison from the manuscript](docs/assets/qualitative_comparison.jpg)

## Release status

This repository releases the source components listed below, method figures, and three video samples. It is not a complete reproduction package for the V4 manuscript. It contains no model weights, training data, or measured benchmark results. A checkpoint is required for inference. The step-500 checkpoint has not been shown to reproduce the V4 manuscript results. A source checkout or a dry run does not verify model quality.

## Public artifacts

- [KineWorld step-500 model repository](https://huggingface.co/pumpkin601/KineWorld): complete Safetensors checkpoint, paired action-normalization statistics, public training configuration, and SHA-256 checksums.
- [KineWorld video dataset repository](https://huggingface.co/datasets/pumpkin601/KineWorld-1000-Videos): archive of 1,000 MP4 videos, with archive size and SHA-256 recorded in the dataset card.

**Using the checkpoint:** follow [Model download](#model-download) and [Track 1 inference](#worldarena2-track-1-inference) below. For input layouts, preprocessing, offline setup, and the separate action-policy interface, see the [checkpoint usage guide](docs/checkpoint_usage.md).

## Contents

| Directory | Purpose |
| --- | --- |
| `diffsynth/` | Wan video pipeline and dual-stream model components |
| `training/` | RoboTwin data reader, training objective, and training entry point |
| `inference/` | Action inference server and RoboTwin policy client |
| `track1/` | Action-flow preparation, WorldArena2 Track 1 generation, and output validation |
| `data_generation/` | Robot-only rendering patch for RoboTwin demonstrations |
| `configs/` | Public input templates for training and inference |

The local HCU cluster orchestration used for one training run is not part of this source package. `training/train.sh` retains the reference run's data and memory checks; it is not a claim that another dataset or hardware setup reproduces that run.

## Environment

Use Linux, Python 3.10 or newer, a CUDA-capable PyTorch installation appropriate for your device, and `ffmpeg` for MP4 output. Install PyTorch and torchvision following their own device-specific instructions, then install the package from this directory:

```bash
git clone https://github.com/modaxiansheng/KineWorld.git
cd KineWorld
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The base video weights are `Wan-AI/Wan2.2-TI2V-5B`; the tokenizer files come from `Wan-AI/Wan2.1-T2V-1.3B`. Download access, GPU memory, and compatible attention kernels depend on the inference environment. Robot-only action-flow rendering also needs a compatible RoboTwin checkout, SAPIEN/Vulkan runtime, and RAFT weights. These assets and third-party model weights are not included here and retain their own terms.

## Model download

Run the following Bash commands from the repository root. The released checkpoint is **`step-500.safetensors`** (13.11 GB), not a LoRA adapter or a standalone Diffusers pipeline. Use this repository's loader, rather than `DiffusionPipeline.from_pretrained("pumpkin601/KineWorld")`.

```bash
hf download pumpkin601/KineWorld \
  step-500.safetensors action_norm_stats.npz training_config_public.json SHA256SUMS \
  --revision 37e8f86c6c3cf45cde743162bf9b6de583cf1b73 \
  --local-dir checkpoints/KineWorld

(cd checkpoints/KineWorld && sha256sum -c SHA256SUMS)
```

`hf` is supplied by `huggingface_hub`. If it is unavailable, run `python -m pip install --upgrade huggingface_hub` in the same environment. Both checksum entries must report `OK`.

| File | Used for |
| --- | --- |
| `step-500.safetensors` | Full-mode training checkpoint containing DiT, clean-flow input weights, and the action expert; video-only inference skips the action expert |
| `action_norm_stats.npz` | Paired normalization for action-policy inference; not needed by the Track 1 video generator |
| `training_config_public.json` | Checkpoint-specific settings; reference metadata, not an inference config automatically loaded by the scripts |
| `SHA256SUMS` | Checksum verification for the checkpoint and normalization file |

Download the separate Wan dependencies into the layout expected by the loader. Keep the `Wan-AI/` subdirectory:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
            'models_t5_umt5-xxl-enc-bf16.pth' 'Wan2.2_VAE.pth' \
  --local-dir models/Wan-AI/Wan2.2-TI2V-5B

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include 'google/*' \
  --local-dir models/Wan-AI/Wan2.1-T2V-1.3B
```

These downloads are additional to the 13.11-GB checkpoint. The Track 1 loader can also fetch missing base files automatically. Complete the downloads before an offline run.

## WorldArena2 Track 1 inference

Use this entry point to generate videos from an initial image, instruction, and action trajectory. Prepare the official input collection and action-flow files first, following the [checkpoint usage guide](docs/checkpoint_usage.md#2-prepare-the-inputs) or [track1/README.md](track1/README.md). The default `action_flow` mode requires real per-chunk action flow; it does not silently fall back to text-only generation.

Start with one episode using the downloaded checkpoint:

```bash
python track1/infer_track1.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root outputs/kineworld-step500-episode1 \
  --checkpoint-path checkpoints/KineWorld/step-500.safetensors \
  --model-cache-dir models \
  --conditioning-mode action_flow \
  --action-flow-provider precomputed \
  --precomputed-flow-root /path/to/action_flow \
  --episode-start 1 --episode-end 1 \
  --num-inference-steps 25 --seed 1 \
  --device cuda:0
```

The video is written to `outputs/kineworld-step500-episode1/videos/episode1.mp4` (640 x 480, 24 fps), with a run configuration and per-episode records. **Even for one selected episode, the current adapter requires the complete `episode1` through `episode1000` input layout.** The [guide](docs/checkpoint_usage.md) explains this requirement, output validation, and full-collection generation.

Alternatively, replace `--checkpoint-path ...` with these three options to download the checkpoint directly through the inference script. Do not combine the two checkpoint sources:

```bash
--checkpoint-repo pumpkin601/KineWorld \
--checkpoint-file step-500.safetensors \
--checkpoint-revision 37e8f86c6c3cf45cde743162bf9b6de583cf1b73
```

`--dry-run` checks episode discovery and sharding without loading a checkpoint or running the model. It does not validate action-flow files or generated-video quality. A real smoke test must omit `--dry-run` and inspect the resulting MP4 and records.

## RoboTwin action policy

This is a separate, experimental interface that predicts actions, not the action-conditioned video generator above. It needs `action_norm_stats.npz` from the same release. **Do not use the generic `start_server.sh` defaults for step-500:** the published configuration uses head-camera input and `cond_layer_stride=2`, whereas that wrapper uses three cameras and stride 1. See the [explicit step-500 server command and its limitations](docs/checkpoint_usage.md#5-optional-action-policy-server) before using the RoboTwin client in `inference/robotwin_policy/`.

## Training inputs

`training/train.sh` requires an existing RoboTwin training root, a JSONL training manifest and its SHA-256, and a compatible warm-start checkpoint and its SHA-256. These are explicit inputs; the script verifies the manifest and checkpoint before launching. For example:

```bash
DATASET_BASE_PATH=/path/to/robotwin_training \
TRAINING_MANIFEST=/path/to/train.jsonl \
TRAINING_MANIFEST_SHA256=YOUR_64_HEX_SHA256 \
RESUME_CHECKPOINT=/path/to/warm_start.safetensors \
RESUME_CHECKPOINT_SHA256=YOUR_64_HEX_SHA256 \
NUM_GPUS=1 bash training/train.sh
```

The reference recipe uses head-camera RoboTwin data, 14-dimensional actions, robot-only optical flow, and a VRAM check calibrated for the original 64-GiB accelerator class. Other hardware or data need independent validation. The step-500 weight file is an output of training, not the warm-start input above.

## License and provenance

The code package contains an Apache-2.0 [LICENSE](LICENSE) and [third-party notices](NOTICE). Components adapted from DiffSynth-Studio and RoboTwin are identified there. Wan model weights, RoboTwin assets/data, and any other external components must be obtained under their respective licenses. The [step-500 model card](https://huggingface.co/pumpkin601/KineWorld) records the release's training provenance and scope. No score is claimed by this source package.
