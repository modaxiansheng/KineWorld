"""
Dual-stream (RGB + optical-flow) Wan2.2 world model + IDM action expert training
for RoboTwin.

The video backbone jointly denoises an RGB stream and a head-camera optical-flow
stream with joint self-attention; multi-camera views are combined by T-shape
tiling (head at full resolution, wrists at half). The action expert predicts an
action chunk by cross-attending to the video DiT's per-layer hidden states,
captured while the video is denoised (see ``dual_stream_capture``). Video and
action losses are optimized jointly.

Data can be read online (real RAFT optical flow computed on full-scene frames by
default) or from a precomputed latent cache (see ``precompute_latents.py``).
The legacy robot-only mode is explicit and requires paired render files.
"""

import os
import sys
import math
import json
import hashlib
import contextlib
import random

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import (
    ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.pipelines.wan_video_dual_stream import model_fn_wan_video_dual_stream
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth import load_state_dict

from action_dit import build_action_expert_idm
from dual_stream_capture import capture_video_layer_features, concat_layer_feats
from objective_contract import (
    TRACK1_CONDITIONAL_RGB,
    freeze_conditional_flow_head,
    validate_video_objective,
)
from resume_contract import (
    ResumeBindingError,
    build_resume_binding,
    build_trainer_state,
    resolve_vram_probe_optimizer_step,
    validate_trainer_state,
)
from trainer_startup_contract import publish_trainer_ready
from dataset_action_robotwin import (
    RoboTwinActionFlowDataset,
    TRACK1_FINAL_TRAIN_MANIFEST_SHA256,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FORMAL_MIN_VRAM_GIB = 54.0
FORMAL_MAX_VRAM_GIB = 63.0


class DtypeShardedAdamW(torch.optim.Optimizer):
    """One native ZRO AdamW per dense parameter dtype.

    PyTorch 2.5's ``ZeroRedundancyOptimizer`` rejects mixed BF16/FP32
    parameter lists. KineWorld intentionally contains both, so this adapter
    partitions parameters by dense tensor type and advances the resulting ZRO
    instances as one optimizer. The public parameter group remains the source
    of truth for the LR scheduler and is synchronized before every step.

    Full optimizer-state serialization is deliberately unsupported here; the
    training entrypoint enforces ``full_state_keep=0`` and writes periodic
    safetensors weight checkpoints instead.
    """

    def __init__(self, params, **adamw_kwargs):
        params = [parameter for parameter in params if parameter.requires_grad]
        if not params:
            raise ValueError("DtypeShardedAdamW received no trainable parameters")
        super().__init__(params, dict(adamw_kwargs))

        from torch.distributed.optim import ZeroRedundancyOptimizer

        by_dense_type = {}
        for parameter in params:
            if parameter.is_sparse:
                raise ValueError("DtypeShardedAdamW does not support sparse parameters")
            by_dense_type.setdefault(torch.typename(parameter), []).append(parameter)

        self._zro_optimizers = [
            ZeroRedundancyOptimizer(
                by_dense_type[dense_type],
                optimizer_class=torch.optim.AdamW,
                parameters_as_bucket_view=False,
                overlap_with_ddp=False,
                **adamw_kwargs,
            )
            for dense_type in sorted(by_dense_type)
        ]
        self._step_has_run = False

    def _sync_public_group(self):
        public_group = self.param_groups[0]
        for optimizer in self._zro_optimizers:
            for child_group in optimizer.param_groups:
                for name in self.defaults:
                    if name in public_group:
                        child_group[name] = public_group[name]

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("DtypeShardedAdamW does not support closures")
        self._sync_public_group()
        # Set before touching a child: even a partially applied multi-dtype
        # step must never be represented by the empty public shell state.
        self._step_has_run = True
        result = None
        for optimizer in self._zro_optimizers:
            child_result = optimizer.step()
            if result is None:
                result = child_result
        return result

    def zero_grad(self, set_to_none=True):
        for optimizer in self._zro_optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Accelerate performs one empty-state round trip while wrapping an
        # optimizer for device placement. Permit exactly that startup probe.
        # Once any child AdamW state exists, fail closed so a caller cannot
        # mistake this public shell for a consolidated full optimizer state.
        if self._step_has_run:
            raise RuntimeError(
                "DtypeShardedAdamW full-state serialization is disabled; "
                "use safetensors weight checkpoints"
            )
        return super().state_dict()

    def load_state_dict(self, state_dict):
        if state_dict.get("state"):
            raise RuntimeError(
                "DtypeShardedAdamW full-state resume is disabled; "
                "use a safetensors weight checkpoint"
            )
        result = super().load_state_dict(state_dict)
        self._sync_public_group()
        return result


class DtypeHybridAdamW(torch.optim.Optimizer):
    """Mathematically exact AdamW with partially replicated optimizer state.

    Ordinary DDP still replicates the model and gradients on every rank. A
    deterministic fraction of trainable parameters uses regular AdamW, while
    the remainder uses dtype-safe ``ZeroRedundancyOptimizer`` children. Both
    paths run the same AdamW update; only the physical placement of persistent
    first/second-moment tensors differs. This provides a real-workload memory
    point between fully replicated AdamW and fully sharded ZRO without adding
    synthetic allocations.
    """

    def __init__(self, params, *, replicated_fraction, **adamw_kwargs):
        params = [parameter for parameter in params if parameter.requires_grad]
        if not params:
            raise ValueError("DtypeHybridAdamW received no trainable parameters")
        replicated_fraction = float(replicated_fraction)
        if not 0.0 < replicated_fraction < 1.0:
            raise ValueError("replicated_fraction must be strictly between 0 and 1")
        super().__init__(params, dict(adamw_kwargs))

        from torch.distributed.optim import ZeroRedundancyOptimizer

        total_numel = sum(parameter.numel() for parameter in params)
        target_numel = int(round(total_numel * replicated_fraction))
        replicated_params = []
        sharded_params = []
        replicated_numel = 0
        for parameter in params:
            if parameter.is_sparse:
                raise ValueError("DtypeHybridAdamW does not support sparse parameters")
            if replicated_numel < target_numel:
                replicated_params.append(parameter)
                replicated_numel += parameter.numel()
            else:
                sharded_params.append(parameter)
        if not replicated_params or not sharded_params:
            raise ValueError(
                "replicated_fraction produced an empty replicated or sharded partition"
            )

        by_dense_type = {}
        for parameter in sharded_params:
            by_dense_type.setdefault(torch.typename(parameter), []).append(parameter)

        self._replicated_optimizer = torch.optim.AdamW(
            replicated_params, **adamw_kwargs
        )
        self._zro_optimizers = [
            ZeroRedundancyOptimizer(
                by_dense_type[dense_type],
                optimizer_class=torch.optim.AdamW,
                parameters_as_bucket_view=False,
                overlap_with_ddp=False,
                **adamw_kwargs,
            )
            for dense_type in sorted(by_dense_type)
        ]
        self._child_optimizers = [
            self._replicated_optimizer,
            *self._zro_optimizers,
        ]
        self.replicated_fraction = replicated_numel / total_numel
        self.replicated_numel = replicated_numel
        self.sharded_numel = total_numel - replicated_numel
        self._step_has_run = False
        print(
            "[HybridAdamW] "
            f"replicated={self.replicated_numel:,} params "
            f"sharded={self.sharded_numel:,} params "
            f"actual_replicated_fraction={self.replicated_fraction:.6f}"
        )

    def _sync_public_group(self):
        public_group = self.param_groups[0]
        for optimizer in self._child_optimizers:
            for child_group in optimizer.param_groups:
                for name in self.defaults:
                    if name in public_group:
                        child_group[name] = public_group[name]

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("DtypeHybridAdamW does not support closures")
        self._sync_public_group()
        self._step_has_run = True
        result = None
        for optimizer in self._child_optimizers:
            child_result = optimizer.step()
            if result is None:
                result = child_result
        return result

    def zero_grad(self, set_to_none=True):
        for optimizer in self._child_optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        if self._step_has_run:
            raise RuntimeError(
                "DtypeHybridAdamW full-state serialization is disabled; "
                "use safetensors weight checkpoints"
            )
        return super().state_dict()

    def load_state_dict(self, state_dict):
        if state_dict.get("state"):
            raise RuntimeError(
                "DtypeHybridAdamW full-state resume is disabled; "
                "use a safetensors weight checkpoint"
            )
        result = super().load_state_dict(state_dict)
        self._sync_public_group()
        return result


def dual_stream_action_collate_fn(batch):
    """Collate dual-stream samples: tiled RGB + single flow.

    Supports both online (PIL-based) and cached (latent tensor) samples. When
    the cache also contains pre-encoded T5 embeddings, stacks them so the
    trainer can skip the text encoder entirely in cache-mode training.
    """
    if "rgb_input_latents" in batch[0]:
        out = {
            "rgb_input_latents": torch.stack([s["rgb_input_latents"] for s in batch]),
            "flow_input_latents": torch.stack([s["flow_input_latents"] for s in batch]),
            "video_prompt": [s["video_prompt"] for s in batch],
            "action_prompt": [s["action_prompt"] for s in batch],
            "actions": np.stack([s["actions"] for s in batch], axis=0),
            "task": [s["task"] for s in batch],
        }
        if "variant" in batch[0]:
            out["variant"] = [s["variant"] for s in batch]
        if "num_frames_pixel" in batch[0]:
            out["num_frames_pixel"] = [int(s["num_frames_pixel"]) for s in batch]
        if "video_context" in batch[0] and "action_context" in batch[0]:
            out["video_context"] = torch.stack([s["video_context"] for s in batch])
            out["action_context"] = torch.stack([s["action_context"] for s in batch])
        return out
    return {
        "tiled_rgb_video": [sample["tiled_rgb_video"] for sample in batch],
        "flow_video": [sample["flow_video"] for sample in batch],
        "video_prompt": [sample["video_prompt"] for sample in batch],
        "action_prompt": [sample["action_prompt"] for sample in batch],
        "actions": np.stack([sample["actions"] for sample in batch], axis=0),
        "task": [sample["task"] for sample in batch],
    }


class EpisodeAwareCacheBatchSampler(torch.utils.data.Sampler):
    """Batch cached chunks with episode locality while covering every chunk.

    Chunks from the same episode share a cached (decoded) episode tensor, so
    grouping ``chunks_per_episode`` chunks of one episode per batch keeps the
    per-episode LRU cache warm. Every chunk is still visited once per epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        chunks_per_episode: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if not getattr(dataset, "load_from_cache", False):
            raise ValueError("EpisodeAwareCacheBatchSampler requires cache mode")
        if not hasattr(dataset, "_cached_chunks"):
            raise ValueError("dataset is missing _cached_chunks")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.chunks_per_episode = int(chunks_per_episode)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.chunks_per_episode <= 0:
            raise ValueError("chunks_per_episode must be positive")
        if self.chunks_per_episode > self.batch_size:
            raise ValueError("chunks_per_episode must be <= batch_size")
        if self.batch_size % self.chunks_per_episode != 0:
            raise ValueError(
                "batch_size must be divisible by chunks_per_episode so local "
                "batches have a stable number of episode groups"
            )

        episode_to_indices = {}
        for idx, info in enumerate(dataset._cached_chunks):
            episode_to_indices.setdefault(info["pt_path"], []).append(idx)

        self.episode_to_indices = episode_to_indices
        self.episode_keys = sorted(episode_to_indices)
        self.groups_per_batch = self.batch_size // self.chunks_per_episode
        self.num_regular_groups = sum(
            len(indices) // self.chunks_per_episode
            for indices in episode_to_indices.values()
        )
        self.num_tail_chunks = sum(
            len(indices) % self.chunks_per_episode
            for indices in episode_to_indices.values()
        )
        self.num_samples = sum(len(v) for v in episode_to_indices.values())
        self._num_batches = self._compute_num_batches()

    def _compute_num_batches(self) -> int:
        full_regular_batches = self.num_regular_groups // self.groups_per_batch
        leftover_regular_groups = self.num_regular_groups % self.groups_per_batch
        remainder_chunks = (
            leftover_regular_groups * self.chunks_per_episode
            + self.num_tail_chunks
        )
        remainder_batches = remainder_chunks // self.batch_size
        has_partial = remainder_chunks % self.batch_size != 0
        if has_partial and not self.drop_last:
            remainder_batches += 1
        return full_regular_batches + remainder_batches

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        regular_groups = []
        tail_indices = []
        episode_keys = list(self.episode_keys)
        rng.shuffle(episode_keys)

        for key in episode_keys:
            indices = list(self.episode_to_indices[key])
            rng.shuffle(indices)
            split = (len(indices) // self.chunks_per_episode) * self.chunks_per_episode
            for start in range(0, split, self.chunks_per_episode):
                regular_groups.append(indices[start:start + self.chunks_per_episode])
            tail_indices.extend(indices[split:])

        rng.shuffle(regular_groups)

        batch = []
        for group in regular_groups:
            batch.extend(group)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        tail_indices = batch + tail_indices
        rng.shuffle(tail_indices)
        for start in range(0, len(tail_indices), self.batch_size):
            batch = tail_indices[start:start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


class AbsoluteEpochRandomSampler(Sampler):
    """Deterministic online-data order keyed by the absolute epoch.

    ``DataLoader(shuffle=True)`` draws a fresh permutation from the restored
    process RNG before ``skip_first_batches`` can skip already-consumed work.
    On a mid-epoch resume that produces a different permutation.  This sampler
    makes the global order a pure function of ``seed + absolute_epoch`` so the
    skipped prefix and the remaining suffix are identical after a restart.
    """

    def __init__(self, data_source, seed: int = 42):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        if self.seed < 0:
            raise ValueError("data shuffle seed must be non-negative")

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())


class FlowActionTrainingModule(DiffusionTrainingModule):
    """Dual-stream video + IDM per-layer-conditioned action expert."""

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        tokenizer_model_id=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        resume_checkpoint=None,
        video_objective="track1_conditional_rgb",
        flow_loss_weight=0.0,
        action_loss_weight=5.0,
        action_dim=14,
        num_action_layers=30,
        num_frames=49,
        cameras=None,
        action_expert_checkpoint=None,
        action_snr_shift=3.0,
        flow_motion_boost=2.0,
        text_context_dim=0,
        ref_aug_strength=0.1,
        fp32_modulation=True,
        # ---- IDM conditioning ----
        cond_noise_prob=0.5,
        cond_detach=True,
        cond_layer_stride=1,
        action_expert_dim=1024,
        action_expert_heads=16,
        action_expert_ffn_dim=4096,
        # ---- action-head options ----
        action_pred_target="velocity",
        action_use_rope=True,
        proprio_mode="text",
        loss_timestep_weighting=True,
    ):
        super().__init__()
        self.video_objective = validate_video_objective(video_objective)

        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1],
            )
        tokenizer_model_id = (
            tokenizer_model_id
            or os.environ.get("KINEWORLD_TOKENIZER_MODEL_ID")
            or "Wan-AI/Wan2.1-T2V-1.3B"
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
            tokenizer_config=ModelConfig(
                model_id=tokenizer_model_id,
                origin_file_pattern="google/*",
            ),
            # The server snapshot intentionally co-locates DiT/T5/VAE/tokenizer.
            # Honor the explicit model IDs instead of redirecting T5 to Wan2.1.
            redirect_common_files=False,
        )
        self.tokenizer_model_id = tokenizer_model_id

        self.vae_z_dim = getattr(self.pipe.vae, "z_dim", 16)
        self.cameras = cameras or ["head_camera", "left_camera", "right_camera"]
        self.lora_mode = lora_base_model is not None

        self.flow_stream = init_flow_stream(self.pipe.dit)

        # ---- IDM action expert: conditions on video DiT per-layer features ----
        video_dim = int(self.pipe.dit.dim)
        self.action_pred_target = action_pred_target
        self.action_expert = build_action_expert_idm(
            action_dim=action_dim,
            num_frames=num_frames,
            video_dim=video_dim,
            text_context_dim=text_context_dim,
            joint_state_dim=action_dim,
            num_layers=num_action_layers,
            dim=action_expert_dim,
            num_heads=action_expert_heads,
            ffn_dim=action_expert_ffn_dim,
            pred_target=action_pred_target,
            use_rope=action_use_rope,
            proprio_mode=proprio_mode,
        )

        lora_resume_state = None
        if resume_checkpoint is not None:
            if self.lora_mode:
                lora_resume_state = self._load_lora_resume_checkpoint_pre(resume_checkpoint)
            else:
                self._load_resume_checkpoint(resume_checkpoint)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        if self.lora_mode:
            self._unfreeze_dual_stream_modules()
            if lora_resume_state is not None:
                self._load_lora_resume_checkpoint_post(lora_resume_state)

        frozen_flow_head_params = freeze_conditional_flow_head(
            self.flow_stream, self.video_objective
        )
        if frozen_flow_head_params:
            print(
                "[VideoObjective] froze unused flow prediction head: "
                f"{frozen_flow_head_params:,} parameters"
            )

        if fp32_modulation:
            self._upcast_modulation_to_fp32()

        self.ref_aug_strength = ref_aug_strength
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.flow_loss_weight = float(flow_loss_weight)
        self.action_loss_weight = float(action_loss_weight)
        if self.video_objective == "track1_conditional_rgb" and self.flow_loss_weight != 0.0:
            raise ValueError(
                "track1_conditional_rgb keeps flow clean and does not supervise "
                "the flow head; flow_loss_weight must be exactly 0"
            )
        if not 0.0 <= self.flow_loss_weight <= 1.0:
            raise ValueError("flow_loss_weight must be in [0, 1]")
        if self.action_loss_weight < 0.0:
            raise ValueError("action_loss_weight must be non-negative")
        self.flow_motion_boost = flow_motion_boost

        self.cond_noise_prob = float(cond_noise_prob)
        self.cond_detach = bool(cond_detach)
        self.cond_layer_stride = max(1, int(cond_layer_stride))
        self.loss_timestep_weighting = bool(loss_timestep_weighting)

        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=action_snr_shift,
        )
        self.action_scheduler.set_timesteps(1000, training=True)
        print(f"[ActionScheduler] snr_shift={action_snr_shift}, "
              f"cond_noise_prob={self.cond_noise_prob}, cond_detach={self.cond_detach}, "
              f"cond_layer_stride={self.cond_layer_stride}")
        print(
            f"[VideoObjective] {self.video_objective}; "
            f"flow_supervised={self.video_objective == 'joint_dual_stream'}; "
            f"flow_loss_weight={self.flow_loss_weight}"
        )

        if action_expert_checkpoint is not None and os.path.exists(action_expert_checkpoint):
            state_dict = load_state_dict(action_expert_checkpoint)
            prefix = "action_expert."
            action_keys = {k[len(prefix):]: v for k, v in state_dict.items()
                           if k.startswith(prefix)}
            if not action_keys:
                action_keys = state_dict
            missing, unexpected = self.action_expert.load_state_dict(action_keys, strict=False)
            loaded = len(action_keys) - len(unexpected)
            print(f"[ActionExpertIDM] Loaded {loaded} keys from {action_expert_checkpoint}")

    def _load_resume_checkpoint(self, ckpt_path):
        """Warm-start the video DiT + flow_stream; load action keys only when
        they are shape-compatible.

        The action expert's video cross-attn K/V are ``Linear(video_dim -> dim)``.
        Action keys are filtered by name AND shape so warm-starting from a
        checkpoint that lacks a matching action expert simply loads zero action
        keys (fresh action expert, warm video), while resuming an IDM run loads
        all of them.
        """
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path)
        dit_keys, flow_keys, action_keys = {}, {}, {}
        for k, v in state_dict.items():
            if k.startswith("action_expert."):
                action_keys[k[len("action_expert."):]] = v
            elif k.startswith("flow_stream."):
                flow_keys[k[len("flow_stream."):]] = v
            else:
                dit_keys[k] = v
        dit_result = self.pipe.dit.load_state_dict(dit_keys, strict=False)
        flow_result = self.flow_stream.load_state_dict(flow_keys, strict=False)
        ae_state = self.action_expert.state_dict()
        compat = {
            k: v for k, v in action_keys.items()
            if k in ae_state and tuple(v.shape) == tuple(ae_state[k].shape)
        }
        action_result = self.action_expert.load_state_dict(compat, strict=False)
        # The launcher selects the verified source profile by checkpoint digest,
        # so a harmless local filename change does not alter coverage checks.
        warm_start_profile = os.environ.get("KINEWORLD_WARM_START_PROFILE", "custom")
        official_robotwin = warm_start_profile == "robotwin_pretrained"
        official_stage1 = warm_start_profile == "worldarena_stage1"
        if official_robotwin or official_stage1:
            coverage_errors = []
            expected_source_counts = (
                (825, 6, 1191) if official_robotwin else (825, 5, 0)
            )
            actual_source_counts = (
                len(dit_keys), len(flow_keys), len(action_keys)
            )
            if actual_source_counts != expected_source_counts:
                coverage_errors.append(
                    "source DiT/flow/action counts="
                    f"{actual_source_counts} != {expected_source_counts}"
                )
            if dit_result.missing_keys or dit_result.unexpected_keys:
                coverage_errors.append(
                    "DiT missing/unexpected="
                    f"{len(dit_result.missing_keys)}/{len(dit_result.unexpected_keys)}"
                )
            if official_robotwin:
                if flow_result.missing_keys or flow_result.unexpected_keys:
                    coverage_errors.append(
                        "flow missing/unexpected="
                        f"{len(flow_result.missing_keys)}/"
                        f"{len(flow_result.unexpected_keys)}"
                    )
                incompatible_action_shapes = len(action_keys) - len(compat)
                if (
                    action_result.missing_keys
                    or action_result.unexpected_keys
                    or incompatible_action_shapes
                ):
                    coverage_errors.append(
                        "action missing/unexpected/incompatible="
                        f"{len(action_result.missing_keys)}/"
                        f"{len(action_result.unexpected_keys)}/"
                        f"{incompatible_action_shapes}"
                    )
            else:
                # The audited Stage1 checkpoint predates stream_embed. This is
                # its one intentional random-init parameter; every other flow
                # key must load and no action keys may be present.
                if (
                    set(flow_result.missing_keys) != {"stream_embed"}
                    or flow_result.unexpected_keys
                ):
                    coverage_errors.append(
                        "Stage1 flow missing/unexpected="
                        f"{flow_result.missing_keys}/"
                        f"{flow_result.unexpected_keys}; only stream_embed may be missing"
                    )
                if action_keys:
                    coverage_errors.append(
                        "Stage1 unexpectedly contains action_expert keys"
                    )
            if coverage_errors:
                raise RuntimeError(
                    "official warm-start checkpoint failed exact module coverage: "
                    + "; ".join(coverage_errors)
                )
            if official_stage1:
                print(
                    "[Resume] Stage1 A/B: flow_stream.stream_embed and the "
                    "entire action expert are intentionally random-initialized"
                )
        print(
            f"[Resume] loaded dit={len(dit_keys)} flow={len(flow_keys)} "
            f"action(shape-compatible)={len(compat)}/{len(action_keys)}; "
            f"exact_official_coverage={official_robotwin or official_stage1}"
        )

    def _unfreeze_dual_stream_modules(self):
        """In LoRA mode, unfreeze flow_stream and the DiT modulation params."""
        self.flow_stream.requires_grad_(True)
        self.flow_stream.train()
        n_flow = sum(p.numel() for p in self.flow_stream.parameters())

        modulation_nn_modules = [
            self.pipe.dit.time_projection,
            self.pipe.dit.time_embedding,
        ]
        for module in modulation_nn_modules:
            module.requires_grad_(True)
            module.train()

        modulation_params = []
        for block in self.pipe.dit.blocks:
            if hasattr(block, 'modulation') and isinstance(block.modulation, nn.Parameter):
                block.modulation.requires_grad = True
                modulation_params.append(block.modulation)
        if hasattr(self.pipe.dit.head, 'modulation') and isinstance(self.pipe.dit.head.modulation, nn.Parameter):
            self.pipe.dit.head.modulation.requires_grad = True
            modulation_params.append(self.pipe.dit.head.modulation)

        n_mod_nn = sum(p.numel() for m in modulation_nn_modules for p in m.parameters())
        n_mod_param = sum(p.numel() for p in modulation_params)
        print(f"[LoRA+DualStream] Unfroze flow_stream ({n_flow:,} params)")
        print(f"[LoRA+Modulation] Unfroze time_projection, time_embedding, "
              f"{len(modulation_params)} modulation params ({n_mod_nn + n_mod_param:,} params)")

    @staticmethod
    def _register_fp32_hooks(module):
        module.float()

        def pre_hook(_mod, args):
            return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)

        def post_hook(_mod, _args, output):
            return output.bfloat16() if isinstance(output, torch.Tensor) else output

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def _upcast_modulation_to_fp32(self):
        """Keep modulation, time-MLP and LayerNorm math in fp32 (bf16 elsewhere)."""
        upcasted = 0
        for block in self.pipe.dit.blocks:
            if hasattr(block, 'modulation') and isinstance(block.modulation, nn.Parameter):
                block.modulation.data = block.modulation.data.float()
                upcasted += block.modulation.numel()
        if hasattr(self.pipe.dit.head, 'modulation') and isinstance(self.pipe.dit.head.modulation, nn.Parameter):
            self.pipe.dit.head.modulation.data = self.pipe.dit.head.modulation.data.float()
            upcasted += self.pipe.dit.head.modulation.numel()

        n_hooked = 0
        for module in [self.pipe.dit.time_embedding, self.pipe.dit.time_projection]:
            self._register_fp32_hooks(module)
            upcasted += sum(p.numel() for p in module.parameters())
            n_hooked += 1

        for module in self.pipe.dit.modules():
            if isinstance(module, nn.LayerNorm):
                self._register_fp32_hooks(module)
                upcasted += sum(p.numel() for p in module.parameters())
                n_hooked += 1

        print(f"[FP32Modulation] Upcasted {upcasted:,} params to float32 "
              f"({n_hooked} modules with FP32 hooks)")

    def _load_lora_resume_checkpoint_pre(self, ckpt_path):
        print(f"[LoRA Resume] Loading checkpoint: {ckpt_path}")
        state_dict = load_state_dict(ckpt_path)
        lora_keys = {k: v for k, v in state_dict.items()
                     if "lora_A" in k or "lora_B" in k}
        module_keys = {k: v for k, v in state_dict.items()
                       if k not in lora_keys}
        dit_keys = {}
        flow_keys = {}
        for k, v in module_keys.items():
            if k.startswith("flow_stream."):
                flow_keys[k.replace("flow_stream.", "")] = v
            else:
                dit_keys[k] = v
        if dit_keys:
            self.pipe.dit.load_state_dict(dit_keys, strict=False)
            print(f"[LoRA Resume Pre] Loaded {len(dit_keys)} DiT module keys")
        if flow_keys:
            self.flow_stream.load_state_dict(flow_keys, strict=False)
            print(f"[LoRA Resume Pre] Loaded {len(flow_keys)} FlowStream module keys")
        print(f"[LoRA Resume] Deferred {len(lora_keys)} LoRA keys")
        return lora_keys

    def _load_lora_resume_checkpoint_post(self, lora_state_dict):
        mapped = self.mapping_lora_state_dict(lora_state_dict)
        self.pipe.dit.load_state_dict(mapped, strict=False)
        print(f"[LoRA Resume Post] Loaded {len(mapped)} LoRA keys into DiT")

    def _forward_preprocess_from_cache(self, data):
        """Build inputs dict from pre-encoded latents (cache mode).

        Only runs noise sampling; skips VAE entirely. Uses pre-encoded T5
        embeddings stacked by the collate fn when present so the text encoder
        can be left off GPU (and off disk) for the whole run.
        """
        dev = self.pipe.device
        dt = self.pipe.torch_dtype

        rgb_input_latents = data["rgb_input_latents"].to(dtype=dt, device=dev)
        flow_input_latents = data["flow_input_latents"].to(dtype=dt, device=dev)
        B = rgb_input_latents.shape[0]

        if "video_context" in data and "action_context" in data:
            # Contexts already encoded (per-episode, per-instruction) and
            # re-padded to (B, 512, 4096) by the collate fn.
            context = data["video_context"].to(dtype=dt, device=dev)
            action_context = data["action_context"].to(dtype=dt, device=dev)
        else:
            # Encode text on CPU to keep the ~11 GB T5 off the GPU (online
            # per-chunk VAE+RAFT already saturates VRAM); move the small
            # encoded contexts to the GPU.
            if next(self.pipe.text_encoder.parameters()).device.type != "cpu":
                self.pipe.text_encoder.to("cpu")
            context = self.pipe.prompter.encode_prompt(
                data["video_prompt"], positive=True, device="cpu",
            ).to(dtype=dt, device=dev)
            action_context = self.pipe.prompter.encode_prompt(
                data["action_prompt"], positive=True, device="cpu",
            ).to(dtype=dt, device=dev)

        rgb_noise = torch.randn_like(rgb_input_latents)
        flow_noise = torch.randn_like(flow_input_latents)

        # First-frame latent for the flow-motion weight.
        flow_fz = flow_input_latents[:, :, 0:1]
        flow_motion_weight = None
        if self.flow_motion_boost > 0:
            with torch.no_grad():
                deviation = (flow_input_latents - flow_fz).abs().mean(dim=1, keepdim=True)
                dev_max = deviation.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
                motion_w = 1.0 + self.flow_motion_boost * (deviation / dev_max)
            flow_motion_weight = motion_w

        # Authoritative pixel-frame count from cache metadata when the
        # dataloader forwarded it; otherwise invert the VAE temporal
        # compression: T_pixel = (T_z - 1) * 4 + 1.
        if "num_frames_pixel" in data and len(data["num_frames_pixel"]) > 0:
            num_frames = int(data["num_frames_pixel"][0])
        else:
            num_frames = (rgb_input_latents.shape[2] - 1) * 4 + 1

        return {
            "rgb_input_latents": rgb_input_latents,
            "rgb_noise": rgb_noise,
            "flow_input_latents": flow_input_latents,
            "flow_noise": flow_noise,
            "flow_for_action": flow_input_latents.detach(),
            "rgb_for_action": rgb_input_latents.detach(),
            "flow_motion_weight": flow_motion_weight,
            "context": context,
            "action_context": action_context,
            "fuse_vae_embedding_in_latents": True,
            "num_frames": num_frames,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

    def forward_preprocess(self, data):
        """Encode tiled RGB and flow into separate latent tensors (online mode)."""
        if "rgb_input_latents" in data:
            return self._forward_preprocess_from_cache(data)

        B = len(data["tiled_rgb_video"])

        rgb_frame_0 = data["tiled_rgb_video"][0][0]
        rgb_w, rgb_h = rgb_frame_0.size
        num_frames = len(data["tiled_rgb_video"][0])
        rgb_h, rgb_w, num_frames = self.pipe.check_resize_height_width(rgb_h, rgb_w, num_frames)

        flow_frame_0 = data["flow_video"][0][0]
        flow_w, flow_h = flow_frame_0.size
        flow_h, flow_w, _ = self.pipe.check_resize_height_width(flow_h, flow_w, num_frames)

        self.pipe.load_models_to_device(["text_encoder"])
        context = self.pipe.prompter.encode_prompt(
            data["video_prompt"], positive=True, device=self.pipe.device
        )
        action_context = self.pipe.prompter.encode_prompt(
            data["action_prompt"], positive=True, device=self.pipe.device
        )
        self.pipe.load_models_to_device(["vae"])

        dev = self.pipe.device
        dt = self.pipe.torch_dtype

        per_rgb_input = []
        per_rgb_noise = []
        per_flow_input = []
        per_flow_noise = []
        per_flow_for_action = []
        per_rgb_for_action = []
        per_flow_motion_weight = []

        for i in range(B):
            rgb_vid = self.pipe.preprocess_video(
                [f.resize((rgb_w, rgb_h)) for f in data["tiled_rgb_video"][i]]
            )
            rgb_first = self.pipe.preprocess_image(
                data["tiled_rgb_video"][i][0].resize((rgb_w, rgb_h))
            ).transpose(0, 1)

            rgb_z = self.pipe.vae.encode(rgb_vid, device=dev).to(dtype=dt, device=dev)
            rgb_fz = self.pipe.vae.encode([rgb_first], device=dev).to(dtype=dt, device=dev)

            # Align latent frame 0 with the I2V first-image encoding; the prefix
            # latent is kept clean in ``forward`` via the prefix override.
            rgb_z[:, :, 0:1] = rgb_fz
            rgb_noise = torch.randn_like(rgb_z)

            per_rgb_input.append(rgb_z)
            per_rgb_noise.append(rgb_noise)

            flow_vid = self.pipe.preprocess_video(
                [f.resize((flow_w, flow_h)) for f in data["flow_video"][i]]
            )
            flow_first = self.pipe.preprocess_image(
                data["flow_video"][i][0].resize((flow_w, flow_h))
            ).transpose(0, 1)

            flow_z = self.pipe.vae.encode(flow_vid, device=dev).to(dtype=dt, device=dev)
            flow_fz = self.pipe.vae.encode([flow_first], device=dev).to(dtype=dt, device=dev)

            flow_z[:, :, 0:1] = flow_fz
            flow_noise = torch.randn_like(flow_z)

            per_flow_input.append(flow_z)
            per_flow_noise.append(flow_noise)

            per_flow_for_action.append(flow_z.detach())
            per_rgb_for_action.append(rgb_z.detach())

            if self.flow_motion_boost > 0:
                with torch.no_grad():
                    deviation = (flow_z - flow_fz).abs().mean(dim=1, keepdim=True)
                    dev_max = deviation.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
                    motion_w = 1.0 + self.flow_motion_boost * (deviation / dev_max)
                per_flow_motion_weight.append(motion_w)

        rgb_input_latents = torch.cat(per_rgb_input, dim=0)
        rgb_noise = torch.cat(per_rgb_noise, dim=0)
        flow_input_latents = torch.cat(per_flow_input, dim=0)
        flow_noise = torch.cat(per_flow_noise, dim=0)
        flow_for_action = torch.cat(per_flow_for_action, dim=0)
        rgb_for_action = torch.cat(per_rgb_for_action, dim=0)
        flow_motion_weight = (
            torch.cat(per_flow_motion_weight, dim=0) if per_flow_motion_weight else None
        )

        return {
            "rgb_input_latents": rgb_input_latents,
            "rgb_noise": rgb_noise,
            "flow_input_latents": flow_input_latents,
            "flow_noise": flow_noise,
            "flow_for_action": flow_for_action,
            "rgb_for_action": rgb_for_action,
            "flow_motion_weight": flow_motion_weight,
            "context": context,
            "action_context": action_context,
            "fuse_vae_embedding_in_latents": True,
            "num_frames": num_frames,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

    @staticmethod
    def _timestep_weights(scheduler, timestep):
        """Per-sample bell-shaped timestep loss weight.

        Uses the FlowMatchScheduler's mean-normalized ``linear_timesteps_weights``
        (built by ``set_timesteps(..., training=True)``), gathered per sample by
        nearest timestep. Returns None if the scheduler has no training weights.
        """
        tw = getattr(scheduler, "linear_timesteps_weights", None)
        if tw is None:
            return None
        ts = scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        tw = tw.to(device=timestep.device, dtype=torch.float32)
        ids = (ts.unsqueeze(0) - timestep.to(torch.float32).unsqueeze(1)).abs().argmin(dim=1)
        return tw[ids]

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        B = inputs["rgb_input_latents"].shape[0]

        # Clean latent snapshot for the action cond branch, taken BEFORE ref-aug
        # mutates the first frame in place. ``rgb_for_action`` is a detached view
        # that shares storage with ``rgb_input_latents``, so ref-aug below would
        # otherwise leak the augmented first frame into the action condition.
        rgb_clean_cond = inputs["rgb_input_latents"].detach().clone()
        flow_clean_cond = inputs["flow_input_latents"].detach().clone()

        max_tb = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_tb = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)

        timestep_id = torch.randint(min_tb, max_tb, (B,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # ---- Reference augmentation of the first (conditioning) frame ----
        if self.ref_aug_strength > 0:
            aug_scale = random.random() * self.ref_aug_strength
            prefixes = (
                ["rgb"]
                if self.video_objective == "track1_conditional_rgb"
                else ["rgb", "flow"]
            )
            for prefix in prefixes:
                clean_pref = inputs[f"{prefix}_input_latents"][:, :, :1].clone()
                aug_pref = clean_pref + aug_scale * torch.randn_like(clean_pref)
                inputs[f"{prefix}_input_latents"][:, :, :1] = aug_pref

        # ======================= Video-loss branch ==========================
        rgb_noisy = self.pipe.scheduler.add_noise(
            inputs["rgb_input_latents"], inputs["rgb_noise"], timestep)
        rgb_noisy[:, :, :1] = inputs["rgb_input_latents"][:, :, :1]
        rgb_target = self.pipe.scheduler.training_target(
            inputs["rgb_input_latents"], inputs["rgb_noise"], timestep)

        conditional_rgb = self.video_objective == "track1_conditional_rgb"
        if conditional_rgb:
            # Exact Track1 inference contract: clean flow is a condition, not a
            # denoising target. RGB frame 0 is clean/t0 and RGB future frames
            # are noisy/t; every flow token is clean/t0.
            flow_model_input = inputs["flow_input_latents"]
            flow_target = None
            flow_timestep_mode = "clean"
        else:
            flow_model_input = self.pipe.scheduler.add_noise(
                inputs["flow_input_latents"], inputs["flow_noise"], timestep)
            flow_model_input[:, :, :1] = inputs["flow_input_latents"][:, :, :1]
            flow_target = self.pipe.scheduler.training_target(
                inputs["flow_input_latents"], inputs["flow_noise"], timestep)
            flow_timestep_mode = "joint"

        rgb_pred, flow_pred = model_fn_wan_video_dual_stream(
            dit=self.pipe.dit,
            flow_stream=self.flow_stream,
            latents=rgb_noisy,
            flow_latents=flow_model_input,
            timestep=timestep,
            context=inputs["context"],
            fuse_vae_embedding_in_latents=inputs.get("fuse_vae_embedding_in_latents", True),
            use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
            flow_timestep_mode=flow_timestep_mode,
            predict_flow=not conditional_rgb,
        )

        rgb_ff_mask = torch.ones(
            1, 1, rgb_pred.shape[2], 1, 1,
            device=rgb_pred.device, dtype=torch.float32)
        rgb_ff_mask[:, :, :1] = 0.0
        # Per-sample video losses (so the per-sample timestep reweighting can be
        # applied). Frame 0 (conditioning) is masked out.
        rgb_se = ((rgb_pred.float() - rgb_target.float()) ** 2) * rgb_ff_mask
        rgb_denom = rgb_ff_mask.expand_as(rgb_se).sum(dim=(1, 2, 3, 4)).clamp(min=1)
        rgb_ps = rgb_se.sum(dim=(1, 2, 3, 4)) / rgb_denom               # (B,)

        if conditional_rgb:
            flow_ps = torch.zeros_like(rgb_ps)
            loss_video_ps = rgb_ps
        else:
            flow_ff_mask = torch.ones(
                1, 1, flow_pred.shape[2], 1, 1,
                device=flow_pred.device, dtype=torch.float32)
            flow_ff_mask[:, :, :1] = 0.0
            flow_se = ((flow_pred.float() - flow_target.float()) ** 2) * flow_ff_mask
            flow_motion_weight = inputs.get("flow_motion_weight")
            if flow_motion_weight is not None:
                flow_se = flow_se * flow_motion_weight
                flow_denom = (flow_motion_weight * flow_ff_mask).expand_as(flow_se).sum(
                    dim=(1, 2, 3, 4)).clamp(min=1)
            else:
                flow_denom = flow_ff_mask.expand_as(flow_se).sum(
                    dim=(1, 2, 3, 4)).clamp(min=1)
            flow_ps = flow_se.sum(dim=(1, 2, 3, 4)) / flow_denom
            w = self.flow_loss_weight
            loss_video_ps = (1.0 - w) * rgb_ps + w * flow_ps
        if self.loss_timestep_weighting:
            vw = self._timestep_weights(self.pipe.scheduler, timestep)
            loss_video = (loss_video_ps * vw).mean() if vw is not None else loss_video_ps.mean()
        else:
            loss_video = loss_video_ps.mean()
        loss_rgb = rgb_ps.mean()
        loss_flow = flow_ps.mean()

        # ======================= Action branch (IDM) ========================
        actions_gt = torch.from_numpy(data["actions"]).to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device)

        action_noise = torch.randn_like(actions_gt)
        act_num_ts = self.action_scheduler.num_train_timesteps
        action_timestep_id = torch.randint(0, act_num_ts, (B,))
        action_timestep = self.action_scheduler.timesteps[action_timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device)

        noisy_actions = self.action_scheduler.add_noise(actions_gt, action_noise, action_timestep)
        # Supervision target:
        #   velocity: noise - actions_gt   (flow-matching velocity)
        #   x0:       actions_gt           (clean action)
        if self.action_pred_target == "velocity":
            action_target = self.action_scheduler.training_target(
                actions_gt, action_noise, action_timestep)
        else:
            action_target = actions_gt
        # GT anchor on action[0] = current qpos (loss-masked below).
        noisy_actions[:, :1] = actions_gt[:, :1]

        # ---- Build the "cond video" for per-layer feature conditioning ----
        # With probability cond_noise_prob, noise the cond video across the full
        # scheduler spectrum and tell the action expert the noise level;
        # otherwise keep it clean (timestep 0). The first frame is kept clean.
        with torch.no_grad():
            rgb_clean = rgb_clean_cond
            flow_clean = flow_clean_cond

            video_cond_timestep = torch.zeros(
                B, dtype=self.pipe.torch_dtype, device=self.pipe.device)
            rgb_cond = rgb_clean
            flow_cond = flow_clean

            if self.cond_noise_prob > 0.0:
                cond_mask = torch.rand(B, device=rgb_clean.device) < self.cond_noise_prob
                if bool(cond_mask.any()):
                    cond_tid = torch.randint(min_tb, max_tb, (B,), device=rgb_clean.device)
                    cond_t = self.pipe.scheduler.timesteps.to(device=rgb_clean.device)[cond_tid]
                    rgb_noised = self.pipe.scheduler.add_noise(
                        rgb_clean, torch.randn_like(rgb_clean), cond_t)
                    sel_rgb = cond_mask.view(B, *([1] * (rgb_clean.dim() - 1)))
                    rgb_cond = torch.where(sel_rgb, rgb_noised, rgb_clean)
                    rgb_cond[:, :, :1] = rgb_clean[:, :, :1]
                    if self.video_objective == "joint_dual_stream":
                        flow_noised = self.pipe.scheduler.add_noise(
                            flow_clean, torch.randn_like(flow_clean), cond_t)
                        sel_flow = cond_mask.view(
                            B, *([1] * (flow_clean.dim() - 1))
                        )
                        flow_cond = torch.where(sel_flow, flow_noised, flow_clean)
                        flow_cond[:, :, :1] = flow_clean[:, :, :1]
                    else:
                        # Track1 IDM auxiliary sees the same clean-flow/t0
                        # condition as the world-model inference path.
                        flow_cond = flow_clean
                    video_cond_timestep = torch.where(
                        cond_mask, cond_t.to(video_cond_timestep.dtype),
                        torch.zeros_like(video_cond_timestep))

        # ---- Capture the video DiT per-layer features ----
        # cond_detach=False: the action loss also trains the video backbone
        # through this second pass (gradient checkpointing keeps it affordable).
        # cond_detach=True: no grad into the video DiT here (video is trained by
        # the video loss only), cheaper.
        cm = torch.no_grad() if self.cond_detach else contextlib.nullcontext()
        with cm:
            feats = capture_video_layer_features(
                dit=self.pipe.dit,
                flow_stream=self.flow_stream,
                latents=rgb_cond,
                flow_latents=flow_cond,
                timestep=video_cond_timestep,
                context=inputs["context"],
                fuse_vae_embedding_in_latents=True,
                use_gradient_checkpointing=(
                    (not self.cond_detach) and self.use_gradient_checkpointing
                ),
                flow_timestep_mode=(
                    "clean"
                    if self.video_objective == "track1_conditional_rgb"
                    else "joint"
                ),
            )
            video_layer_feats = concat_layer_feats(feats, self.cond_layer_stride)
            # concat_layer_feats materializes the Action Expert contexts.  Do
            # not also retain every separate RGB/flow block output for the
            # rest of this forward.  This is especially important when the
            # capture ran under no_grad (cond_detach=True): no autograd graph
            # needs the original list, and keeping both representations costs
            # roughly another full stack of 30 video-layer activations.
            del feats
            if self.cond_detach:
                video_layer_feats = [f.detach() for f in video_layer_feats]

        text_context = inputs.get("action_context", inputs.get("context"))
        joint_state_input = actions_gt[:, 0]

        with torch.amp.autocast("cuda", dtype=self.pipe.torch_dtype):
            action_pred = self.action_expert(
                noisy_actions=noisy_actions,
                timestep=action_timestep,
                video_layer_feats=video_layer_feats,
                text_context=text_context,
                joint_state=joint_state_input,
                cond_timestep=video_cond_timestep,
                return_x0=True,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )

        action_loss_mask = torch.ones_like(action_pred)
        action_loss_mask[:, :1] = 0.0
        act_se = ((action_pred.float() - action_target.float()) ** 2) * action_loss_mask
        act_denom = action_loss_mask.sum(dim=(1, 2)).clamp(min=1)
        action_ps = act_se.sum(dim=(1, 2)) / act_denom                 # (B,)
        if self.loss_timestep_weighting:
            aw = self._timestep_weights(self.action_scheduler, action_timestep)
            loss_action = (action_ps * aw).mean() if aw is not None else action_ps.mean()
        else:
            loss_action = action_ps.mean()

        loss = loss_video + self.action_loss_weight * loss_action

        return {
            "loss": loss,
            "loss_rgb": loss_rgb.detach(),
            "loss_flow": loss_flow.detach(),
            "loss_action": loss_action.detach(),
            "loss_video": loss_video.detach(),
        }


def _build_lr_scheduler(optimizer, scheduler_type, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine (or constant) decay to base_lr * 0.01."""
    scheduler_type = str(scheduler_type).strip().lower()
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_steps - 1)
    remaining = max(total_steps - warmup_steps, 1)

    if scheduler_type == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=base_lr * 0.01)
    elif scheduler_type == "constant":
        main = ConstantLR(optimizer, factor=1.0, total_iters=remaining)
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")

    if warmup_steps <= 0:
        return main
    warmup = LinearLR(
        optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])


def _save_full_state(
    accelerator,
    output_path,
    optimizer_step,
    global_step,
    keep,
    resume_binding,
):
    """Save FULL training state (model + optimizer + scheduler + RNG) via
    accelerator.save_state, plus a trainer_state.json with both counters and
    the immutable workload binding, so training can be resumed exactly
    (optimizer momentum + LR schedule + step), not just warm-started from
    weights. Keeps only the newest ``keep`` dirs.
    """
    import shutil
    import glob as _glob
    state_root = os.path.join(output_path, "state")
    state_dir = os.path.join(state_root, f"step-{optimizer_step}")
    accelerator.save_state(state_dir)
    if accelerator.is_main_process:
        trainer_state = build_trainer_state(
            global_step=global_step,
            optimizer_step=optimizer_step,
            resume_binding=resume_binding,
        )
        with open(
            os.path.join(state_dir, "trainer_state.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(trainer_state, f, indent=2, sort_keys=True)
            f.write("\n")
        if keep > 0:
            dirs = sorted(
                [d for d in _glob.glob(os.path.join(state_root, "step-*")) if os.path.isdir(d)],
                key=lambda p: int(os.path.basename(p).split("step-")[-1]),
            )
            for old in dirs[:-keep]:
                shutil.rmtree(old, ignore_errors=True)
    accelerator.wait_for_everyone()


def _capture_vram_probe(
    accelerator,
    output_path: str,
    optimizer_step: int,
    min_allocated_gib: float,
    min_reserved_gib: float,
    max_vram_gib: float = FORMAL_MAX_VRAM_GIB,
) -> dict:
    """Record per-process CUDA/HCU memory and enforce world-wide gates.

    Both lower bounds deliberately use the *current* allocated/reserved values,
    never a peak or a synthetic ballast allocation. The probe is called after
    the real forward/backward pass and gradient clipping, immediately before
    ``optimizer.step()``. Every rank writes its own atomic JSON record, then all
    ranks gather the values so that a low-memory or incorrectly isolated
    process fails the entire distributed job instead of silently continuing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VRAM probe is fail-closed and torch.cuda.is_available() is false"
        )

    device_index = torch.cuda.current_device()
    torch.cuda.synchronize(device_index)
    gib = float(2 ** 30)
    record = {
        "optimizer_step": int(optimizer_step),
        "rank": int(accelerator.process_index),
        "local_rank": int(accelerator.local_process_index),
        "world_size": int(accelerator.num_processes),
        "device_index": int(device_index),
        "device_name": torch.cuda.get_device_name(device_index),
        "memory_allocated_gib": torch.cuda.memory_allocated(device_index) / gib,
        "memory_reserved_gib": torch.cuda.memory_reserved(device_index) / gib,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device_index) / gib,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device_index) / gib,
        "minimum_required_allocated_gib": float(min_allocated_gib),
        "minimum_required_reserved_gib": float(min_reserved_gib),
        "maximum_allowed_reserved_gib": float(max_vram_gib),
        "measurement_point": "after_backward_before_optimizer_step",
        "synthetic_allocation_used": False,
    }
    allocated_lower_passed = (
        record["memory_allocated_gib"] >= float(min_allocated_gib)
    )
    reserved_lower_passed = (
        record["memory_reserved_gib"] >= float(min_reserved_gib)
    )
    upper_passed = (
        record["memory_reserved_gib"] <= float(max_vram_gib)
        and record["max_memory_reserved_gib"] <= float(max_vram_gib)
    )
    record["allocated_lower_bound_passed"] = allocated_lower_passed
    record["reserved_lower_bound_passed"] = reserved_lower_passed
    record["lower_bound_passed"] = (
        allocated_lower_passed and reserved_lower_passed
    )
    record["upper_bound_passed"] = upper_passed
    record["passed"] = record["lower_bound_passed"] and upper_passed

    probe_dir = os.path.join(
        output_path, "vram_probe", f"optimizer_step_{optimizer_step:08d}"
    )
    os.makedirs(probe_dir, exist_ok=True)
    destination = os.path.join(
        probe_dir, f"rank_{accelerator.process_index:05d}.json"
    )
    temporary = f"{destination}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    print(f"[VRAM_PROBE] {json.dumps(record, sort_keys=True)}", flush=True)

    local_memory = torch.tensor(
        [
            record["memory_allocated_gib"],
            record["memory_reserved_gib"],
            record["max_memory_allocated_gib"],
            record["max_memory_reserved_gib"],
        ],
        dtype=torch.float32,
        device=accelerator.device,
    )
    gathered = accelerator.gather(local_memory).reshape(-1, 4).float().cpu()
    allocated_by_rank = gathered[:, 0].tolist()
    reserved_by_rank = gathered[:, 1].tolist()
    max_allocated_by_rank = gathered[:, 2].tolist()
    max_reserved_by_rank = gathered[:, 3].tolist()
    failing_ranks = [
        rank
        for rank, (allocated, reserved, max_reserved) in enumerate(
            zip(allocated_by_rank, reserved_by_rank, max_reserved_by_rank)
        )
        if (
            allocated < float(min_allocated_gib)
            or reserved < float(min_reserved_gib)
            or reserved > float(max_vram_gib)
            or max_reserved > float(max_vram_gib)
        )
    ]

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        summary = {
            "optimizer_step": int(optimizer_step),
            "world_size": int(accelerator.num_processes),
            "minimum_required_allocated_gib": float(min_allocated_gib),
            "minimum_required_reserved_gib": float(min_reserved_gib),
            "maximum_allowed_reserved_gib": float(max_vram_gib),
            "measurement_point": "after_backward_before_optimizer_step",
            "synthetic_allocation_used": False,
            "ranks": [
                {
                    "rank": rank,
                    "memory_allocated_gib": allocated_by_rank[rank],
                    "memory_reserved_gib": reserved_by_rank[rank],
                    "max_memory_allocated_gib": max_allocated_by_rank[rank],
                    "max_memory_reserved_gib": max_reserved_by_rank[rank],
                    "passed": rank not in failing_ranks,
                }
                for rank in range(len(reserved_by_rank))
            ],
            "failing_ranks": failing_ranks,
            "passed": not failing_ranks,
        }
        summary_path = os.path.join(probe_dir, "summary.json")
        summary_tmp = f"{summary_path}.tmp.{os.getpid()}"
        with open(summary_tmp, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(summary_tmp, summary_path)
        print(f"[VRAM_PROBE_SUMMARY] {json.dumps(summary, sort_keys=True)}", flush=True)
    accelerator.wait_for_everyone()

    if failing_ranks:
        raise RuntimeError(
            "VRAM fail-closed gate rejected ranks "
            f"{failing_ranks}: current memory_allocated must be >= "
            f"{float(min_allocated_gib):.3f} GiB and current memory_reserved "
            f"must be >= {float(min_reserved_gib):.3f} GiB on every process"
            + f" and current/peak memory_reserved must be <= "
            f"{float(max_vram_gib):.3f} GiB"
            + f"; allocated={allocated_by_rank}; reserved={reserved_by_rank}; "
            f"peak_reserved={max_reserved_by_rank}"
        )
    return record


def launch_training_task(dataset, model, model_logger, start_epoch=0, args=None):
    """Training loop with AdamW, warmup/cosine LR, and grad clipping.

    ``zro_adamw`` keeps ordinary DDP model/gradient replication while sharding
    only AdamW's persistent optimizer state across the data-parallel ranks.
    ``hybrid_adamw`` runs identical AdamW updates while replicating a calibrated
    fraction of optimizer state and sharding the remainder. Neither mode
    changes the model, batch, loss, or per-rank forward/backward workload.
    ``foreach=False`` also avoids AdamW's transient foreach buffer on the
    memory-constrained HCUs.
    """
    import swanlab
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    lr = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    grad_accum = args.gradient_accumulation_steps
    find_unused = args.find_unused_parameters
    save_every_n_epochs = getattr(args, "save_every_n_epochs", 1)
    vram_probe_step = int(getattr(args, "vram_probe_step", 3))
    min_allocated_gib = float(
        getattr(args, "min_allocated_gib", FORMAL_MIN_VRAM_GIB)
    )
    min_reserved_gib = float(
        getattr(args, "min_vram_gib", FORMAL_MIN_VRAM_GIB)
    )
    max_vram_gib = float(
        getattr(args, "max_vram_gib", FORMAL_MAX_VRAM_GIB)
    )
    max_optimizer_steps = int(getattr(args, "max_optimizer_steps", 0))
    stop_after_vram_probe = bool(getattr(args, "stop_after_vram_probe", False))
    if vram_probe_step <= 0:
        raise ValueError("vram_probe_step must be a positive optimizer step")
    if not all(math.isfinite(value) for value in (
        min_allocated_gib, min_reserved_gib, max_vram_gib,
    )):
        raise ValueError("VRAM gates must be finite numbers")
    if min_allocated_gib < FORMAL_MIN_VRAM_GIB:
        raise ValueError("min_allocated_gib is a hard gate and must be >= 54 GiB")
    if min_reserved_gib < FORMAL_MIN_VRAM_GIB:
        raise ValueError("min_vram_gib is a hard gate and must be >= 54 GiB")
    if not min_reserved_gib <= max_vram_gib <= FORMAL_MAX_VRAM_GIB:
        raise ValueError(
            "max_vram_gib is a hard safety ceiling and must satisfy "
            "min_vram_gib <= max_vram_gib <= 63 GiB"
        )
    if max_optimizer_steps < 0:
        raise ValueError("max_optimizer_steps must be non-negative")

    # Accelerator initializes the torch.distributed process group. Native
    # ZeroRedundancyOptimizer must therefore be constructed after this point.
    # step_scheduler_with_optimizer=False + manual scheduler.step() on
    # sync_gradients keeps the LR schedule correct under gradient accumulation.
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=find_unused,
            # Reuse DDP communication buckets as gradient storage after the
            # first iteration; this avoids a second gradient-sized allocation.
            gradient_as_bucket_view=True,
        )],
    )

    optimizer_type = str(getattr(args, "optimizer_type", "adamw")).strip().lower()
    optimizer_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.95),
        "foreach": False,
    }
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.trainable_modules(), **optimizer_kwargs)
    elif optimizer_type in {"zro_adamw", "hybrid_adamw"}:
        if int(getattr(args, "full_state_keep", 2)) != 0:
            raise ValueError(
                f"{optimizer_type} requires full_state_keep=0; use the periodic "
                "safetensors weight checkpoints for this run"
            )
        if getattr(args, "resume_state_dir", None):
            raise ValueError(
                f"{optimizer_type} does not support accelerator full-state resume; "
                "resume from a safetensors weight checkpoint instead"
            )
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                f"{optimizer_type} requires an initialized distributed process group"
            )
        if torch.distributed.get_world_size() < 2:
            raise RuntimeError(f"{optimizer_type} requires at least two distributed ranks")
        if optimizer_type == "zro_adamw":
            optimizer = DtypeShardedAdamW(
                model.trainable_modules(), **optimizer_kwargs
            )
        else:
            optimizer = DtypeHybridAdamW(
                model.trainable_modules(),
                replicated_fraction=float(args.hybrid_optimizer_replicated_fraction),
                **optimizer_kwargs,
            )
    else:
        raise ValueError(f"unsupported optimizer_type: {optimizer_type}")

    cache_episode_batching = bool(
        args.load_from_cache and getattr(args, "cache_episode_batching", False)
    )
    online_epoch_sampler = None
    if cache_episode_batching:
        batch_sampler = EpisodeAwareCacheBatchSampler(
            dataset, batch_size=args.batch_size,
            chunks_per_episode=args.cache_chunks_per_episode,
            drop_last=args.cache_episode_drop_last,
            seed=args.cache_episode_batch_seed,
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=dual_stream_action_collate_fn, num_workers=num_workers,
        )
    else:
        online_epoch_sampler = AbsoluteEpochRandomSampler(
            dataset, seed=args.data_shuffle_seed,
        )
        dataloader = DataLoader(
            dataset, sampler=online_epoch_sampler, shuffle=False,
            batch_size=args.batch_size,
            collate_fn=dual_stream_action_collate_fn, num_workers=num_workers,
        )

    # The dataloader here is the GLOBAL batch stream, so per-process optimizer
    # steps/epoch = ceil(len(dataloader) / (num_processes * grad_accum)). The
    # cosine horizon spans num_epochs of these steps unless --lr_max_steps
    # overrides it.
    num_processes = max(int(accelerator.num_processes), 1)
    opt_steps_per_epoch = max(
        math.ceil(len(dataloader) / (num_processes * max(grad_accum, 1))), 1)
    if getattr(args, "lr_max_steps", 0):
        total_steps = int(args.lr_max_steps)
    else:
        total_steps = max(opt_steps_per_epoch * num_epochs, 1)
    # Absolute --lr_warmup_steps (>0) takes precedence over the ratio.
    if getattr(args, "lr_warmup_steps", 0) and int(args.lr_warmup_steps) > 0:
        warmup_steps = int(args.lr_warmup_steps)
    else:
        warmup_steps = int(total_steps * getattr(args, "lr_warmup_ratio", 0.05))
    scheduler = _build_lr_scheduler(
        optimizer, getattr(args, "lr_scheduler_type", "cosine"),
        total_steps, warmup_steps, lr,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    current_resume_binding = build_resume_binding(
        world_size=accelerator.num_processes,
        per_device_batch=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        dataset_num_workers=num_workers,
        data_shuffle_seed=args.data_shuffle_seed,
        training_manifest_sha256=args.training_manifest_sha256,
        video_objective=args.video_objective,
        flow_mode=args.flow_mode,
        size=args.size,
        num_frames=args.num_frames,
        num_video_frames=args.num_video_frames,
        visual_stride=args.visual_stride,
    )

    # ---- Exact resume (optimizer + scheduler + step), if --resume_state_dir ----
    from accelerate import skip_first_batches
    full_state_keep = int(getattr(args, "full_state_keep", 2))
    resume_state_dir = getattr(args, "resume_state_dir", None)
    skip_batches = 0
    restored_optimizer_step = None
    if resume_state_dir:
        ts_path = os.path.join(resume_state_dir, "trainer_state.json")
        if not os.path.isfile(ts_path):
            raise RuntimeError(
                "exact resume requires trainer_state.json, but it is missing: "
                f"{ts_path}"
            )
        with open(ts_path, encoding="utf-8") as f:
            trainer_state = json.load(f)
        try:
            global_step, restored_optimizer_step = validate_trainer_state(
                trainer_state, current_resume_binding
            )
        except ResumeBindingError as error:
            raise RuntimeError(str(error)) from error

        # Loading mutates model/optimizer/scheduler/RNG state.  Do it only
        # after the immutable workload binding above has matched exactly.
        accelerator.load_state(resume_state_dir)

        # Epoch and dataloader position are meaningful only for the workload
        # whose binding was validated before load_state.
        start_epoch = global_step // max(len(dataloader), 1)
        skip_batches = global_step % max(len(dataloader), 1)
        if accelerator.is_main_process:
            print(
                f"[Resume-STATE] loaded {resume_state_dir}: "
                f"global_step={global_step}, "
                f"optimizer_step={restored_optimizer_step}, "
                f"start_epoch={start_epoch}, skip_batches={skip_batches} "
                f"(optimizer + LR scheduler restored; NO re-warmup)"
            )
    else:
        global_step = start_epoch * len(dataloader)
    optimizer_step = (
        restored_optimizer_step
        if restored_optimizer_step is not None
        else start_epoch * opt_steps_per_epoch
    )
    model_logger.num_steps = optimizer_step
    runtime_vram_probe_step = resolve_vram_probe_optimizer_step(
        configured_probe_step=vram_probe_step,
        restored_optimizer_step=restored_optimizer_step,
    )
    if optimizer_step >= runtime_vram_probe_step:
        raise RuntimeError(
            f"mandatory VRAM probe step {runtime_vram_probe_step} has already "
            f"passed (current optimizer step is approximately {optimizer_step}); "
            "an exact resume state is required to schedule a new in-process gate"
        )
    if max_optimizer_steps and runtime_vram_probe_step > max_optimizer_steps:
        raise RuntimeError(
            f"mandatory in-process VRAM probe step {runtime_vram_probe_step} is "
            f"later than max_optimizer_steps={max_optimizer_steps}"
        )
    if accelerator.is_main_process:
        if restored_optimizer_step is None:
            schedule_description = (
                f"fresh absolute optimizer step {runtime_vram_probe_step}"
            )
        else:
            schedule_description = (
                f"restored step {restored_optimizer_step} + calibrated "
                f"{vram_probe_step}-step delay = {runtime_vram_probe_step}"
            )
        print(
            "[VRAM] mandatory real-memory gate scheduled at "
            f"{schedule_description}",
            flush=True,
        )
    vram_probe_complete = False
    stop_training = False
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VRAM probe is mandatory but torch.cuda.is_available() is false"
        )
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    # A host may release its per-HCU launch locks only after the actual trainer
    # is known to be viable.  This global/local barrier pair runs after
    # accelerator.prepare and exact-resume validation/load, so an import,
    # model-load, DDP-rendezvous, or resume failure cannot publish readiness.
    publish_trainer_ready(accelerator)

    if accelerator.is_main_process:
        training_config = {
            "learning_rate": lr, "weight_decay": weight_decay,
            "betas": [0.9, 0.95], "lr_scheduler_type": getattr(args, "lr_scheduler_type", "cosine"),
            "lr_warmup_ratio": getattr(args, "lr_warmup_ratio", 0.05),
            "lr_total_steps": total_steps, "lr_warmup_steps": warmup_steps,
            "num_epochs": num_epochs, "start_epoch": start_epoch,
            "gradient_accumulation_steps": grad_accum,
            "num_frames": args.num_frames, "batch_size": args.batch_size,
            "flow_loss_weight": args.flow_loss_weight,
            "video_objective": args.video_objective,
            "flow_supervised": args.video_objective == "joint_dual_stream",
            "action_loss_weight": args.action_loss_weight,
            "action_snr_shift": args.action_snr_shift,
            "cond_noise_prob": args.cond_noise_prob, "cond_detach": args.cond_detach,
            "cond_layer_stride": args.cond_layer_stride,
            "action_pred_target": args.action_pred_target,
            "action_pos_mode": args.action_pos_mode, "proprio_mode": args.proprio_mode,
            "loss_timestep_weighting": args.loss_timestep_weighting,
            "num_action_layers": args.num_action_layers,
            "dataset_base_path": args.dataset_base_path,
            "training_manifest": args.training_manifest,
            "training_manifest_sha256": args.training_manifest_sha256,
            "variants": args.variants,
            "cameras": args.cameras,
            "camera_prefix": args.camera_prefix,
            "track1_prompt_template": args.track1_prompt_template,
            "flow_mode": args.flow_mode,
            "flow_method": args.flow_method,
            "size": args.size,
            "tokenizer_model_id": args.tokenizer_model_id,
            "resume_checkpoint": args.resume_checkpoint, "output_path": args.output_path,
            "load_from_cache": args.load_from_cache, "cache_root": args.cache_root,
            "data_shuffle_seed": args.data_shuffle_seed,
            "save_steps": save_steps, "mode": "dual_stream_flow_action_idm",
            "vram_probe_step": vram_probe_step,
            "runtime_vram_probe_step": runtime_vram_probe_step,
            "vram_probe_is_exact_resume": restored_optimizer_step is not None,
            "min_allocated_gib": min_allocated_gib,
            "min_vram_gib": min_reserved_gib,
            "max_vram_gib": max_vram_gib,
            "max_optimizer_steps": max_optimizer_steps,
            "stop_after_vram_probe": stop_after_vram_probe,
            "resume_binding": current_resume_binding,
        }
        os.makedirs(model_logger.output_path, exist_ok=True)
        # Continue the SAME SwanLab run across resumes so the loss curve is one
        # uninterrupted line. swanlab.log() below passes an explicit
        # step=optimizer_step, so accumulation and resume share one x-axis.
        # id priority: env SWANLAB_RESUME_ID > persisted id file (only for a
        # true resume via resume_state_dir). resume="allow" attaches if the run
        # exists, else creates one with that id.
        _sw_init = dict(project="kineworld-robotwin-idm", config=training_config)
        _rid_file = os.path.join(model_logger.output_path, "swanlab_run_id.txt")
        _sw_resume_id = os.environ.get("SWANLAB_RESUME_ID", "").strip()
        if not _sw_resume_id and resume_state_dir and os.path.exists(_rid_file):
            with open(_rid_file) as _f:
                _sw_resume_id = _f.read().strip()
        if _sw_resume_id:
            _sw_init.update(id=_sw_resume_id, resume="allow")
            print(f"[SwanLab] resuming run id={_sw_resume_id} "
                  f"(append to the same curve from optimizer step {optimizer_step})")
        _sw_run = swanlab.init(**_sw_init)
        try:
            _rid = getattr(_sw_run, "id", None) or _sw_resume_id
            if _rid:
                with open(_rid_file, "w") as _f:
                    _f.write(str(_rid))
        except Exception as _e:
            print(f"[SwanLab] could not persist run id: {_e}")
        with open(os.path.join(model_logger.output_path, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        print(f"[Train] batches/epoch={len(dataloader)}, lr_total_steps={total_steps}, "
              f"warmup={warmup_steps}, scheduler={getattr(args, 'lr_scheduler_type', 'cosine')}")

    for epoch_id in range(start_epoch, num_epochs):
        # Re-seed the per-epoch shuffle to the ABSOLUTE epoch index so a resumed
        # run reproduces the same batch order this epoch used originally
        # (EpisodeAwareCacheBatchSampler keys its RNG off seed+epoch).
        if cache_episode_batching:
            batch_sampler.set_epoch(epoch_id)
        elif online_epoch_sampler is not None:
            online_epoch_sampler.set_epoch(epoch_id)
        active_loader = dataloader
        if epoch_id == start_epoch and skip_batches > 0:
            active_loader = skip_first_batches(dataloader, skip_batches)
        for data in tqdm(active_loader, desc=f"Epoch {epoch_id}"):
            with accelerator.accumulate(model):
                loss_dict = model(data)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Stable real-workload sample: the actual batch has completed
                # forward/backward and gradients are present, while the
                # optimizer has not yet changed or released its state.
                if (
                    accelerator.sync_gradients
                    and optimizer_step + 1 == runtime_vram_probe_step
                ):
                    _capture_vram_probe(
                        accelerator=accelerator,
                        output_path=model_logger.output_path,
                        optimizer_step=optimizer_step + 1,
                        min_allocated_gib=min_allocated_gib,
                        min_reserved_gib=min_reserved_gib,
                        max_vram_gib=max_vram_gib,
                    )
                    vram_probe_complete = True
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    optimizer_step += 1
                    model_logger.on_step_end(accelerator, model, save_steps)
                    if model_logger.num_steps != optimizer_step:
                        raise RuntimeError(
                            "optimizer/model logger step counters diverged: "
                            f"{optimizer_step} != {model_logger.num_steps}"
                        )
                    # Save full optimizer/scheduler/RNG state only at a real
                    # optimizer boundary. global_step is a microbatch counter.
                    if (
                        full_state_keep > 0
                        and save_steps
                        and optimizer_step % save_steps == 0
                    ):
                        _save_full_state(
                            accelerator,
                            model_logger.output_path,
                            optimizer_step=optimizer_step,
                            global_step=global_step + 1,
                            keep=full_state_keep,
                            resume_binding=current_resume_binding,
                        )
                    if (
                        optimizer_step == runtime_vram_probe_step
                        and stop_after_vram_probe
                    ):
                        stop_training = True
                    if accelerator.is_main_process:
                        swanlab.log({
                            "loss": loss.item(),
                            "loss_rgb": loss_dict["loss_rgb"].item(),
                            "loss_flow": loss_dict["loss_flow"].item(),
                            "loss_action": loss_dict["loss_action"].item(),
                            "loss_video": loss_dict["loss_video"].item(),
                            "epoch": epoch_id,
                            "global_step": global_step + 1,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }, step=optimizer_step)
                # AcceleratedOptimizer keeps accumulated gradients when this is
                # a non-sync microbatch and clears them only after a real step.
                optimizer.zero_grad()
                global_step += 1
                if max_optimizer_steps and optimizer_step >= max_optimizer_steps:
                    stop_training = True
            if stop_training:
                break

        if stop_training:
            break

        if save_steps is None:
            is_first = (epoch_id == start_epoch)
            is_last = (epoch_id == num_epochs - 1)
            if (epoch_id + 1) % save_every_n_epochs == 0 or is_first or is_last:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

    if not vram_probe_complete:
        raise RuntimeError(
            "training ended before mandatory in-process VRAM probe optimizer "
            f"step {runtime_vram_probe_step}"
        )
    if not stop_after_vram_probe:
        model_logger.on_training_end(accelerator, model, save_steps)
    elif accelerator.is_main_process:
        print(
            "[Canary] VRAM probe passed; stopping without writing a full model checkpoint",
            flush=True,
        )
    if accelerator.is_main_process:
        swanlab.finish()


def parse_start_epoch(resume_checkpoint):
    if resume_checkpoint is None:
        return 0
    basename = os.path.basename(resume_checkpoint)
    import re
    m = re.match(r"epoch-(\d+)", basename)
    if m:
        return int(m.group(1)) + 1
    return 0


def _build_parser():
    parser = wan_parser()
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true")
    parser.add_argument("--task_names", type=str, nargs="*", default=None)
    parser.add_argument("--size", type=int, nargs=2, default=[320, 256], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num_video_frames", type=int, default=None)
    parser.add_argument("--visual_stride", type=int, default=1)
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["aloha-agilex_clean_50"])
    parser.add_argument(
        "--training_manifest",
        type=str,
        required=True,
        help="Audited JSONL episode allowlist; extracted trees are never globbed.",
    )
    parser.add_argument(
        "--training_manifest_sha256",
        type=str,
        default=TRACK1_FINAL_TRAIN_MANIFEST_SHA256,
        help="Expected SHA-256 of the audited final-train manifest.",
    )
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["head_camera", "left_camera", "right_camera"])
    parser.add_argument(
        "--camera_prefix", type=str, default=None,
        help=("Prompt prefix describing the camera layout. Pass an empty string "
              "for Track 1 head-only training; the dataset default describes a "
              "T-shape head+wrist composite."),
    )
    prompt_template_group = parser.add_mutually_exclusive_group()
    prompt_template_group.add_argument(
        "--track1_prompt_template", dest="track1_prompt_template",
        action="store_true",
        help=("Normalize video/action prompts to the official Track1 head-only prefix; "
              "already-prefixed instructions are left unchanged."),
    )
    prompt_template_group.add_argument(
        "--no_track1_prompt_template", dest="track1_prompt_template",
        action="store_false",
        help="Disable the Track1 prompt template for an explicit ablation.",
    )
    parser.set_defaults(track1_prompt_template=True)
    parser.add_argument("--flow_method", type=str, default="raft", choices=["raft", "farneback"])
    parser.add_argument("--flow_device", type=str, default="cuda")
    parser.add_argument("--flow_max_magnitude", type=float, default=None)
    parser.add_argument("--flow_mode", type=str, default="robot_only",
                        choices=["full_scene", "robot_only"])
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument(
        "--tokenizer_model_id", type=str,
        default=os.environ.get("KINEWORLD_TOKENIZER_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B"),
        help="Local model ID under ./models containing google/umt5-xxl.",
    )
    parser.add_argument(
        "--video_objective",
        choices=["track1_conditional_rgb", "joint_dual_stream"],
        default="track1_conditional_rgb",
        help=("Track1 default uses clean flow/t0 to condition noisy RGB future "
              "tokens; joint_dual_stream is the legacy explicit A/B objective."),
    )
    parser.add_argument("--flow_loss_weight", type=float, default=0.0)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--num_action_layers", type=int, default=30,
                        help="Action expert depth. 1:1 with the video DiT "
                             "(Wan2.2-5B = 30 layers) by default.")
    parser.add_argument("--action_norm_path", type=str, default=None)
    parser.add_argument("--action_expert_checkpoint", type=str, default=None)
    parser.add_argument("--action_snr_shift", type=float, default=1.0)
    parser.add_argument("--flow_motion_boost", type=float, default=0.0)
    parser.add_argument("--text_context_dim", type=int, default=0)
    parser.add_argument("--ref_aug_strength", type=float, default=0.0)
    parser.add_argument("--fp32_modulation", action="store_true", default=False)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--load_from_cache", action="store_true", default=False)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--cache_episode_batching", action="store_true", default=False)
    parser.add_argument("--cache_chunks_per_episode", type=int, default=8)
    parser.add_argument("--cache_episode_lru_size", type=int, default=0)
    parser.add_argument("--cache_episode_drop_last", action="store_true", default=False)
    parser.add_argument("--cache_episode_batch_seed", type=int, default=42)
    parser.add_argument(
        "--data_shuffle_seed", type=int, default=42,
        help="Base seed for absolute-epoch online-data permutations.",
    )

    # ---- IDM conditioning ----
    parser.add_argument("--cond_noise_prob", type=float, default=0.5,
                        help="Fraction of steps where the cond video is "
                             "full-spectrum noised (else kept clean).")
    parser.add_argument("--cond_detach", action="store_true", default=False,
                        help="Detach the cond-video feature capture (no grad "
                             "into the video DiT from the action loss). The "
                             "video DiT is still trained by the video loss. "
                             "Turn ON to halve memory/compute.")
    parser.add_argument("--cond_layer_stride", type=int, default=1,
                        help="Subsample captured video layers (memory knob). "
                             "1 keeps every layer.")
    parser.add_argument("--action_expert_dim", type=int, default=1024)
    parser.add_argument("--action_expert_heads", type=int, default=16)
    parser.add_argument("--action_expert_ffn_dim", type=int, default=4096)

    # ---- action-head options ----
    parser.add_argument("--action_pred_target", type=str, default="velocity",
                        choices=["velocity", "x0"],
                        help="velocity=head predicts (noise-x0); "
                             "x0=head predicts the clean action.")
    parser.add_argument("--action_pos_mode", type=str, default="rope",
                        choices=["rope", "learned"],
                        help="rope=1D RoPE on action self-attn; "
                             "learned=additive learnable pos_embed.")
    parser.add_argument("--proprio_mode", type=str, default="text",
                        choices=["text", "state_token"],
                        help="text=append proprio token to text context; "
                             "state_token=prepend to the action sequence.")
    parser.add_argument("--loss_timestep_weighting", type=str, default="on",
                        choices=["on", "off"],
                        help="on=bell-shaped per-sample timestep loss weighting "
                             "on BOTH video & action losses; off=plain mean MSE.")

    # ---- LR schedule ----
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "constant"],
                        help="cosine+warmup or constant.")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.05,
                        help="Linear warmup fraction of the horizon. "
                             "Ignored if --lr_warmup_steps > 0.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Absolute warmup steps (overrides --lr_warmup_ratio "
                             "when > 0).")
    parser.add_argument("--lr_max_steps", type=int, default=0,
                        help="Cosine horizon in OPTIMIZER steps. 0 => derive "
                             "from num_epochs * batches/epoch.")
    parser.add_argument(
        "--optimizer_type", type=str, default="adamw",
        choices=["adamw", "zro_adamw", "hybrid_adamw"],
        help=("adamw replicates optimizer state on every DDP rank; zro_adamw "
              "shards only optimizer state while preserving ordinary DDP "
              "model, gradient, batch, and loss semantics; hybrid_adamw "
              "replicates a calibrated fraction and shards the rest."),
    )
    parser.add_argument(
        "--hybrid_optimizer_replicated_fraction",
        type=float,
        default=0.70,
        help=("Fraction of trainable parameter AdamW states replicated by "
              "hybrid_adamw; the remainder is dtype-safe ZRO-sharded."),
    )

    # ---- Exact resume (optimizer + scheduler + step) ----
    parser.add_argument("--resume_state_dir", type=str, default=None,
                        help="accelerator.load_state dir (output_path/state/step-N) "
                             "for EXACT resume: restores optimizer momentum + LR "
                             "scheduler + global step (no LR re-warmup). Different "
                             "from --resume_checkpoint (weights-only warm-start).")
    parser.add_argument("--full_state_keep", type=int, default=2,
                        help="How many recent full-state dirs to keep (0=disable "
                             "full-state saving; only weights are saved then).")
    parser.add_argument(
        "--vram_probe_step", type=int, default=3,
        help=("Calibrated 1-indexed optimizer check step. Fresh runs probe at "
              "this absolute step; exact resumes probe this many optimizer "
              "steps after the restored step."),
    )
    parser.add_argument(
        "--min_allocated_gib", type=float, default=FORMAL_MIN_VRAM_GIB,
        help="Fail unless current torch.cuda.memory_allocated is at least this many GiB on every rank.",
    )
    parser.add_argument(
        "--min_vram_gib", type=float, default=FORMAL_MIN_VRAM_GIB,
        help="Fail unless current torch.cuda.memory_reserved is at least this many GiB on every rank.",
    )
    parser.add_argument(
        "--max_vram_gib", type=float, default=FORMAL_MAX_VRAM_GIB,
        help=("Hard safety ceiling for current and peak memory_reserved on "
              "every rank; must not exceed 63 GiB."),
    )
    parser.add_argument(
        "--max_optimizer_steps", type=int, default=0,
        help="Optional optimizer-step cap (0 means no cap); useful for bounded canaries.",
    )
    parser.add_argument(
        "--stop_after_vram_probe", action="store_true", default=False,
        help="Stop after a passing VRAM probe without saving a full model checkpoint.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    if args.action_norm_path is not None:
        sentinel = args.action_norm_path + ".ready"
        error_sentinel = args.action_norm_path + ".error.json"
        norm_timeout_s = int(
            os.environ.get("ACTION_NORM_WAIT_TIMEOUT_SECONDS", "7200")
        )
        if norm_timeout_s <= 0:
            raise ValueError("ACTION_NORM_WAIT_TIMEOUT_SECONDS must be positive")
        identity = {
            "version": 1,
            "training_manifest_sha256": args.training_manifest_sha256,
            "variants": sorted(args.variants),
            "task_names": sorted(args.task_names or []),
        }

        def _atomic_json(path, payload):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            temporary = f"{path}.tmp.{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, path)

        def _sha256(path):
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        if not os.path.exists(sentinel):
            if rank == 0:
                try:
                    from dataset_action_robotwin import compute_global_action_norm_stats
                    norm_stats = compute_global_action_norm_stats(
                        args.dataset_base_path,
                        args.variants,
                        args.task_names,
                        training_manifest_path=args.training_manifest,
                        training_manifest_sha256=args.training_manifest_sha256,
                    )
                    os.makedirs(
                        os.path.dirname(args.action_norm_path) or ".", exist_ok=True
                    )
                    temporary_norm = (
                        f"{args.action_norm_path}.tmp.{os.getpid()}.npz"
                    )
                    norm_stats.save(temporary_norm)
                    os.replace(temporary_norm, args.action_norm_path)
                    ready_payload = dict(identity)
                    ready_payload["action_norm_sha256"] = _sha256(
                        args.action_norm_path
                    )
                    _atomic_json(sentinel, ready_payload)
                    print(
                        f"[Rank 0] Action norm stats saved to "
                        f"{args.action_norm_path} for manifest "
                        f"{args.training_manifest_sha256}"
                    )
                except Exception as error:
                    _atomic_json(
                        error_sentinel,
                        {
                            **identity,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                    raise
            else:
                import time
                started = time.monotonic()
                while not os.path.exists(sentinel):
                    if os.path.exists(error_sentinel):
                        with open(error_sentinel, "r", encoding="utf-8") as stream:
                            failure = json.load(stream)
                        raise RuntimeError(
                            "rank 0 action-norm computation failed: "
                            f"{failure.get('error_type')}: {failure.get('error')}"
                        )
                    if time.monotonic() - started > norm_timeout_s:
                        raise TimeoutError(
                            f"timed out after {norm_timeout_s}s waiting for "
                            f"action-norm sentinel {sentinel}"
                        )
                    time.sleep(1)

        with open(sentinel, "r", encoding="utf-8") as stream:
            ready = json.load(stream)
        for key, expected in identity.items():
            if ready.get(key) != expected:
                raise ValueError(
                    f"action norm sentinel identity mismatch for {key}: "
                    f"{ready.get(key)!r} != {expected!r}"
                )
        if not os.path.isfile(args.action_norm_path):
            raise FileNotFoundError(args.action_norm_path)
        actual_norm_sha256 = _sha256(args.action_norm_path)
        if ready.get("action_norm_sha256") != actual_norm_sha256:
            raise ValueError(
                "action norm file SHA-256 does not match its ready sentinel: "
                f"{actual_norm_sha256} != {ready.get('action_norm_sha256')}"
            )

    dataset = RoboTwinActionFlowDataset(
        data_root=args.dataset_base_path,
        variants=args.variants,
        cameras=args.cameras,
        size=tuple(args.size),
        num_frames=args.num_frames,
        num_video_frames=args.num_video_frames,
        visual_stride=args.visual_stride,
        task_names=args.task_names,
        camera_prefix=args.camera_prefix,
        track1_prompt_template=args.track1_prompt_template,
        flow_method=args.flow_method,
        flow_device=args.flow_device,
        flow_max_magnitude=args.flow_max_magnitude,
        flow_mode=args.flow_mode,
        action_norm_path=args.action_norm_path,
        training_manifest_path=args.training_manifest,
        training_manifest_sha256=args.training_manifest_sha256,
        load_from_cache=args.load_from_cache,
        cache_root=args.cache_root,
        cache_episode_lru_size=args.cache_episode_lru_size,
    )

    model = FlowActionTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        tokenizer_model_id=args.tokenizer_model_id,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        resume_checkpoint=args.resume_checkpoint,
        video_objective=args.video_objective,
        flow_loss_weight=args.flow_loss_weight,
        action_loss_weight=args.action_loss_weight,
        action_dim=args.action_dim,
        num_action_layers=args.num_action_layers,
        num_frames=args.num_frames,
        cameras=args.cameras,
        action_expert_checkpoint=args.action_expert_checkpoint,
        action_snr_shift=args.action_snr_shift,
        flow_motion_boost=args.flow_motion_boost,
        text_context_dim=args.text_context_dim,
        ref_aug_strength=args.ref_aug_strength,
        fp32_modulation=args.fp32_modulation,
        cond_noise_prob=args.cond_noise_prob,
        cond_detach=args.cond_detach,
        cond_layer_stride=args.cond_layer_stride,
        action_expert_dim=args.action_expert_dim,
        action_expert_heads=args.action_expert_heads,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
        action_pred_target=args.action_pred_target,
        action_use_rope=(args.action_pos_mode == "rope"),
        proprio_mode=args.proprio_mode,
        loss_timestep_weighting=(args.loss_timestep_weighting == "on"),
    )
    start_epoch = parse_start_epoch(args.resume_checkpoint)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launch_training_task(
        dataset, model, model_logger, start_epoch=start_epoch, args=args
    )
