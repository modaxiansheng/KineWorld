#!/usr/bin/env python3
"""KineWorld Stage-1 world-model loader used by the Track-1 adapter.

This module deliberately keeps heavyweight imports inside runtime functions so
the command-line adapter can expose ``--help`` on machines without PyTorch.
The Stage-1 sampling path uses this contract: flow is a clean
condition and is never denoised; only RGB is denoised. Track-1 action-flow can
be supplied per chunk, while all-white zero flow is an explicit baseline only.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


LOGGER = logging.getLogger("kineworld.track1.model")
KINEWORLD_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    checkpoint_path: Path | None,
    checkpoint_repo: str,
    checkpoint_file: str,
    checkpoint_revision: str | None,
    cache_dir: Path,
) -> tuple[Path, str]:
    """Resolve a local checkpoint or download one from Hugging Face."""
    if checkpoint_path is not None:
        resolved = checkpoint_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved, "local"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
                "huggingface_hub is required to download the requested "
                "checkpoint; install KineWorld requirements or pass --checkpoint-path"
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        hf_hub_download(
            repo_id=checkpoint_repo,
            filename=checkpoint_file,
            revision=checkpoint_revision,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    return resolved, f"hf://{checkpoint_repo}/{checkpoint_file}"


@dataclass
class PreparedEpisode:
    """GPU-resident conditioning reused by all chunks of one episode."""

    context: Any
    flow_clean: Any
    width: int
    height: int
    num_frames: int
    instruction: str
    conditioning_mode: str


class Stage1WorldModel:
    """Wan2.2 + KineWorld Stage-1 inference in world-model mode."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        base_model_id: str,
        tokenizer_model_id: str,
        model_cache_dir: Path,
        device: str,
        checkpoint_sha256: str | None = None,
    ) -> None:
        if str(KINEWORLD_ROOT) not in sys.path:
            sys.path.insert(0, str(KINEWORLD_ROOT))

        try:
            import torch
            from diffsynth.models.utils import load_state_dict
            from diffsynth.models.wan_video_dit_dual_stream import init_flow_stream
            from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
        except ImportError as error:
            raise RuntimeError(
                "KineWorld inference dependencies are unavailable. Install "
                "requirements.txt in the GPU environment."
            ) from error

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16
        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_sha256 = checkpoint_sha256 or sha256_file(
            self.checkpoint_path
        )

        model_cache_dir = model_cache_dir.expanduser().resolve()
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        def model_config(pattern: str, *, offload: str | None = "cpu") -> Any:
            return ModelConfig(
                model_id=base_model_id,
                origin_file_pattern=pattern,
                offload_device=offload,
                local_model_path=str(model_cache_dir),
                download_resource="huggingface",
            )

        LOGGER.info("Loading Wan2.2 Stage-1 base pipeline from %s", base_model_id)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.dtype,
            device=device,
            model_configs=[
                model_config("models_t5_umt5-xxl-enc-bf16.pth"),
                model_config("diffusion_pytorch_model*.safetensors"),
                model_config("Wan2.2_VAE.pth"),
            ],
            tokenizer_config=ModelConfig(
                model_id=tokenizer_model_id,
                origin_file_pattern="google/*",
                local_model_path=str(model_cache_dir),
                download_resource="huggingface",
            ),
            redirect_common_files=False,
        )
        flow_stream = init_flow_stream(pipe.dit)

        LOGGER.info("Loading KineWorld checkpoint %s", self.checkpoint_path)
        state_dict = load_state_dict(str(self.checkpoint_path))
        dit_keys: dict[str, Any] = {}
        flow_keys: dict[str, Any] = {}
        ignored_action_keys = 0
        for key, value in state_dict.items():
            if key.startswith("action_expert."):
                ignored_action_keys += 1
            elif key.startswith("flow_stream."):
                flow_keys[key[len("flow_stream.") :]] = value
            else:
                dit_keys[key] = value

        if not dit_keys or not flow_keys:
            raise ValueError(
                "Checkpoint must contain both DiT and flow_stream weights; "
                f"found dit={len(dit_keys)}, flow={len(flow_keys)}"
            )

        fp32_dit_values = {
            key: value.clone()
            for key, value in dit_keys.items()
            if value.dtype == torch.float32
        }
        dit_missing, dit_unexpected = pipe.dit.load_state_dict(dit_keys, strict=False)
        flow_missing, flow_unexpected = flow_stream.load_state_dict(
            flow_keys, strict=False
        )
        dit_loaded = len(dit_keys) - len(dit_unexpected)
        flow_loaded = len(flow_keys) - len(flow_unexpected)
        if dit_loaded <= 0 or flow_loaded <= 0:
            raise ValueError(
                "Checkpoint did not load usable Stage-1 weights: "
                f"dit_loaded={dit_loaded}, flow_loaded={flow_loaded}"
            )

        self.load_report = {
            "dit_keys": len(dit_keys),
            "dit_loaded": dit_loaded,
            "dit_missing": len(dit_missing),
            "dit_unexpected": len(dit_unexpected),
            "flow_keys": len(flow_keys),
            "flow_loaded": flow_loaded,
            "flow_missing": len(flow_missing),
            "flow_unexpected": len(flow_unexpected),
            "ignored_action_expert_keys": ignored_action_keys,
            "fp32_dit_values": len(fp32_dit_values),
        }
        LOGGER.info("Checkpoint load report: %s", self.load_report)

        pipe.enable_vram_management()
        if fp32_dit_values:
            self._apply_fp32_modulation(pipe.dit, fp32_dit_values)

        pipe.dit.eval()
        flow_stream = flow_stream.to(
            device=device, dtype=self.dtype
        ).eval()
        self.pipe = pipe
        self.flow_stream = flow_stream

        del state_dict, dit_keys, flow_keys, fp32_dit_values
        gc.collect()

    def _apply_fp32_modulation(self, dit: Any, fp32_state_values: dict[str, Any]) -> None:
        """Restore the fp32 modulation behavior used during training."""
        torch = self.torch
        from diffsynth.vram_management.layers import (
            AutoWrappedLinear,
            WanAutoCastLayerNorm,
        )

        param_map = dict(dit.named_parameters())
        for key, fp32_value in fp32_state_values.items():
            if key in param_map:
                param_map[key].data = fp32_value.to(device=param_map[key].device)

        for sequence in (dit.time_embedding, dit.time_projection):
            for module in sequence.modules():
                if isinstance(module, AutoWrappedLinear):
                    module.offload_dtype = torch.float32
                    module.onload_dtype = torch.float32
                    module.computation_dtype = torch.float32

        def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(
                value.float() if isinstance(value, torch.Tensor) else value
                for value in args
            )

        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            return output.bfloat16() if isinstance(output, torch.Tensor) else output

        for sequence in (dit.time_embedding, dit.time_projection):
            sequence.register_forward_pre_hook(pre_hook)
            sequence.register_forward_hook(post_hook)
        for module in dit.modules():
            if isinstance(module, WanAutoCastLayerNorm):
                module.offload_dtype = torch.float32
                module.onload_dtype = torch.float32

    def prepare_episode(
        self,
        *,
        first_frame: Any,
        instruction: str,
        num_frames: int,
        native_width: int,
        conditioning_mode: str,
    ) -> PreparedEpisode:
        """Encode text and, only for the baseline, cache clean zero flow."""
        from PIL import Image

        if num_frames % 4 != 1:
            raise ValueError("Stage-1 chunk length must satisfy 4k+1")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must be non-empty")
        if conditioning_mode not in {"action_flow", "zero_flow"}:
            raise ValueError(f"unsupported conditioning mode: {conditioning_mode}")
        image = first_frame.convert("RGB")
        width, height = image.size
        native_height = max(1, round(height * native_width / width))
        tiled_height, tiled_width, checked_frames = (
            self.pipe.check_resize_height_width(
                native_height, native_width, num_frames
            )
        )
        if checked_frames != num_frames:
            raise ValueError(
                f"Wan shape checker changed chunk frames {num_frames} -> {checked_frames}"
            )

        torch = self.torch
        with torch.inference_mode():
            self.pipe.load_models_to_device(["text_encoder"])
            # Plain instruction only. RoboTwin's T-shape prompt is intentionally
            # not used by this WorldArena Stage-1 world-model path.
            context = self.pipe.prompter.encode_prompt(
                instruction, positive=True, device=self.device
            )

            flow_clean = None
            if conditioning_mode == "zero_flow":
                self.pipe.load_models_to_device(["vae"])
                zero_flow = Image.new(
                    "RGB", (tiled_width, tiled_height), (255, 255, 255)
                )
                flow_video = self.pipe.preprocess_video([zero_flow] * num_frames)
                flow_clean = self.pipe.vae.encode(
                    flow_video, device=self.device
                ).to(dtype=self.dtype, device=self.device)

        return PreparedEpisode(
            context=context,
            flow_clean=flow_clean,
            width=tiled_width,
            height=tiled_height,
            num_frames=num_frames,
            instruction=instruction,
            conditioning_mode=conditioning_mode,
        )

    def generate_keyframe_chunk(
        self,
        *,
        conditioning_frame: Any,
        prepared: PreparedEpisode,
        num_inference_steps: int,
        sigma_shift: float,
        seed: int,
        flow_frames: Sequence[Any] | None = None,
        flow_clean_latents: Any | None = None,
    ) -> list[Any]:
        """Generate one 9-keyframe chunk from current RGB and clean flow."""
        from PIL import Image

        torch = self.torch
        pipe = self.pipe
        cond_pil = conditioning_frame.convert("RGB").resize(
            (prepared.width, prepared.height), Image.Resampling.BICUBIC
        )

        with torch.inference_mode():
            pipe.load_models_to_device(["vae"])
            rgb_video = pipe.preprocess_video([cond_pil])
            rgb_prefix = pipe.vae.encode(
                rgb_video, device=self.device
            ).to(dtype=self.dtype, device=self.device)
            latent_frames = (prepared.num_frames - 1) // 4 + 1
            upscale = pipe.vae.upsampling_factor
            latent_height = prepared.height // upscale
            latent_width = prepared.width // upscale
            z_dim = getattr(pipe.vae, "z_dim", 16)

            if prepared.conditioning_mode == "action_flow":
                if (flow_frames is None) == (flow_clean_latents is None):
                    raise ValueError(
                        "action_flow requires exactly one of flow_frames or "
                        "flow_clean_latents for every chunk"
                    )
                if flow_frames is not None:
                    if len(flow_frames) != prepared.num_frames:
                        raise ValueError(
                            f"action_flow chunk has {len(flow_frames)} flow frames; "
                            f"expected {prepared.num_frames}"
                        )
                    expected_size = (prepared.width, prepared.height)
                    if any(frame.size != expected_size for frame in flow_frames):
                        raise ValueError(
                            "action-flow provider must render directly at model "
                            f"size {expected_size}; encoded flow is not resized"
                        )
                    prepared_flow = [frame.convert("RGB") for frame in flow_frames]
                    flow_video = pipe.preprocess_video(prepared_flow)
                    chunk_flow_clean = pipe.vae.encode(
                        flow_video, device=self.device
                    ).to(dtype=self.dtype, device=self.device)
                else:
                    if not isinstance(flow_clean_latents, torch.Tensor):
                        raise TypeError("flow_clean_latents must be a torch.Tensor")
                    expected_flow_shape = (
                        1,
                        z_dim,
                        latent_frames,
                        latent_height,
                        latent_width,
                    )
                    if tuple(flow_clean_latents.shape) != expected_flow_shape:
                        raise ValueError(
                            "flow_clean_latents shape mismatch: "
                            f"got {tuple(flow_clean_latents.shape)}, "
                            f"expected {expected_flow_shape}"
                        )
                    chunk_flow_clean = flow_clean_latents.to(
                        dtype=self.dtype, device=self.device
                    )
            else:
                if flow_frames is not None or flow_clean_latents is not None:
                    raise ValueError(
                        "zero_flow mode does not accept external action flow"
                    )
                if prepared.flow_clean is None:
                    raise RuntimeError("zero_flow latent was not prepared")
                chunk_flow_clean = prepared.flow_clean

            rgb_latents = pipe.generate_noise(
                (1, z_dim, latent_frames, latent_height, latent_width),
                seed=int(seed),
                rand_device="cpu",
            ).to(dtype=self.dtype, device=self.device)
            rgb_latents[:, :, :1] = rgb_prefix

            pipe.scheduler.set_timesteps(
                int(num_inference_steps), shift=float(sigma_shift)
            )
            pipe.load_models_to_device(pipe.in_iteration_models)
            for step_index, timestep in enumerate(pipe.scheduler.timesteps):
                timestep_tensor = timestep.unsqueeze(0).to(
                    dtype=self.dtype, device=self.device
                )
                rgb_prediction = self._world_model_rgb_pred(
                    rgb_latents=rgb_latents,
                    flow_clean_latents=chunk_flow_clean,
                    rgb_timestep=timestep_tensor,
                    context=prepared.context,
                )
                rgb_latents = pipe.scheduler.step(
                    rgb_prediction,
                    pipe.scheduler.timesteps[step_index],
                    rgb_latents,
                )
                # Keep the official-current frame clean at every denoising step.
                rgb_latents[:, :, :1] = rgb_prefix

            pipe.load_models_to_device(["vae"])
            frames = pipe.vae_output_to_video(
                pipe.vae.decode(rgb_latents, device=self.device)
            )
            pipe.load_models_to_device([])

        if len(frames) != prepared.num_frames:
            raise RuntimeError(
                f"Stage-1 decoded {len(frames)} frames; expected {prepared.num_frames}"
            )
        frames = [frame.convert("RGB") for frame in frames]
        # The decoded prefix can differ after VAE round-trip; pin it exactly
        # before interpolation and later pin the official PNG before encoding.
        frames[0] = cond_pil.copy()
        return frames

    # Backward-compatible alias for callers outside the Track-1 adapter.
    def generate_keyframes(self, **kwargs: Any) -> list[Any]:
        return self.generate_keyframe_chunk(**kwargs)

    def release_episode(self, prepared: PreparedEpisode | None) -> None:
        if prepared is None:
            return
        prepared.context = None
        prepared.flow_clean = None
        self.pipe.load_models_to_device([])
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def _world_model_rgb_pred(
        self,
        *,
        rgb_latents: Any,
        flow_clean_latents: Any,
        rgb_timestep: Any,
        context: Any,
    ) -> Any:
        """Exact dual-stream Stage-1 forward used by the public Space."""
        torch = self.torch
        from einops import rearrange
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
        from diffsynth.pipelines.wan_video_dual_stream import (
            _dual_stream_block_fn,
        )

        dit = self.pipe.dit
        flow_stream = self.flow_stream
        batch = rgb_latents.shape[0]
        dtype = rgb_latents.dtype
        device = rgb_latents.device

        rgb_spatial = rgb_latents.shape[3] * rgb_latents.shape[4] // 4
        rgb_temporal = rgb_latents.shape[2]
        flow_spatial = (
            flow_clean_latents.shape[3] * flow_clean_latents.shape[4] // 4
        )
        flow_temporal = flow_clean_latents.shape[2]

        per_token_timesteps = []
        for batch_index in range(batch):
            timestep = (
                rgb_timestep[batch_index]
                if rgb_timestep.dim() >= 1 and rgb_timestep.shape[0] > 1
                else rgb_timestep
            )
            rgb_token_timestep = torch.cat(
                [
                    torch.zeros(1, rgb_spatial, dtype=dtype, device=device),
                    torch.ones(
                        rgb_temporal - 1,
                        rgb_spatial,
                        dtype=dtype,
                        device=device,
                    )
                    * timestep,
                ]
            ).flatten()
            flow_token_timestep = torch.zeros(
                flow_temporal * flow_spatial, dtype=dtype, device=device
            )
            per_token_timesteps.append(
                torch.cat([rgb_token_timestep, flow_token_timestep])
            )

        token_timesteps = torch.stack(per_token_timesteps, dim=0)
        time_embedding = dit.time_embedding(
            sinusoidal_embedding_1d(
                dit.freq_dim, token_timesteps.reshape(-1)
            ).reshape(batch, -1, dit.freq_dim)
        )
        time_modulation = dit.time_projection(time_embedding).unflatten(
            2, (6, dit.dim)
        )
        embedded_context = dit.text_embedding(context)

        rgb_patches = dit.patchify(rgb_latents)
        rgb_f, rgb_h, rgb_w = rgb_patches.shape[2:]
        rgb_tokens = rearrange(
            rgb_patches, "b c f h w -> b (f h w) c"
        ).contiguous()
        rgb_token_count = rgb_tokens.shape[1]
        rgb_timestep_count = rgb_spatial * rgb_temporal
        rgb_time_embedding = time_embedding[:, :rgb_timestep_count]

        flow_patches = flow_stream.patchify(flow_clean_latents)
        flow_f, flow_h, flow_w = flow_patches.shape[2:]
        flow_tokens = rearrange(
            flow_patches, "b c f h w -> b (f h w) c"
        ).contiguous()
        flow_tokens = flow_tokens + flow_stream.stream_embed.to(
            dtype=flow_tokens.dtype, device=flow_tokens.device
        )

        rgb_freqs = torch.cat(
            [
                dit.freqs[0][:rgb_f]
                .view(rgb_f, 1, 1, -1)
                .expand(rgb_f, rgb_h, rgb_w, -1),
                dit.freqs[1][:rgb_h]
                .view(1, rgb_h, 1, -1)
                .expand(rgb_f, rgb_h, rgb_w, -1),
                dit.freqs[2][:rgb_w]
                .view(1, 1, rgb_w, -1)
                .expand(rgb_f, rgb_h, rgb_w, -1),
            ],
            dim=-1,
        ).reshape(rgb_f * rgb_h * rgb_w, 1, -1).to(rgb_tokens.device)
        flow_freqs = torch.cat(
            [
                dit.freqs[0][:flow_f]
                .view(flow_f, 1, 1, -1)
                .expand(flow_f, flow_h, flow_w, -1),
                dit.freqs[1][:flow_h]
                .view(1, flow_h, 1, -1)
                .expand(flow_f, flow_h, flow_w, -1),
                dit.freqs[2][:flow_w]
                .view(1, 1, flow_w, -1)
                .expand(flow_f, flow_h, flow_w, -1),
            ],
            dim=-1,
        ).reshape(flow_f * flow_h * flow_w, 1, -1).to(flow_tokens.device)

        for block in dit.blocks:
            rgb_tokens, flow_tokens = _dual_stream_block_fn(
                block,
                rgb_tokens,
                flow_tokens,
                embedded_context,
                time_modulation,
                rgb_freqs,
                flow_freqs,
                rgb_token_count,
            )

        rgb_output = dit.head(rgb_tokens, rgb_time_embedding)
        return dit.unpatchify(rgb_output, (rgb_f, rgb_h, rgb_w))
