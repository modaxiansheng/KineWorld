"""
Load the dual-stream Wan2.2 pipeline (VAE + T5 + DiT + FlowStream) for inference.

Weights are read from a training checkpoint: DiT and flow_stream keys are loaded
into the pipeline, action_expert keys are ignored here (the action expert is
built separately). Modulation / time-MLP / LayerNorm params trained in fp32 are
restored to fp32 so inference matches training precision.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from diffsynth.models.utils import load_state_dict
from diffsynth.models.wan_video_dit_dual_stream import FlowStreamModule, init_flow_stream
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

log = logging.getLogger("pipeline_loader")


def _apply_fp32_modulation(dit, fp32_state_values: Dict[str, torch.Tensor]):
    """Restore fp32 precision for modulation / time-MLP / LayerNorm params."""
    from diffsynth.vram_management.layers import AutoWrappedLinear, WanAutoCastLayerNorm

    param_map = dict(dit.named_parameters())
    restored = 0
    for key, fp32_value in fp32_state_values.items():
        if key in param_map:
            param_map[key].data = fp32_value.to(device=param_map[key].device)
            restored += param_map[key].numel()

    for seq_module in [dit.time_embedding, dit.time_projection]:
        for sub in seq_module.modules():
            if isinstance(sub, AutoWrappedLinear):
                sub.offload_dtype = torch.float32
                sub.onload_dtype = torch.float32
                sub.computation_dtype = torch.float32

    def _pre_hook(_mod, args):
        return tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)

    def _post_hook(_mod, _args, output):
        return output.bfloat16() if isinstance(output, torch.Tensor) else output

    n_hooked = 0
    for seq_module in [dit.time_embedding, dit.time_projection]:
        seq_module.register_forward_pre_hook(_pre_hook)
        seq_module.register_forward_hook(_post_hook)
        n_hooked += 1

    for module in dit.modules():
        if isinstance(module, WanAutoCastLayerNorm):
            module.offload_dtype = torch.float32
            module.onload_dtype = torch.float32
            n_hooked += 1

    log.info(
        f"[FP32Modulation] Restored {restored:,} fp32 params, "
        f"{n_hooked} modules configured for fp32 computation"
    )


def build_pipeline(
    local_model_path: str,
    device: torch.device,
    lora_path: Optional[str] = None,
    full_path: Optional[str] = None,
) -> Tuple[WanVideoPipeline, FlowStreamModule]:
    """Load Wan2.2 pipeline + FlowStreamModule for dual-stream inference."""
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                local_model_path=local_model_path,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/*",
            local_model_path=local_model_path,
        ),
    )

    flow_stream = init_flow_stream(pipe.dit)

    fp32_dit_values = None

    if lora_path is not None and os.path.exists(lora_path):
        state_dict = load_state_dict(lora_path)
        lora_keys = {k: v for k, v in state_dict.items()
                     if "lora_A" in k or "lora_B" in k}
        dit_keys = {}
        flow_keys = {}
        for k, v in state_dict.items():
            if k in lora_keys or k.startswith("action_expert."):
                continue
            if k.startswith("flow_stream."):
                flow_keys[k.replace("flow_stream.", "")] = v
            else:
                dit_keys[k] = v
        if dit_keys:
            pipe.dit.load_state_dict(dit_keys, strict=False)
            log.info(f"DiT (LoRA module): loaded {len(dit_keys)} keys")
        if flow_keys:
            flow_stream.load_state_dict(flow_keys, strict=False)
            log.info(f"FlowStream (LoRA): loaded {len(flow_keys)} keys")
        if lora_keys:
            pipe.load_lora(pipe.dit, state_dict=lora_keys, alpha=1.0)
            log.info(f"Loaded {len(lora_keys)} LoRA keys")

    if full_path is not None and os.path.exists(full_path):
        state_dict = load_state_dict(full_path)
        dit_keys = {}
        flow_keys = {}
        for k, v in state_dict.items():
            if k.startswith("action_expert."):
                continue
            if k.startswith("flow_stream."):
                flow_keys[k.replace("flow_stream.", "")] = v
            else:
                dit_keys[k] = v

        fp32_dit_values = {k: v.clone() for k, v in dit_keys.items()
                           if v.dtype == torch.float32}

        if dit_keys:
            missing, unexpected = pipe.dit.load_state_dict(dit_keys, strict=False)
            log.info(
                f"DiT (full): loaded {len(dit_keys) - len(unexpected)} keys, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
        if flow_keys:
            missing, unexpected = flow_stream.load_state_dict(flow_keys, strict=False)
            log.info(
                f"FlowStream (full): loaded {len(flow_keys) - len(unexpected)} keys, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

    pipe.enable_vram_management()

    if fp32_dit_values:
        _apply_fp32_modulation(pipe.dit, fp32_dit_values)

    flow_stream = flow_stream.to(device=device, dtype=torch.bfloat16).eval()

    return pipe, flow_stream
