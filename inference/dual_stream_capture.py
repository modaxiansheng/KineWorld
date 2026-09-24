"""
Dual-stream video forward that captures per-layer hidden states.

Reuses ``_dual_stream_block_fn`` from diffsynth so the block math matches the
training/inference forward (``model_fn_wan_video_dual_stream``). After every DiT
block it collects the (rgb_tokens, flow_tokens) pair so the action expert can
cross-attend to per-layer video features. The setup section (timestep embedding
with per-token fuse, patchify, RoPE, stream embed) mirrors
``model_fn_wan_video_dual_stream``.
"""

import torch
import torch.utils.checkpoint
from einops import rearrange

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_dual_stream import _dual_stream_block_fn


def capture_video_layer_features(
    dit,
    flow_stream,
    latents,
    flow_latents,
    timestep,
    context,
    fuse_vae_embedding_in_latents=True,
    use_gradient_checkpointing=False,
):
    """Run the dual-stream video DiT and return per-layer [rgb, flow] tokens.

    Args mirror ``model_fn_wan_video_dual_stream``:
        dit:            WanModel (video expert).
        flow_stream:    FlowStreamModule.
        latents:        (B, C, T, H_r, W_r) RGB latents (noisy or clean cond).
        flow_latents:   (B, C, T, H_f, W_f) Flow latents.
        timestep:       (B,) diffusion timestep of the visual condition.
        context:        (B, S, text_dim) raw T5 embeddings.
        fuse_vae_embedding_in_latents: per-token timestep with first frame=0.

    Returns:
        feats: list of length ``len(dit.blocks)``. Each entry is a tuple
               ``(rgb_tokens, flow_tokens)`` with shape
               (B, S_rgb, dim) / (B, S_flow, dim) -- the block OUTPUT hidden
               states at that layer (dim = dit.dim).
    """
    B = latents.shape[0]

    # ---- Timestep (identical to model_fn_wan_video_dual_stream) ----
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        rgb_spatial = latents.shape[3] * latents.shape[4] // 4
        rgb_temporal = latents.shape[2]
        flow_spatial = flow_latents.shape[3] * flow_latents.shape[4] // 4
        flow_temporal = flow_latents.shape[2]

        t_per_token_list = []
        for b in range(B):
            ts_b = (
                timestep[b]
                if timestep.dim() >= 1 and timestep.shape[0] > 1
                else timestep
            )
            rgb_tpt = torch.cat([
                torch.zeros(1, rgb_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(rgb_temporal - 1, rgb_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()
            flow_tpt = torch.cat([
                torch.zeros(1, flow_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(flow_temporal - 1, flow_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()
            t_per_token_list.append(torch.cat([rgb_tpt, flow_tpt]))

        t_per_token = torch.stack(t_per_token_list, dim=0)
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, t_per_token.reshape(-1))
            .reshape(B, -1, dit.freq_dim)
        )
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype)
        )
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # ---- Context ----
    context = dit.text_embedding(context)

    # ---- Patchify ----
    rgb_5d = dit.patchify(latents)
    f_r, h_r, w_r = rgb_5d.shape[2:]
    rgb_tokens = rearrange(rgb_5d, 'b c f h w -> b (f h w) c').contiguous()
    n_rgb = rgb_tokens.shape[1]

    flow_5d = flow_stream.patchify(flow_latents)
    f_f, h_f, w_f = flow_5d.shape[2:]
    flow_tokens = rearrange(flow_5d, 'b c f h w -> b (f h w) c').contiguous()

    flow_tokens = flow_tokens + flow_stream.stream_embed.to(
        dtype=flow_tokens.dtype, device=flow_tokens.device)

    # ---- RoPE ----
    rgb_freqs = torch.cat([
        dit.freqs[0][:f_r].view(f_r, 1, 1, -1).expand(f_r, h_r, w_r, -1),
        dit.freqs[1][:h_r].view(1, h_r, 1, -1).expand(f_r, h_r, w_r, -1),
        dit.freqs[2][:w_r].view(1, 1, w_r, -1).expand(f_r, h_r, w_r, -1),
    ], dim=-1).reshape(f_r * h_r * w_r, 1, -1).to(rgb_tokens.device)

    flow_freqs = torch.cat([
        dit.freqs[0][:f_f].view(f_f, 1, 1, -1).expand(f_f, h_f, w_f, -1),
        dit.freqs[1][:h_f].view(1, h_f, 1, -1).expand(f_f, h_f, w_f, -1),
        dit.freqs[2][:w_f].view(1, 1, w_f, -1).expand(f_f, h_f, w_f, -1),
    ], dim=-1).reshape(f_f * h_f * w_f, 1, -1).to(flow_tokens.device)

    # ---- Block loop with per-layer capture ----
    feats = []
    ckpt = use_gradient_checkpointing and torch.is_grad_enabled()
    for block in dit.blocks:
        if ckpt:
            rgb_tokens, flow_tokens = torch.utils.checkpoint.checkpoint(
                _dual_stream_block_fn,
                block, rgb_tokens, flow_tokens,
                context, t_mod, rgb_freqs, flow_freqs, n_rgb,
                use_reentrant=False,
            )
        else:
            rgb_tokens, flow_tokens = _dual_stream_block_fn(
                block, rgb_tokens, flow_tokens,
                context, t_mod, rgb_freqs, flow_freqs, n_rgb,
            )
        feats.append((rgb_tokens, flow_tokens))

    return feats


def concat_layer_feats(feats, layer_stride=1):
    """Concatenate [rgb, flow] per captured layer -> list of (B, S_rgb+S_flow, dim).

    ``layer_stride`` subsamples captured layers (memory knob); 1 keeps all.
    """
    selected = feats[::layer_stride] if layer_stride > 1 else feats
    return [torch.cat([rgb, flow], dim=1) for (rgb, flow) in selected]
