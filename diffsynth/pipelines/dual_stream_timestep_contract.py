"""Pure layout contract for dual-stream per-token diffusion timesteps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DualStreamTimestepLayout:
    rgb_first: slice
    rgb_future: slice
    flow_first: slice
    flow_future: slice
    flow_all: slice
    rgb_tokens: int
    total_tokens: int


def dual_stream_timestep_layout(
    rgb_temporal: int,
    rgb_spatial: int,
    flow_temporal: int,
    flow_spatial: int,
) -> DualStreamTimestepLayout:
    """Return exact flattened token slices for RGB followed by flow."""
    values = (rgb_temporal, rgb_spatial, flow_temporal, flow_spatial)
    if any(int(value) <= 0 for value in values):
        raise ValueError(f"dual-stream token dimensions must be positive: {values}")
    rgb_tokens = int(rgb_temporal) * int(rgb_spatial)
    flow_tokens = int(flow_temporal) * int(flow_spatial)
    flow_start = rgb_tokens
    return DualStreamTimestepLayout(
        rgb_first=slice(0, int(rgb_spatial)),
        rgb_future=slice(int(rgb_spatial), rgb_tokens),
        flow_first=slice(flow_start, flow_start + int(flow_spatial)),
        flow_future=slice(flow_start + int(flow_spatial), rgb_tokens + flow_tokens),
        flow_all=slice(flow_start, rgb_tokens + flow_tokens),
        rgb_tokens=rgb_tokens,
        total_tokens=rgb_tokens + flow_tokens,
    )
