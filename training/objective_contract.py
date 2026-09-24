"""Small dependency-free invariants for KineWorld training objectives."""


TRACK1_CONDITIONAL_RGB = "track1_conditional_rgb"
JOINT_DUAL_STREAM = "joint_dual_stream"
VIDEO_OBJECTIVES = {TRACK1_CONDITIONAL_RGB, JOINT_DUAL_STREAM}


def validate_video_objective(value: str) -> str:
    value = str(value)
    if value not in VIDEO_OBJECTIVES:
        raise ValueError(f"unsupported video_objective={value!r}")
    return value


def freeze_conditional_flow_head(flow_stream, video_objective: str) -> int:
    """Freeze the flow-only prediction head when no flow loss is computed."""
    objective = validate_video_objective(video_objective)
    if objective != TRACK1_CONDITIONAL_RGB:
        return 0
    flow_stream.flow_head.requires_grad_(False)
    remaining = [
        name
        for name, parameter in flow_stream.flow_head.named_parameters()
        if parameter.requires_grad
    ]
    if remaining:
        raise RuntimeError(
            f"conditional objective left unused flow-head params trainable: {remaining}"
        )
    return sum(parameter.numel() for parameter in flow_stream.flow_head.parameters())
