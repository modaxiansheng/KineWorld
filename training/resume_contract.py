"""Dependency-free contract for exact training-state resumes.

An Accelerate state directory restores mutable optimizer, scheduler, and RNG
state.  That state is only meaningful for the immutable workload which wrote
it, so every exact resume is bound to the fields below and fails closed if any
field (or the schema itself) changes.
"""

from __future__ import annotations

from typing import Any, Mapping


RESUME_BINDING_SCHEMA_VERSION = 1

_INTEGER_FIELDS = (
    "world_size",
    "per_device_batch",
    "gradient_accumulation_steps",
    "dataset_num_workers",
    "data_shuffle_seed",
    "size_w",
    "size_h",
    "num_frames",
    "visual_stride",
)
_OPTIONAL_INTEGER_FIELDS = ("num_video_frames",)
_STRING_FIELDS = (
    "training_manifest_sha256",
    "video_objective",
    "flow_mode",
)
RESUME_BINDING_FIELDS = (
    "schema_version",
    *_INTEGER_FIELDS,
    *_OPTIONAL_INTEGER_FIELDS,
    *_STRING_FIELDS,
)


class ResumeBindingError(ValueError):
    """Raised when an exact-resume state does not match its workload."""


def _required_int(name: str, value: Any, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ResumeBindingError(f"{name} must be an integer")
    result = value
    if positive and result <= 0:
        raise ResumeBindingError(f"{name} must be positive")
    return result


def _required_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResumeBindingError(f"{name} must be a non-empty string")
    return value


def resolve_vram_probe_optimizer_step(
    *, configured_probe_step: Any, restored_optimizer_step: Any = None
) -> int:
    """Resolve the mandatory in-process VRAM-gate optimizer step.

    ``configured_probe_step`` remains the calibrated check point (step 3 for
    formal Track1 runs).  A fresh process therefore probes at that absolute
    optimizer step.  An exact resume must prove the same real-memory policy
    again in the *new* process, so the calibrated value becomes a fixed delay
    after the restored optimizer step.  There is deliberately no separate
    resume-delay input that could weaken or bypass the calibration contract.
    """

    check_step = _required_int(
        "configured_probe_step", configured_probe_step, positive=True
    )
    if restored_optimizer_step is None:
        return check_step
    restored_step = _required_int(
        "restored_optimizer_step", restored_optimizer_step
    )
    if restored_step < 0:
        raise ResumeBindingError(
            "restored_optimizer_step must be non-negative"
        )
    return restored_step + check_step


def build_resume_binding(
    *,
    world_size: Any,
    per_device_batch: Any,
    gradient_accumulation_steps: Any,
    dataset_num_workers: Any,
    data_shuffle_seed: Any,
    training_manifest_sha256: Any,
    video_objective: Any,
    flow_mode: Any,
    size: Any,
    num_frames: Any,
    num_video_frames: Any,
    visual_stride: Any,
) -> dict[str, Any]:
    """Return the canonical immutable workload binding for trainer state."""

    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ResumeBindingError("size must contain exactly [width, height]")
    binding: dict[str, Any] = {
        "schema_version": RESUME_BINDING_SCHEMA_VERSION,
        "world_size": _required_int("world_size", world_size, positive=True),
        "per_device_batch": _required_int(
            "per_device_batch", per_device_batch, positive=True
        ),
        "gradient_accumulation_steps": _required_int(
            "gradient_accumulation_steps",
            gradient_accumulation_steps,
            positive=True,
        ),
        "dataset_num_workers": _required_int(
            "dataset_num_workers", dataset_num_workers
        ),
        "data_shuffle_seed": _required_int(
            "data_shuffle_seed", data_shuffle_seed
        ),
        "training_manifest_sha256": _required_string(
            "training_manifest_sha256", training_manifest_sha256
        ),
        "video_objective": _required_string("video_objective", video_objective),
        "flow_mode": _required_string("flow_mode", flow_mode),
        "size_w": _required_int("size_w", size[0], positive=True),
        "size_h": _required_int("size_h", size[1], positive=True),
        "num_frames": _required_int("num_frames", num_frames, positive=True),
        "num_video_frames": (
            None
            if num_video_frames is None
            else _required_int(
                "num_video_frames", num_video_frames, positive=True
            )
        ),
        "visual_stride": _required_int(
            "visual_stride", visual_stride, positive=True
        ),
    }
    if binding["dataset_num_workers"] < 0:
        raise ResumeBindingError("dataset_num_workers must be non-negative")
    return binding


def validate_resume_binding(
    stored: Any, current: Mapping[str, Any]
) -> None:
    """Require an exact schema, type, and value match.

    Unknown or missing fields fail as well: silently accepting a newer or
    legacy schema would make the resume less exact than its name promises.
    """

    if not isinstance(stored, dict):
        raise ResumeBindingError("trainer_state.resume_binding must be an object")
    if not isinstance(current, Mapping):
        raise ResumeBindingError("current resume binding must be a mapping")

    required = set(RESUME_BINDING_FIELDS)
    stored_keys = set(stored)
    current_keys = set(current)
    problems: list[str] = []
    if stored_keys != required:
        missing = sorted(required - stored_keys)
        extra = sorted(stored_keys - required)
        if missing:
            problems.append(f"stored binding missing fields {missing}")
        if extra:
            problems.append(f"stored binding has unknown fields {extra}")
    if current_keys != required:
        missing = sorted(required - current_keys)
        extra = sorted(current_keys - required)
        if missing:
            problems.append(f"current binding missing fields {missing}")
        if extra:
            problems.append(f"current binding has unknown fields {extra}")

    for field in RESUME_BINDING_FIELDS:
        if field not in stored or field not in current:
            continue
        old = stored[field]
        new = current[field]
        if type(old) is not type(new) or old != new:
            problems.append(f"{field}: saved={old!r}, current={new!r}")

    if problems:
        raise ResumeBindingError(
            "exact-resume workload binding mismatch; refusing to load state: "
            + "; ".join(problems)
        )


def build_trainer_state(
    *, global_step: Any, optimizer_step: Any, resume_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the JSON payload written beside an Accelerate state directory."""

    global_step_int = _required_int("global_step", global_step)
    optimizer_step_int = _required_int("optimizer_step", optimizer_step)
    if global_step_int < 0 or optimizer_step_int < 0:
        raise ResumeBindingError("trainer step counters must be non-negative")
    canonical_binding = dict(resume_binding)
    validate_resume_binding(canonical_binding, canonical_binding)
    return {
        "global_step": global_step_int,
        "optimizer_step": optimizer_step_int,
        "resume_binding": canonical_binding,
    }


def validate_trainer_state(
    payload: Any, current_binding: Mapping[str, Any]
) -> tuple[int, int]:
    """Validate exact-resume metadata and return its two step counters."""

    if not isinstance(payload, dict):
        raise ResumeBindingError("trainer_state.json must contain an object")
    required = {"global_step", "optimizer_step", "resume_binding"}
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing required fields {missing}")
        if unknown:
            details.append(f"has unknown fields {unknown}")
        raise ResumeBindingError(
            "trainer_state.json " + "; ".join(details)
        )
    global_step = _required_int("global_step", payload["global_step"])
    optimizer_step = _required_int("optimizer_step", payload["optimizer_step"])
    if global_step < 0 or optimizer_step < 0:
        raise ResumeBindingError("trainer step counters must be non-negative")
    validate_resume_binding(payload["resume_binding"], current_binding)
    return global_step, optimizer_step
