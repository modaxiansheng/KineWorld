#!/usr/bin/env python3
"""Strict per-episode validation for variable-length WorldArena2 Track-1 videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import struct
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_DIRECTORY = "fixed_scene_task"
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
ACTION_DATASET = "/joint_action/vector"
ACTION_DIM = 14
KEYFRAMES_PER_CHUNK = 9
VISUAL_STRIDE = 4
CHUNK_HORIZON = (KEYFRAMES_PER_CHUNK - 1) * VISUAL_STRIDE
PRECOMPUTED_MANIFEST_SCHEMA_VERSION = 2
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FLOW_MODEL_RESIZE_POLICY = "pil_rgb_default_resize_matching_training_online_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MP4s against each episode's /joint_action/vector length"
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument(
        "--records-dir",
        type=Path,
        help="per-episode provenance directory (default: sibling per_episode)",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="run manifest JSON (default: sibling run_config.json)",
    )
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True)
    parser.add_argument(
        "--default-frame-count",
        type=int,
        help="explicit fallback only when an episode HDF5 file is absent",
    )
    parser.add_argument("--expected-fps", type=float, default=24.0)
    parser.add_argument("--expected-width", type=int, default=640)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--minimum-first-frame-psnr", type=float, default=30.0)
    parser.add_argument(
        "--required-control-type",
        choices=("action_driven", "text_driven", "none"),
        default="action_driven",
        help="default final gate rejects the explicit zero-flow baseline",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path, path / "dataset_track1"):
        if all(
            (candidate / name / TASK_DIRECTORY).is_dir()
            for name in ("data", "first_frame", "instructions")
        ):
            return candidate
    raise FileNotFoundError(
        f"expected data/, first_frame/, instructions/ under {path}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _sha256_file_identity(path_text: str, size: int, mtime_ns: int) -> str:
    # size/mtime are cache-key identity guards; hashing still reads path_text.
    del size, mtime_ns
    return sha256_file(Path(path_text))


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def invalid_sha256_fields(value: Any, path: str = "$") -> list[str]:
    """Return every sha256-named field whose value is not 64 hex chars."""
    invalid: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "sha256" in str(key).lower() and not is_sha256_hex(child):
                invalid.append(child_path)
            invalid.extend(invalid_sha256_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            invalid.extend(invalid_sha256_fields(child, f"{path}[{index}]"))
    return invalid


def compute_action_chunk_sha256(dataset: Any, source_indices: Sequence[int]) -> str:
    """Hash selected rows as C-contiguous little-endian float32 ``[K,14]``."""
    digest = hashlib.sha256()
    for source_index in source_indices:
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"action source index is not an integer: {source_index!r}")
        row = dataset[source_index]
        try:
            if len(row) != ACTION_DIM:
                raise ValueError(
                    f"action row {source_index} has {len(row)} values; expected {ACTION_DIM}"
                )
            values = tuple(float(value) for value in row)
        except TypeError as error:
            raise ValueError(f"action row {source_index} is not one-dimensional") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"action row {source_index} contains NaN/Inf")
        if not (0.0 <= values[6] <= 1.0 and 0.0 <= values[13] <= 1.0):
            raise ValueError(
                f"action row {source_index} grippers at columns 6/13 are not in [0,1]"
            )
        # struct '<14f' is exactly the provider's np.asarray(..., dtype='<f4',
        # order='C').tobytes(order='C') representation, without requiring
        # NumPy in the CPU unit test.
        digest.update(struct.pack(f"<{ACTION_DIM}f", *values))
    return digest.hexdigest()


def expected_chunk_starts(frame_count: int) -> list[int]:
    return list(range(0, frame_count - 1, CHUNK_HORIZON))


def expected_action_indices(chunk_start: int, frame_count: int) -> list[int]:
    return [
        min(frame_count - 1, chunk_start + offset * VISUAL_STRIDE)
        for offset in range(KEYFRAMES_PER_CHUNK)
    ]


def _same_resolved_path(declared: Any, actual: Path) -> bool:
    if not isinstance(declared, str):
        return False
    try:
        return Path(declared).expanduser().resolve() == actual.resolve()
    except OSError:
        return False


def _number_matches(value: Any, expected: float, tolerance: float = 0.0) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """JSON equality that does not treat booleans as integers."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _verify_path_sha256(mapping: Any, label: str, errors: list[str]) -> None:
    if not isinstance(mapping, Mapping):
        errors.append(f"{label} provenance is missing")
        return
    path_value = mapping.get("path")
    claimed = mapping.get("sha256")
    if not isinstance(path_value, str) or not is_sha256_hex(claimed):
        errors.append(f"{label} path/SHA-256 is invalid")
        return
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        errors.append(f"{label} file is unavailable for rehash: {path}")
        return
    stat = path.stat()
    actual = _sha256_file_identity(str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if actual != claimed.lower():
        errors.append(f"{label} SHA-256 mismatch: {path}")


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def inspect_rgb8_png(path: Path, *, require_all_white: bool) -> tuple[int, int]:
    """Validate a non-interlaced RGB8 PNG and optionally decode its white sentinel."""
    payload = path.read_bytes()
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")
    offset = len(PNG_SIGNATURE)
    width = height = None
    idat_parts: list[bytes] = []
    saw_iend = False
    chunk_index = 0
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"truncated PNG chunk in {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError(f"truncated PNG data in {path}")
        chunk_data = payload[data_start:data_end]
        claimed_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != claimed_crc:
            raise ValueError(f"PNG CRC mismatch in {path}")
        if chunk_index == 0 and chunk_type != b"IHDR":
            raise ValueError(f"PNG IHDR is not first in {path}")
        if chunk_type == b"IHDR":
            if width is not None or length != 13:
                raise ValueError(f"invalid PNG IHDR in {path}")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filtering,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk_data)
            if width < 1 or height < 1:
                raise ValueError(f"invalid PNG dimensions in {path}")
            if (bit_depth, color_type, compression, filtering, interlace) != (
                8,
                2,
                0,
                0,
                0,
            ):
                raise ValueError(
                    f"precomputed flow PNG must be non-interlaced RGB8: {path}"
                )
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            if length != 0:
                raise ValueError(f"invalid PNG IEND in {path}")
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
        chunk_index += 1
    if width is None or height is None or not idat_parts or not saw_iend:
        raise ValueError(f"incomplete PNG structure in {path}")
    if offset != len(payload):
        raise ValueError(f"trailing bytes after PNG IEND in {path}")
    if not require_all_white:
        return width, height

    stride = width * 3
    expected_bytes = height * (stride + 1)
    if expected_bytes > 100 * 1024 * 1024:
        raise ValueError(f"PNG sentinel is unreasonably large: {path}")
    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error as error:
        raise ValueError(f"invalid PNG compressed data in {path}") from error
    if len(raw) != expected_bytes:
        raise ValueError(f"PNG decompressed byte count mismatch in {path}")
    previous = bytearray(stride)
    raw_offset = 0
    for _row_index in range(height):
        filter_type = raw[raw_offset]
        encoded = raw[raw_offset + 1 : raw_offset + 1 + stride]
        decoded = bytearray(stride)
        for byte_index, encoded_value in enumerate(encoded):
            left = decoded[byte_index - 3] if byte_index >= 3 else 0
            above = previous[byte_index]
            upper_left = previous[byte_index - 3] if byte_index >= 3 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth_predictor(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_type} in {path}")
            decoded[byte_index] = (encoded_value + predictor) & 0xFF
        if any(value != 255 for value in decoded):
            raise ValueError(f"precomputed frame zero is not a white sentinel: {path}")
        previous = decoded
        raw_offset += stride + 1
    return width, height


def validate_renderer_provenance(renderer: Any) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if not isinstance(renderer, Mapping):
        return None, ["renderer provenance is missing"]
    if renderer.get("embodiment") != "aloha-agilex":
        errors.append("renderer embodiment is not aloha-agilex")
    camera = renderer.get("camera")
    if not isinstance(camera, Mapping) or camera.get("name") != "head_camera":
        errors.append("renderer camera is not head_camera")
    expected_layout = [
        "left_arm[0:6]",
        "left_gripper[6]",
        "right_arm[7:13]",
        "right_gripper[13]",
    ]
    if renderer.get("action_layout") != expected_layout:
        errors.append("renderer action layout is not official Aloha-AgileX 14D")
    _verify_path_sha256(renderer.get("urdf"), "renderer URDF", errors)
    _verify_path_sha256(
        renderer.get("embodiment_config"), "renderer embodiment config", errors
    )
    _verify_path_sha256(
        renderer.get("renderer_module"), "renderer module", errors
    )
    try:
        identity = canonical_signature(dict(renderer))
    except (TypeError, ValueError):
        errors.append("renderer provenance is not canonical JSON")
        identity = None
    return identity, errors


def validate_precomputed_chunk_manifest(
    *,
    expected_conditioner: Mapping[str, Any],
    precomputed: Any,
    flow_provenance: Mapping[str, Any],
    episode_id: int,
    chunk_start: int,
    frame_count: int,
    expected_indices: list[int],
    target_size: tuple[int, int],
    actual_action_digest: str | None,
) -> list[str]:
    """Bind one precomputed provenance block to its v2 manifest and PNG bytes."""
    label = f"chunk at start {chunk_start}"
    errors: list[str] = []
    if not isinstance(precomputed, Mapping):
        return [f"{label} lacks required precomputed provenance"]
    root_value = expected_conditioner.get("root")
    if not isinstance(root_value, str):
        return ["run precomputed conditioner root is missing"]
    precomputed_root = Path(root_value).expanduser().resolve()
    if not precomputed_root.is_dir():
        errors.append(f"run precomputed root is unavailable: {precomputed_root}")
    if not _same_resolved_path(precomputed.get("root"), precomputed_root):
        errors.append(f"{label} precomputed root disagrees with run conditioner")

    expected_manifest = (
        precomputed_root
        / f"episode{episode_id}"
        / f"chunk_{chunk_start:06d}"
        / "manifest.json"
    ).resolve()
    if not _same_resolved_path(precomputed.get("manifest"), expected_manifest):
        errors.append(f"{label} precomputed manifest path is not the expected chunk")
    if not expected_manifest.is_file():
        errors.append(f"{label} precomputed manifest is unavailable: {expected_manifest}")
        return errors
    claimed_manifest_digest = precomputed.get("manifest_sha256")
    actual_manifest_digest = sha256_file(expected_manifest)
    if not is_sha256_hex(claimed_manifest_digest):
        errors.append(f"{label} precomputed manifest SHA-256 is invalid")
    elif claimed_manifest_digest.lower() != actual_manifest_digest:
        errors.append(f"{label} precomputed manifest SHA-256 mismatch")

    try:
        manifest = load_json_mapping(expected_manifest, "precomputed manifest")
    except RuntimeError as error:
        errors.append(str(error))
        return errors
    expected_fields = {
        "schema_version": PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "chunk_start": chunk_start,
        "frame_count": frame_count,
        "keyframe_count": KEYFRAMES_PER_CHUNK,
        "visual_stride": VISUAL_STRIDE,
        "source_indices": expected_indices,
        "target_size": [target_size[0], target_size[1]],
        "action_sha256_float32_le": actual_action_digest,
    }
    for field, expected in expected_fields.items():
        if not _strict_json_equal(manifest.get(field), expected):
            errors.append(
                f"{label} manifest {field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if actual_action_digest is None or not is_sha256_hex(
        manifest.get("action_sha256_float32_le")
    ):
        errors.append(f"{label} manifest action SHA-256 cannot be verified")

    manifest_provenance = manifest.get("provenance")
    flow_without_precomputed = dict(flow_provenance)
    flow_without_precomputed.pop("precomputed", None)
    if not isinstance(manifest_provenance, Mapping):
        errors.append(f"{label} manifest provenance is missing")
    else:
        try:
            provenance_matches = canonical_signature(
                dict(manifest_provenance)
            ) == canonical_signature(flow_without_precomputed)
        except (TypeError, ValueError):
            provenance_matches = False
        if not provenance_matches:
            errors.append(f"{label} manifest provenance disagrees with episode record")

    frame_names = manifest.get("frames")
    if (
        not isinstance(frame_names, list)
        or len(frame_names) != KEYFRAMES_PER_CHUNK
        or not all(isinstance(name, str) for name in frame_names)
        or len(set(frame_names)) != KEYFRAMES_PER_CHUNK
        or any(
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            for name in frame_names
        )
    ):
        errors.append(f"{label} manifest frames are not nine unique local basenames")
        return errors
    frame_sha256 = manifest.get("frame_sha256")
    if not isinstance(frame_sha256, Mapping) or set(frame_sha256) != set(frame_names):
        errors.append(f"{label} manifest frame_sha256 does not cover all nine frames")
        return errors

    chunk_dir = expected_manifest.parent.resolve()
    for frame_index, frame_name in enumerate(frame_names):
        claimed_frame_digest = frame_sha256.get(frame_name)
        if not is_sha256_hex(claimed_frame_digest):
            errors.append(f"{label} frame SHA-256 is invalid for {frame_name!r}")
            continue
        frame_path = (chunk_dir / frame_name).resolve()
        if frame_path.parent != chunk_dir or not frame_path.is_file():
            errors.append(f"{label} frame is unavailable or escapes chunk dir: {frame_name}")
            continue
        if sha256_file(frame_path) != claimed_frame_digest.lower():
            errors.append(f"{label} frame SHA-256 mismatch: {frame_name}")
            continue
        try:
            actual_size = inspect_rgb8_png(
                frame_path, require_all_white=frame_index == 0
            )
        except (OSError, ValueError) as error:
            errors.append(f"{label} invalid frame {frame_name}: {error}")
            continue
        if actual_size != target_size:
            errors.append(
                f"{label} frame {frame_name} size {actual_size}, expected {target_size}"
            )
    return errors


def psnr(first: Any, second: Any, np: Any) -> float:
    mse = float(np.mean((first.astype(np.float32) - second.astype(np.float32)) ** 2))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def trajectory_frame_count(
    path: Path, *, h5py: Any, default_frame_count: int | None
) -> tuple[int, str]:
    if not path.is_file():
        if default_frame_count is None:
            raise FileNotFoundError(path)
        return default_frame_count, "default_frame_count_fallback"
    with h5py.File(path, "r") as handle:
        key = "joint_action/vector"
        if key not in handle:
            raise KeyError(f"missing /{key} in {path}")
        dataset = handle[key]
        if not dataset.shape:
            raise ValueError(f"/{key} must have a frame dimension in {path}")
        if len(dataset.shape) != 2 or int(dataset.shape[1]) != 14:
            raise ValueError(
                f"/{key} must have shape [N,14], got {tuple(dataset.shape)} in {path}"
            )
        frame_count = int(dataset.shape[0])
    if frame_count < 2:
        raise ValueError(f"invalid /joint_action/vector length {frame_count} in {path}")
    return frame_count, "/joint_action/vector"


def load_json_mapping(path: Path, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {what} {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{what} is not a JSON object: {path}")
    return dict(payload)


def validate_run_manifest(
    *,
    run_manifest: Mapping[str, Any],
    run_manifest_path: Path,
    dataset_root: Path,
    required_control_type: str,
    expected_fps: float,
    expected_width: int,
    expected_height: int,
) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
    signature = run_manifest.get("run_signature")
    signature_inputs = run_manifest.get("signature_inputs")
    conditioning = run_manifest.get("conditioning")
    checkpoint = run_manifest.get("checkpoint")
    expected_mode = (
        "action_flow" if required_control_type == "action_driven" else "zero_flow"
    )
    expected_zero_flag = required_control_type == "text_driven"
    run_conditioner = (
        conditioning.get("conditioner") if isinstance(conditioning, Mapping) else None
    )
    input_conditioner = (
        signature_inputs.get("action_flow_conditioner")
        if isinstance(signature_inputs, Mapping)
        else None
    )
    invalid_sha_fields = invalid_sha256_fields(run_manifest)
    checks = {
        "run_manifest_schema_version": run_manifest.get("schema_version") == 1,
        "run_signature_is_sha256": is_sha256_hex(signature),
        "run_signature_recomputed": isinstance(signature_inputs, Mapping)
        and is_sha256_hex(signature)
        and canonical_signature(dict(signature_inputs)) == signature.lower(),
        "run_dataset_root_matches": _same_resolved_path(
            run_manifest.get("dataset_root"), dataset_root
        ),
        "run_conditioning_mode_matches": isinstance(conditioning, Mapping)
        and conditioning.get("mode") == expected_mode,
        "run_control_type_matches": isinstance(conditioning, Mapping)
        and conditioning.get("control_type") == required_control_type,
        "run_zero_flow_flag_matches": isinstance(conditioning, Mapping)
        and conditioning.get("zero_flow_is_explicit_baseline")
        is expected_zero_flag,
        "run_signature_conditioning_matches": isinstance(signature_inputs, Mapping)
        and signature_inputs.get("conditioning_mode") == expected_mode
        and signature_inputs.get("control_type") == required_control_type,
        "run_conditioner_is_consistent": run_conditioner == input_conditioner,
        "run_track_contract_matches": isinstance(signature_inputs, Mapping)
        and signature_inputs.get("keyframes_per_chunk") == KEYFRAMES_PER_CHUNK
        and signature_inputs.get("visual_stride") == VISUAL_STRIDE
        and signature_inputs.get("frame_count_source") == ACTION_DATASET
        and signature_inputs.get("output_width") == expected_width
        and signature_inputs.get("output_height") == expected_height
        and _number_matches(signature_inputs.get("fps"), expected_fps, 0.05)
        and signature_inputs.get("video_codec") == "h264/libx264/yuv420p",
        "run_checkpoint_sha256_valid": isinstance(checkpoint, Mapping)
        and is_sha256_hex(checkpoint.get("sha256")),
        "run_all_sha256_fields_valid": not invalid_sha_fields,
    }
    errors = [name for name, passed in checks.items() if not passed]
    errors.extend(f"invalid SHA-256 field: {path}" for path in invalid_sha_fields)

    renderer_identity = None
    renderer_errors: list[str] = []
    if required_control_type == "action_driven":
        allowed_conditioners = {
            "action_driven_robot_only_flow.v1",
            "precomputed_action_flow.v1",
        }
        conditioner_name = (
            run_conditioner.get("conditioner")
            if isinstance(run_conditioner, Mapping)
            else None
        )
        if conditioner_name not in allowed_conditioners:
            renderer_errors.append(
                f"untrusted action-flow conditioner: {conditioner_name!r}"
            )
        if conditioner_name == "action_driven_robot_only_flow.v1":
            renderer_identity, renderer_errors_now = validate_renderer_provenance(
                run_conditioner.get("renderer")
                if isinstance(run_conditioner, Mapping)
                else None
            )
            renderer_errors.extend(renderer_errors_now)
        elif conditioner_name == "precomputed_action_flow.v1":
            manifest_schema = run_conditioner.get("manifest_schema_version")
            if manifest_schema != PRECOMPUTED_MANIFEST_SCHEMA_VERSION:
                renderer_errors.append(
                    "precomputed conditioner does not require manifest schema v2"
                )
            root_value = run_conditioner.get("root")
            if not isinstance(root_value, str) or not Path(
                root_value
            ).expanduser().resolve().is_dir():
                renderer_errors.append("precomputed conditioner root is unavailable")
    elif run_conditioner is not None:
        renderer_errors.append("text-driven run unexpectedly declares an action conditioner")
    checks["run_action_provider_trusted"] = not renderer_errors
    errors.extend(renderer_errors)
    summary = {
        "path": str(run_manifest_path),
        "run_signature": signature,
        "conditioning_mode": (
            conditioning.get("mode") if isinstance(conditioning, Mapping) else None
        ),
        "control_type": (
            conditioning.get("control_type")
            if isinstance(conditioning, Mapping)
            else None
        ),
        "checkpoint_sha256": (
            checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
        ),
        "conditioner": run_conditioner,
        "renderer_identity": renderer_identity,
        "signature_inputs": dict(signature_inputs)
        if isinstance(signature_inputs, Mapping)
        else None,
    }
    return checks, errors, summary


def inspect_video(
    *,
    video_path: Path,
    first_frame_path: Path,
    expected_frames: int,
    expected_fps: float,
    expected_width: int,
    expected_height: int,
    cv2: Any,
    np: Any,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc_value = int(round(capture.get(cv2.CAP_PROP_FOURCC)))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))

    decoded_frames = 0
    first_frame = None
    middle_frame = None
    last_frame = None
    middle_index = expected_frames // 2
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded_frames == 0:
            first_frame = frame.copy()
        if decoded_frames == middle_index:
            middle_frame = frame.copy()
        last_frame = frame
        decoded_frames += 1
    capture.release()
    if first_frame is None or last_frame is None:
        raise RuntimeError(f"no decodable frames in {video_path}")
    if middle_frame is None:
        middle_frame = last_frame

    source = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"could not read {first_frame_path}")
    source_height, source_width = source.shape[:2]
    source = cv2.resize(
        source, (expected_width, expected_height), interpolation=cv2.INTER_LANCZOS4
    )
    first_frame_psnr = psnr(source, first_frame, np)
    temporal_delta_middle = float(
        np.mean(
            np.abs(
                middle_frame.astype(np.float32) - first_frame.astype(np.float32)
            )
        )
    )
    temporal_delta_last = float(
        np.mean(
            np.abs(last_frame.astype(np.float32) - first_frame.astype(np.float32))
        )
    )
    codec_is_h264 = fourcc.lower() in {"avc1", "avc3", "h264", "x264"}
    checks = {
        "declared_frame_count_matches_hdf5": declared_frames == expected_frames,
        "decoded_frame_count_matches_hdf5": decoded_frames == expected_frames,
        "fps_matches": abs(fps - expected_fps) <= 0.05,
        "resolution_matches": (width, height) == (expected_width, expected_height),
        "codec_is_h264": codec_is_h264,
    }
    return {
        "video": str(video_path),
        "sha256": sha256_file(video_path),
        "bytes": video_path.stat().st_size,
        "expected_frames_from_hdf5": expected_frames,
        "declared_frames": declared_frames,
        "decoded_frames": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "first_frame_psnr": first_frame_psnr,
        "source_first_frame_width": int(source_width),
        "source_first_frame_height": int(source_height),
        "temporal_delta_middle": temporal_delta_middle,
        "temporal_delta_last": temporal_delta_last,
        "checks": checks,
    }


def validate_action_chunk_provenance(
    *,
    chunks: Any,
    action_dataset: Any,
    episode_id: int,
    frame_count: int,
    hdf5_path: Path,
    native_size: tuple[int, int],
    flow_source_size: tuple[int, int] | None = None,
    first_frame_size: tuple[int, int],
    expected_conditioner: Mapping[str, Any] | None = None,
) -> tuple[dict[str, bool], list[str], str | None]:
    """Recompute every action digest and validate the 9/4/32 chunk contract."""
    errors: list[str] = []
    structure_ok = True
    digests_ok = True
    renderers_ok = True
    precomputed_ok = True
    expected_starts = expected_chunk_starts(frame_count)
    conditioner_name = (
        expected_conditioner.get("conditioner")
        if isinstance(expected_conditioner, Mapping)
        else None
    )
    precomputed_required = conditioner_name == "precomputed_action_flow.v1"
    manifested_flow_size = flow_source_size or native_size
    if not isinstance(chunks, list) or len(chunks) != len(expected_starts):
        return (
            {
                "provenance_action_chunk_structure": False,
                "provenance_action_digests_recomputed": False,
                "provenance_renderer_verified": False,
                "provenance_precomputed_manifests_verified": False,
            },
            [
                f"chunk list has {len(chunks) if isinstance(chunks, list) else 'invalid'} "
                f"entries; expected {len(expected_starts)}"
            ],
            None,
        )

    renderer_identities: list[str] = []
    for chunk_index, chunk_start in enumerate(expected_starts):
        label = f"chunk[{chunk_index}]"
        chunk = chunks[chunk_index]
        if not isinstance(chunk, Mapping):
            errors.append(f"{label} is not an object")
            structure_ok = False
            digests_ok = False
            renderers_ok = False
            if precomputed_required:
                precomputed_ok = False
            continue
        expected_indices = expected_action_indices(chunk_start, frame_count)
        expected_tail_clamped = len(set(expected_indices)) != len(expected_indices)
        expected_consumed_end = min(chunk_start + CHUNK_HORIZON, frame_count - 1)
        wrapper_expectations = {
            "chunk_index": chunk_index,
            "keyframe_count": KEYFRAMES_PER_CHUNK,
            "generated_frame_range": [chunk_start, chunk_start + CHUNK_HORIZON],
            "consumed_frame_range": [chunk_start, expected_consumed_end],
        }
        for field, expected in wrapper_expectations.items():
            if not _strict_json_equal(chunk.get(field), expected):
                structure_ok = False
                errors.append(
                    f"{label}.{field}={chunk.get(field)!r}, expected {expected!r}"
                )

        flow = chunk.get("flow_conditioning")
        flow_model_adapter = chunk.get("flow_model_adapter")
        if not isinstance(flow, Mapping):
            errors.append(f"{label}.flow_conditioning is missing")
            structure_ok = False
            digests_ok = False
            renderers_ok = False
            if precomputed_required:
                precomputed_ok = False
            continue
        source = flow.get("source")
        flow_chunk = flow.get("chunk")
        actual_action_digest = None
        declared_indices = (
            flow_chunk.get("source_indices")
            if isinstance(flow_chunk, Mapping)
            else None
        )
        digest_indices = None
        if (
            isinstance(declared_indices, list)
            and len(declared_indices) == KEYFRAMES_PER_CHUNK
            and all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < frame_count
                for index in declared_indices
            )
        ):
            digest_indices = declared_indices
            actual_action_digest = compute_action_chunk_sha256(
                action_dataset, digest_indices
            )
        else:
            digests_ok = False
            errors.append(f"{label} action source_indices cannot be rehashed")
        if flow.get("mode") != "action_flow":
            structure_ok = False
            errors.append(f"{label} flow mode is not action_flow")
        if flow.get("control_type") != "official_14d_joint_action":
            structure_ok = False
            errors.append(f"{label} flow control_type is not official 14D action")
        if flow.get("episode_id") != episode_id:
            structure_ok = False
            errors.append(f"{label} flow episode_id mismatch")
        if not isinstance(source, Mapping):
            structure_ok = False
            digests_ok = False
            errors.append(f"{label} action source is missing")
        else:
            if source.get("dataset") != ACTION_DATASET:
                structure_ok = False
                errors.append(f"{label} action dataset is not {ACTION_DATASET}")
            if source.get("shape") != [frame_count, ACTION_DIM]:
                structure_ok = False
                errors.append(f"{label} action shape mismatch")
            if not _same_resolved_path(source.get("hdf5_path"), hdf5_path):
                structure_ok = False
                errors.append(f"{label} HDF5 path mismatch")
            claimed_digest = source.get("selected_action_sha256_float32_le")
            if not is_sha256_hex(claimed_digest):
                digests_ok = False
                errors.append(f"{label} action digest is not 64-digit hex")
            elif digest_indices is None:
                digests_ok = False
            elif actual_action_digest != claimed_digest.lower():
                digests_ok = False
                errors.append(f"{label} action digest mismatch")

        flow_expectations = {
            "start": chunk_start,
            "keyframe_count": KEYFRAMES_PER_CHUNK,
            "visual_stride": VISUAL_STRIDE,
            "source_indices": expected_indices,
            "tail_clamped": expected_tail_clamped,
        }
        if not isinstance(flow_chunk, Mapping):
            structure_ok = False
            errors.append(f"{label} flow chunk metadata is missing")
        else:
            for field, expected in flow_expectations.items():
                if not _strict_json_equal(flow_chunk.get(field), expected):
                    structure_ok = False
                    errors.append(
                        f"{label}.flow.chunk.{field}={flow_chunk.get(field)!r}, "
                        f"expected {expected!r}"
                    )
        if flow.get("target_size") != [
            manifested_flow_size[0],
            manifested_flow_size[1],
        ]:
            structure_ok = False
            errors.append(f"{label} target_size mismatch")
        expected_resize_policy = (
            "identity"
            if manifested_flow_size == native_size
            else FLOW_MODEL_RESIZE_POLICY
        )
        expected_adapter = {
            "source_size": [manifested_flow_size[0], manifested_flow_size[1]],
            "target_size": [native_size[0], native_size[1]],
            "resize_policy": expected_resize_policy,
        }
        if not isinstance(flow_model_adapter, Mapping):
            structure_ok = False
            errors.append(f"{label}.flow_model_adapter is missing")
        else:
            for field, expected in expected_adapter.items():
                if not _strict_json_equal(flow_model_adapter.get(field), expected):
                    structure_ok = False
                    errors.append(
                        f"{label}.flow_model_adapter.{field}="
                        f"{flow_model_adapter.get(field)!r}, expected {expected!r}"
                    )
        if flow.get("first_frame_size") != [first_frame_size[0], first_frame_size[1]]:
            structure_ok = False
            errors.append(f"{label} first_frame_size mismatch")
        flow_details = flow.get("flow")
        if (
            not isinstance(flow_details, Mapping)
            or flow_details.get("frame_zero") != "white_zero_flow_sentinel"
        ):
            structure_ok = False
            errors.append(f"{label} lacks the FlowCodec frame-zero sentinel proof")

        renderer_identity, renderer_errors = validate_renderer_provenance(
            flow.get("renderer")
        )
        if renderer_errors or renderer_identity is None:
            renderers_ok = False
            errors.extend(f"{label}: {error}" for error in renderer_errors)
        else:
            renderer_identities.append(renderer_identity)

        precomputed = flow.get("precomputed")
        if precomputed_required:
            manifest_errors = validate_precomputed_chunk_manifest(
                expected_conditioner=expected_conditioner,
                precomputed=precomputed,
                flow_provenance=flow,
                episode_id=episode_id,
                chunk_start=chunk_start,
                frame_count=frame_count,
                expected_indices=expected_indices,
                target_size=manifested_flow_size,
                actual_action_digest=actual_action_digest,
            )
            if manifest_errors:
                precomputed_ok = False
                errors.extend(f"{label}: {error}" for error in manifest_errors)
        elif precomputed is not None:
            if not isinstance(precomputed, Mapping):
                precomputed_ok = False
                errors.append(f"{label} precomputed provenance is invalid")
            else:
                manifest_value = precomputed.get("manifest")
                manifest_digest = precomputed.get("manifest_sha256")
                if not isinstance(manifest_value, str) or not is_sha256_hex(
                    manifest_digest
                ):
                    precomputed_ok = False
                    errors.append(f"{label} precomputed manifest path/SHA is invalid")
                else:
                    manifest_path = Path(manifest_value).expanduser().resolve()
                    if not manifest_path.is_file():
                        precomputed_ok = False
                        errors.append(
                            f"{label} precomputed manifest is unavailable: {manifest_path}"
                        )
                    elif sha256_file(manifest_path) != manifest_digest.lower():
                        precomputed_ok = False
                        errors.append(f"{label} precomputed manifest SHA mismatch")

    renderer_identity = renderer_identities[0] if renderer_identities else None
    if len(renderer_identities) != len(expected_starts) or len(
        set(renderer_identities)
    ) != 1:
        renderers_ok = False
        errors.append("renderer/URDF provenance changes across chunks")
    checks = {
        "provenance_action_chunk_structure": structure_ok,
        "provenance_action_digests_recomputed": digests_ok,
        "provenance_renderer_verified": renderers_ok,
        "provenance_precomputed_manifests_verified": precomputed_ok,
    }
    return checks, errors, renderer_identity


def inspect_provenance(
    *,
    record_path: Path,
    episode_id: int,
    hdf5_path: Path,
    instruction_path: Path,
    first_frame_path: Path,
    video_path: Path,
    expected_frames: int,
    required_control_type: str,
    run_manifest_summary: Mapping[str, Any],
    video_sha256: str,
    source_first_frame_size: tuple[int, int],
    h5py: Any,
) -> tuple[dict[str, Any], dict[str, bool]]:
    payload = load_json_mapping(record_path, "episode provenance")
    expected_mode = (
        "action_flow" if required_control_type == "action_driven" else "zero_flow"
    )
    source = payload.get("source")
    video = payload.get("video")
    checkpoint = payload.get("checkpoint")
    rollout = payload.get("rollout")
    world_model = payload.get("world_model_mode")
    chunks = rollout.get("chunks") if isinstance(rollout, Mapping) else None
    expected_chunks = len(expected_chunk_starts(expected_frames))
    invalid_sha_fields = invalid_sha256_fields(payload)
    official_instruction = load_json_mapping(
        instruction_path, "official instruction JSON"
    ).get("instruction")
    actual_hdf5_sha256 = sha256_file(hdf5_path)
    actual_instruction_sha256 = sha256_file(instruction_path)
    actual_first_frame_sha256 = sha256_file(first_frame_path)
    run_signature = run_manifest_summary.get("run_signature")
    run_checkpoint_sha256 = run_manifest_summary.get("checkpoint_sha256")
    run_conditioner = run_manifest_summary.get("conditioner")
    run_inputs = run_manifest_summary.get("signature_inputs")
    checks = {
        "provenance_schema_version": payload.get("schema_version") == 1,
        "provenance_status_complete": payload.get("status") == "complete",
        "provenance_episode_identity_matches": payload.get("episode_id") == episode_id
        and payload.get("episode_stem") == f"episode{episode_id}",
        "provenance_run_signature_matches": is_sha256_hex(
            payload.get("run_signature")
        )
        and payload.get("run_signature", "").lower() == run_signature,
        "provenance_control_type_matches": payload.get("control_type")
        == required_control_type,
        "provenance_conditioning_mode_matches": payload.get("conditioning_mode")
        == expected_mode,
        "provenance_adapter_contract_matches_run": isinstance(run_inputs, Mapping)
        and payload.get("adapter_version") == run_inputs.get("adapter_version")
        and payload.get("prompt_policy") == run_inputs.get("prompt_policy"),
        "provenance_instruction_matches": isinstance(official_instruction, str)
        and payload.get("instruction") == official_instruction.strip(),
        "provenance_source_paths_match": isinstance(source, Mapping)
        and _same_resolved_path(source.get("hdf5"), hdf5_path)
        and _same_resolved_path(source.get("instruction_json"), instruction_path)
        and _same_resolved_path(source.get("first_frame_png"), first_frame_path),
        "provenance_source_frames_match": isinstance(source, Mapping)
        and source.get("trajectory_frame_count") == expected_frames,
        "provenance_source_dimensions_match": isinstance(source, Mapping)
        and source.get("first_frame_width") == source_first_frame_size[0]
        and source.get("first_frame_height") == source_first_frame_size[1],
        "provenance_hdf5_sha256_recomputed": isinstance(source, Mapping)
        and is_sha256_hex(source.get("hdf5_sha256"))
        and source.get("hdf5_sha256", "").lower() == actual_hdf5_sha256,
        "provenance_instruction_sha256_recomputed": isinstance(source, Mapping)
        and is_sha256_hex(source.get("instruction_json_sha256"))
        and source.get("instruction_json_sha256", "").lower()
        == actual_instruction_sha256,
        "provenance_first_frame_sha256_recomputed": isinstance(source, Mapping)
        and is_sha256_hex(source.get("first_frame_png_sha256"))
        and source.get("first_frame_png_sha256", "").lower()
        == actual_first_frame_sha256,
        "provenance_video_path_matches": isinstance(video, Mapping)
        and _same_resolved_path(video.get("path"), video_path),
        "provenance_video_frames_match": isinstance(video, Mapping)
        and video.get("frames") == expected_frames,
        "provenance_video_sha256_recomputed": isinstance(video, Mapping)
        and is_sha256_hex(video.get("sha256"))
        and video.get("sha256", "").lower() == video_sha256,
        "provenance_video_bytes_match": isinstance(video, Mapping)
        and video.get("bytes") == video_path.stat().st_size,
        "provenance_video_contract_matches_run": isinstance(video, Mapping)
        and isinstance(run_inputs, Mapping)
        and video.get("fps") == run_inputs.get("fps")
        and video.get("width") == run_inputs.get("output_width")
        and video.get("height") == run_inputs.get("output_height")
        and video.get("codec") == "h264"
        and video.get("encoder") == "libx264"
        and video.get("pixel_format") == "yuv420p",
        "provenance_checkpoint_matches_run": isinstance(checkpoint, Mapping)
        and is_sha256_hex(checkpoint.get("sha256"))
        and checkpoint.get("sha256", "").lower() == run_checkpoint_sha256,
        "provenance_chunk_count_matches": isinstance(rollout, Mapping)
        and rollout.get("chunk_count") == expected_chunks
        and rollout.get("keyframes_per_chunk") == KEYFRAMES_PER_CHUNK
        and rollout.get("visual_stride") == VISUAL_STRIDE
        and rollout.get("target_frame_count") == expected_frames,
        "provenance_rollout_contract_matches_run": isinstance(rollout, Mapping)
        and isinstance(run_inputs, Mapping)
        and rollout.get("native_width") == run_inputs.get("native_width")
        and rollout.get("num_inference_steps")
        == run_inputs.get("num_inference_steps")
        and rollout.get("sigma_shift") == run_inputs.get("sigma_shift")
        and rollout.get("interpolator") == run_inputs.get("interpolator")
        and rollout.get("flow_model_resize_policy")
        == run_inputs.get("flow_model_resize_policy"),
        "provenance_conditioner_matches_run": isinstance(world_model, Mapping)
        and world_model.get("conditioner") == run_conditioner,
        "provenance_zero_flow_flag_matches": isinstance(world_model, Mapping)
        and world_model.get("zero_flow_is_explicit_baseline")
        is (required_control_type == "text_driven")
        and world_model.get("flow_denoised") is False
        and world_model.get("rgb_denoised") is True
        and world_model.get("rgb_prefix_clamped_each_step") is True
        and world_model.get("flow_condition")
        == (
            "per_chunk_action_flow"
            if required_control_type == "action_driven"
            else "clean_white_zero_flow_sentinel"
        ),
        "provenance_all_sha256_fields_valid": not invalid_sha_fields,
    }
    provenance_errors = [
        f"invalid SHA-256 field: {path}" for path in invalid_sha_fields
    ]
    renderer_identity = None
    if required_control_type == "action_driven":
        if not isinstance(rollout, Mapping):
            checks["provenance_action_chunk_structure"] = False
            checks["provenance_action_digests_recomputed"] = False
            checks["provenance_renderer_verified"] = False
            checks["provenance_precomputed_manifests_verified"] = False
            provenance_errors.append("rollout metadata is missing")
        else:
            native_width = rollout.get("native_width")
            native_height = rollout.get("native_height")
            flow_source_width = rollout.get("flow_source_width", native_width)
            flow_source_height = rollout.get("flow_source_height", native_height)
            if not isinstance(native_width, int) or not isinstance(native_height, int):
                native_width, native_height = -1, -1
            if not isinstance(flow_source_width, int) or not isinstance(
                flow_source_height, int
            ):
                flow_source_width, flow_source_height = -1, -1
            with h5py.File(hdf5_path, "r") as handle:
                if ACTION_DATASET not in handle:
                    raise KeyError(f"missing {ACTION_DATASET} in {hdf5_path}")
                action_dataset = handle[ACTION_DATASET]
                if tuple(action_dataset.shape) != (expected_frames, ACTION_DIM):
                    raise ValueError(
                        f"{ACTION_DATASET} shape changed during validation: "
                        f"{tuple(action_dataset.shape)}"
                    )
                action_checks, action_errors, renderer_identity = (
                    validate_action_chunk_provenance(
                        chunks=chunks,
                        action_dataset=action_dataset,
                        episode_id=episode_id,
                        frame_count=expected_frames,
                        hdf5_path=hdf5_path,
                        native_size=(native_width, native_height),
                        flow_source_size=(flow_source_width, flow_source_height),
                        first_frame_size=source_first_frame_size,
                        expected_conditioner=(
                            run_conditioner
                            if isinstance(run_conditioner, Mapping)
                            else None
                        ),
                    )
                )
            checks.update(action_checks)
            provenance_errors.extend(action_errors)
        run_renderer_identity = run_manifest_summary.get("renderer_identity")
        checks["provenance_renderer_matches_run"] = (
            run_renderer_identity is None
            or (
                renderer_identity is not None
                and renderer_identity == run_renderer_identity
            )
        )
    else:
        starts = expected_chunk_starts(expected_frames)
        zero_chunks_ok = isinstance(chunks, list) and len(chunks) == len(starts)
        if isinstance(chunks, list):
            for chunk_index, chunk_start in enumerate(starts):
                chunk = chunks[chunk_index]
                flow = (
                    chunk.get("flow_conditioning")
                    if isinstance(chunk, Mapping)
                    else None
                )
                zero_chunks_ok = zero_chunks_ok and isinstance(flow, Mapping)
                zero_chunks_ok = zero_chunks_ok and flow.get("mode") == "zero_flow"
                zero_chunks_ok = zero_chunks_ok and flow.get("control_type") == "text_driven"
                zero_chunks_ok = zero_chunks_ok and flow.get("explicit_baseline") is True
                zero_chunks_ok = zero_chunks_ok and chunk.get("chunk_index") == chunk_index
                zero_chunks_ok = zero_chunks_ok and chunk.get(
                    "generated_frame_range"
                ) == [chunk_start, chunk_start + CHUNK_HORIZON]
        checks["provenance_zero_flow_chunks_verified"] = bool(zero_chunks_ok)
    summary = {
        "path": str(record_path),
        "run_signature": payload.get("run_signature"),
        "control_type": payload.get("control_type"),
        "conditioning_mode": payload.get("conditioning_mode"),
        "chunk_count": rollout.get("chunk_count") if isinstance(rollout, Mapping) else None,
        "renderer_identity": renderer_identity,
        "errors": provenance_errors,
    }
    return summary, checks


def main() -> None:
    args = parse_args()
    if args.episode_start < 1 or args.episode_end < args.episode_start:
        raise ValueError("invalid episode range")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.default_frame_count is not None and args.default_frame_count < 2:
        raise ValueError("--default-frame-count must be at least 2")
    if (
        args.required_control_type == "action_driven"
        and args.default_frame_count is not None
    ):
        raise ValueError(
            "action_driven validation cannot use --default-frame-count; "
            "the official HDF5 actions are mandatory"
        )
    try:
        import cv2
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "validation requires opencv-python, h5py, and numpy"
        ) from error

    dataset_root = resolve_dataset_root(args.dataset_root)
    videos_dir = args.videos_dir.expanduser().resolve()
    records_dir = (
        args.records_dir.expanduser().resolve()
        if args.records_dir is not None
        else videos_dir.parent / "per_episode"
    )
    run_manifest_path = (
        args.run_config.expanduser().resolve()
        if args.run_config is not None
        else videos_dir.parent / "run_config.json"
    )
    run_manifest_checks: dict[str, bool] = {}
    run_manifest_errors: list[str] = []
    run_manifest_summary: dict[str, Any] = {
        "path": str(run_manifest_path),
        "run_signature": None,
        "checkpoint_sha256": None,
        "conditioner": None,
        "renderer_identity": None,
    }
    if args.required_control_type != "none":
        try:
            run_manifest = load_json_mapping(run_manifest_path, "run manifest")
            (
                run_manifest_checks,
                run_manifest_errors,
                run_manifest_summary,
            ) = validate_run_manifest(
                run_manifest=run_manifest,
                run_manifest_path=run_manifest_path,
                dataset_root=dataset_root,
                required_control_type=args.required_control_type,
                expected_fps=args.expected_fps,
                expected_width=args.expected_width,
                expected_height=args.expected_height,
            )
        except Exception as error:
            run_manifest_checks = {"run_manifest_readable": False}
            run_manifest_errors = [str(error)]
    expected_names = {
        f"episode{episode_id}.mp4"
        for episode_id in range(args.episode_start, args.episode_end + 1)
    }
    actual_names = {path.name for path in videos_dir.glob("*.mp4")}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)

    def inspect_episode(episode_id: int) -> tuple[int, dict[str, Any] | None, str | None]:
        stem = f"episode{episode_id}"
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.is_file():
            return episode_id, None, None
        hdf5_path = dataset_root / "data" / TASK_DIRECTORY / f"{stem}.hdf5"
        first_frame_path = (
            dataset_root / "first_frame" / TASK_DIRECTORY / f"{stem}.png"
        )
        try:
            expected_frames, frame_count_source = trajectory_frame_count(
                hdf5_path,
                h5py=h5py,
                default_frame_count=args.default_frame_count,
            )
            record = inspect_video(
                video_path=video_path,
                first_frame_path=first_frame_path,
                expected_frames=expected_frames,
                expected_fps=args.expected_fps,
                expected_width=args.expected_width,
                expected_height=args.expected_height,
                cv2=cv2,
                np=np,
            )
            record["episode_id"] = episode_id
            record["hdf5"] = str(hdf5_path) if hdf5_path.is_file() else None
            record["frame_count_source"] = frame_count_source
            if args.required_control_type != "none":
                provenance, provenance_checks = inspect_provenance(
                    record_path=records_dir / f"{stem}.json",
                    episode_id=episode_id,
                    hdf5_path=hdf5_path,
                    instruction_path=(
                        dataset_root
                        / "instructions"
                        / TASK_DIRECTORY
                        / f"{stem}.json"
                    ),
                    first_frame_path=first_frame_path,
                    video_path=video_path,
                    expected_frames=expected_frames,
                    required_control_type=args.required_control_type,
                    run_manifest_summary=run_manifest_summary,
                    video_sha256=record["sha256"],
                    source_first_frame_size=(
                        record["source_first_frame_width"],
                        record["source_first_frame_height"],
                    ),
                    h5py=h5py,
                )
                record["provenance"] = provenance
                record["checks"].update(provenance_checks)
            return episode_id, record, None
        except Exception as error:
            return episode_id, None, f"{stem}: {error}"

    episode_ids = list(range(args.episode_start, args.episode_end + 1))
    if args.workers == 1:
        inspected = [inspect_episode(episode_id) for episode_id in episode_ids]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            inspected = list(executor.map(inspect_episode, episode_ids))

    records: list[dict[str, Any]] = []
    errors: list[str] = [
        f"run manifest: {error}" for error in run_manifest_errors
    ]
    hashes: dict[str, list[int]] = {}
    for episode_id, record, inspection_error in inspected:
        if inspection_error is not None:
            errors.append(inspection_error)
            continue
        if record is None:
            continue
        records.append(record)
        hashes.setdefault(record["sha256"], []).append(episode_id)
        if not all(record["checks"].values()):
            errors.append(f"episode{episode_id}: structural check failed")
            provenance = record.get("provenance")
            if isinstance(provenance, Mapping):
                errors.extend(
                    f"episode{episode_id}: {error}"
                    for error in provenance.get("errors", [])[:20]
                )
        if record["first_frame_psnr"] < args.minimum_first_frame_psnr:
            errors.append(
                f"episode{episode_id}: first-frame PSNR "
                f"{record['first_frame_psnr']:.3f} below threshold"
            )

    duplicate_hashes = {
        digest: episode_ids
        for digest, episode_ids in hashes.items()
        if len(episode_ids) > 1
    }
    if missing:
        errors.append(f"missing videos: {missing[:20]}")
    if extra:
        errors.append(f"extra videos: {extra[:20]}")
    if duplicate_hashes:
        errors.append(f"byte-identical episode videos: {duplicate_hashes}")
    if args.required_control_type == "action_driven":
        renderer_identities = {
            record.get("provenance", {}).get("renderer_identity")
            for record in records
            if isinstance(record.get("provenance"), Mapping)
            and record.get("provenance", {}).get("renderer_identity") is not None
        }
        if len(renderer_identities) != 1:
            errors.append(
                "action-driven renderer/URDF identity is not consistent across episodes"
            )
        run_renderer_identity = run_manifest_summary.get("renderer_identity")
        if (
            run_renderer_identity is not None
            and renderer_identities
            and renderer_identities != {run_renderer_identity}
        ):
            errors.append("episode renderer/URDF identity disagrees with run manifest")

    report = {
        "status": "pass" if not errors else "fail",
        "episode_range": [args.episode_start, args.episode_end],
        "frame_count_policy": "/joint_action/vector.shape[0]",
        "required_control_type": args.required_control_type,
        "records_dir": str(records_dir),
        "run_manifest": run_manifest_summary,
        "run_manifest_checks": run_manifest_checks,
        "default_frame_count": args.default_frame_count,
        "expected_video_count": len(expected_names),
        "actual_video_count": len(actual_names),
        "missing": missing,
        "extra": extra,
        "duplicate_hashes": duplicate_hashes,
        "errors": errors,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
