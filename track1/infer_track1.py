#!/usr/bin/env python3
"""Generate WorldArena2 Track-1 videos with KineWorld Stage-1.

The adapter discovers the complete official episode1..1000 layout, reads each
episode's frame count from ``/joint_action/vector``, rolls out as many
autoregressive 9-keyframe chunks as needed at visual_stride=4, pins the
official PNG as frame zero, and writes atomic H.264 MP4 files at 640x480/24 fps.
The default action-flow path requires real per-chunk flow and never falls back
to the explicit ``zero_flow`` text-driven baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from kineworld_stage1 import Stage1WorldModel, resolve_checkpoint, sha256_file


LOGGER = logging.getLogger("kineworld.track1")
TASK_DIRECTORY = "fixed_scene_task"
EXPECTED_EPISODES = 1000
EPISODE_RE = re.compile(r"episode([1-9][0-9]*)$")
KEYFRAMES_PER_CHUNK = 9
VISUAL_STRIDE = 4
DEFAULT_OUTPUT_WIDTH = 640
DEFAULT_OUTPUT_HEIGHT = 480
DEFAULT_OUTPUT_FPS = 24
SCHEMA_VERSION = 1
ADAPTER_VERSION = "kineworld-track1-stage1-v1"
FLOW_MODEL_RESIZE_POLICY = "pil_rgb_default_resize_matching_training_online_v1"


@dataclass(frozen=True)
class EpisodeFiles:
    episode_id: int
    stem: str
    hdf5: Path | None
    png: Path
    instruction: Path


class LinearFrameInterpolator:
    """Deterministic RGB-space interpolation behind a replaceable interface."""

    name = "linear_rgb_v1"

    def expand(self, keyframes: Sequence[Any], stride: int) -> list[Any]:
        import numpy as np
        from PIL import Image

        if len(keyframes) < 2:
            raise ValueError("at least two keyframes are required")
        if stride < 1:
            raise ValueError("stride must be positive")
        size = keyframes[0].size
        if any(frame.size != size for frame in keyframes):
            raise ValueError("all keyframes must have the same size")

        output: list[Any] = []
        for left, right in zip(keyframes[:-1], keyframes[1:]):
            left_rgb = left.convert("RGB")
            right_rgb = right.convert("RGB")
            output.append(left_rgb.copy())
            if stride == 1:
                continue
            left_array = np.asarray(left_rgb, dtype=np.float32)
            right_array = np.asarray(right_rgb, dtype=np.float32)
            for offset in range(1, stride):
                alpha = offset / stride
                blended = np.rint(
                    left_array * (1.0 - alpha) + right_array * alpha
                ).clip(0, 255).astype(np.uint8)
                output.append(Image.fromarray(blended, mode="RGB"))
        output.append(keyframes[-1].convert("RGB").copy())
        expected = (len(keyframes) - 1) * stride + 1
        if len(output) != expected:
            raise AssertionError((len(output), expected))
        return output


def environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def parse_args() -> argparse.Namespace:
    local_rank = environment_int("LOCAL_RANK", 0)
    parser = argparse.ArgumentParser(
        description="KineWorld Stage-1 offline generator for WorldArena2 Track-1"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="directory containing data/, first_frame/, instructions/ (or its parent)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="run directory; videos are written under output-root/videos",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="local KineWorld .safetensors checkpoint",
    )
    parser.add_argument("--checkpoint-repo", help="explicit Hugging Face repository")
    parser.add_argument(
        "--checkpoint-file",
        help="checkpoint filename in --checkpoint-repo",
    )
    parser.add_argument("--checkpoint-revision")
    parser.add_argument(
        "--base-model-id", default="Wan-AI/Wan2.2-TI2V-5B"
    )
    parser.add_argument(
        "--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B"
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument("--device", default=f"cuda:{local_rank}")
    parser.add_argument(
        "--conditioning-mode",
        choices=("action_flow", "zero_flow"),
        default="action_flow",
        help=(
            "action_flow is the fail-closed Track-1 path; zero_flow is an "
            "explicit text-driven baseline"
        ),
    )
    parser.add_argument(
        "--action-flow-provider",
        help="provider name or import spec understood by action_flow_conditioning",
    )
    parser.add_argument(
        "--precomputed-flow-root",
        type=Path,
        help="root containing precomputed per-episode action-flow chunks",
    )
    parser.add_argument(
        "--action-flow-provider-kwargs",
        default="{}",
        help="JSON object forwarded to build_action_flow_conditioner",
    )
    parser.add_argument("--episode-start", type=int, default=1)
    parser.add_argument("--episode-end", type=int, default=EXPECTED_EPISODES)
    parser.add_argument(
        "--num-shards", type=int, default=environment_int("WORLD_SIZE", 1)
    )
    parser.add_argument(
        "--shard-index", type=int, default=environment_int("RANK", 0)
    )
    parser.add_argument(
        "--default-frame-count",
        type=int,
        help=(
            "explicit fallback only when an episode HDF5 file is absent; "
            "official inputs should omit this and must provide /joint_action/vector"
        ),
    )
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_OUTPUT_FPS)
    parser.add_argument("--native-width", type=int, default=320)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--interpolator", choices=("linear",), default="linear"
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video-crf", type=int, default=17)
    parser.add_argument(
        "--video-preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
        ),
        default="medium",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace outputs whose provenance does not match this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate dataset discovery and shard assignment without loading a model",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dry_run:
        local_checkpoint = args.checkpoint_path is not None
        remote_checkpoint = (
            args.checkpoint_repo is not None and args.checkpoint_file is not None
        )
        if local_checkpoint == remote_checkpoint:
            raise ValueError(
                "provide exactly one checkpoint source: --checkpoint-path or "
                "both --checkpoint-repo and --checkpoint-file"
            )
        if local_checkpoint and (
            args.checkpoint_repo is not None or args.checkpoint_file is not None
        ):
            raise ValueError("--checkpoint-path cannot be combined with remote checkpoint options")
    if args.episode_start < 1 or args.episode_end > EXPECTED_EPISODES:
        raise ValueError("episode range must stay within 1..1000")
    if args.episode_end < args.episode_start:
        raise ValueError("--episode-end must be >= --episode-start")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    if args.default_frame_count is not None and args.default_frame_count < 2:
        raise ValueError("--default-frame-count must be at least 2")
    try:
        provider_kwargs = json.loads(args.action_flow_provider_kwargs)
    except json.JSONDecodeError as error:
        raise ValueError("--action-flow-provider-kwargs must be valid JSON") from error
    if not isinstance(provider_kwargs, dict):
        raise ValueError("--action-flow-provider-kwargs must decode to an object")
    args.action_flow_provider_kwargs_parsed = provider_kwargs
    if args.conditioning_mode == "action_flow":
        if args.default_frame_count is not None:
            raise ValueError(
                "action_flow cannot use --default-frame-count because real HDF5 "
                "actions are mandatory"
            )
        if (
            not args.dry_run
            and args.action_flow_provider is None
            and args.precomputed_flow_root is None
        ):
            raise ValueError(
                "action_flow requires --action-flow-provider or "
                "--precomputed-flow-root; use --conditioning-mode zero_flow "
                "only for the explicit text-driven baseline"
            )
    elif args.action_flow_provider is not None or args.precomputed_flow_root is not None:
        raise ValueError("action-flow provider options require --conditioning-mode action_flow")
    if args.output_width < 1 or args.output_height < 1 or args.fps < 1:
        raise ValueError("output dimensions and fps must be positive")
    if args.native_width < 16:
        raise ValueError("--native-width must be at least 16")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.video_crf < 0 or args.video_crf > 51:
        raise ValueError("--video-crf must be in 0..51")


def resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = (path, path / "dataset_track1")
    for candidate in candidates:
        if all(
            (candidate / name / TASK_DIRECTORY).is_dir()
            for name in ("data", "first_frame", "instructions")
        ):
            return candidate
    raise FileNotFoundError(
        f"expected data/, first_frame/, instructions/ under {path}"
    )


def discover_episodes(
    dataset_root: Path, *, allow_missing_hdf5: bool = False
) -> list[EpisodeFiles]:
    directories = {
        "hdf5": dataset_root / "data" / TASK_DIRECTORY,
        "png": dataset_root / "first_frame" / TASK_DIRECTORY,
        "instruction": dataset_root / "instructions" / TASK_DIRECTORY,
    }
    extensions = {"hdf5": ".hdf5", "png": ".png", "instruction": ".json"}
    by_kind: dict[str, dict[int, Path]] = {}
    for kind, directory in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        mapping: dict[int, Path] = {}
        for path in directory.glob(f"*{extensions[kind]}"):
            match = EPISODE_RE.fullmatch(path.stem)
            if match is None:
                raise ValueError(f"unexpected Track-1 filename: {path}")
            episode_id = int(match.group(1))
            if episode_id in mapping:
                raise ValueError(f"duplicate episode {episode_id} in {directory}")
            mapping[episode_id] = path.resolve()
        by_kind[kind] = mapping

    expected_ids = set(range(1, EXPECTED_EPISODES + 1))
    for kind, mapping in by_kind.items():
        actual_ids = set(mapping)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        mismatch_is_allowed = kind == "hdf5" and allow_missing_hdf5 and not extra
        if actual_ids != expected_ids and not mismatch_is_allowed:
            raise ValueError(
                f"{kind} episode set mismatch: "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )

    return [
        EpisodeFiles(
            episode_id=episode_id,
            stem=f"episode{episode_id}",
            hdf5=by_kind["hdf5"].get(episode_id),
            png=by_kind["png"][episode_id],
            instruction=by_kind["instruction"][episode_id],
        )
        for episode_id in range(1, EXPECTED_EPISODES + 1)
    ]


def load_instruction(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    instruction = payload.get("instruction") if isinstance(payload, Mapping) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"missing instruction string in {path}")
    return instruction.strip()


def source_trajectory_frame_count(
    path: Path | None,
    *,
    default_frame_count: int | None,
    require_action_14d: bool,
) -> tuple[int, str]:
    if path is None:
        if default_frame_count is None:
            raise FileNotFoundError("episode HDF5 is missing")
        return default_frame_count, "default_frame_count_fallback"
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to inspect Track-1 HDF5 inputs") from error

    with h5py.File(path, "r") as handle:
        key = "joint_action/vector"
        if key not in handle:
            raise KeyError(f"missing /{key} in {path}")
        dataset = handle[key]
        if not dataset.shape:
            raise ValueError(f"/{key} must have a frame dimension in {path}")
        if require_action_14d and (
            len(dataset.shape) != 2 or int(dataset.shape[1]) != 14
        ):
            raise ValueError(
                f"action_flow requires /{key} shape [N,14], got "
                f"{tuple(dataset.shape)} in {path}"
            )
        frame_count = int(dataset.shape[0])
        if frame_count < 2:
            raise ValueError(f"invalid /{key} frame count {frame_count} in {path}")
        return frame_count, "/joint_action/vector"


def build_action_conditioner(args: argparse.Namespace) -> Any:
    """Load the optional provider only for the strict action-flow path."""
    if args.conditioning_mode != "action_flow":
        return None
    try:
        from action_flow_conditioning import build_action_flow_conditioner
    except ImportError as error:
        raise RuntimeError(
            "action_flow_conditioning.py is unavailable; action_flow refuses "
            "to fall back to white flow"
        ) from error
    conditioner = build_action_flow_conditioner(
        provider_spec=args.action_flow_provider,
        precomputed_root=(
            args.precomputed_flow_root.expanduser().resolve()
            if args.precomputed_flow_root is not None
            else None
        ),
        provider_kwargs=args.action_flow_provider_kwargs_parsed,
    )
    if conditioner is None or not callable(
        getattr(conditioner, "get_chunk_flow", None)
    ):
        raise TypeError(
            "build_action_flow_conditioner must return an object with "
            "get_chunk_flow(...)"
        )
    return conditioner


def conditioner_description(conditioner: Any) -> Mapping[str, Any] | None:
    if conditioner is None:
        return None
    describe = getattr(conditioner, "describe", None)
    description = describe() if callable(describe) else {
        "class": f"{type(conditioner).__module__}.{type(conditioner).__qualname__}"
    }
    if not isinstance(description, Mapping):
        raise TypeError("action-flow conditioner describe() must return a mapping")
    # Provenance must be JSON serializable before the expensive rollout starts.
    json.dumps(description, ensure_ascii=False, sort_keys=True)
    return dict(description)


def close_action_conditioner(conditioner: Any) -> None:
    if conditioner is None:
        return
    close = getattr(conditioner, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            LOGGER.exception("action-flow conditioner cleanup failed")


def normalize_chunk_flow_result(
    result: Any,
) -> tuple[Sequence[Any] | None, Any | None, Mapping[str, Any]]:
    if isinstance(result, Mapping):
        frames = result.get("frames")
        latents = result.get("latents")
        provenance = result.get("provenance", {})
    else:
        frames = getattr(result, "frames", None)
        latents = getattr(result, "latents", None)
        provenance = getattr(result, "provenance", {})
    if (frames is None) == (latents is None):
        raise ValueError(
            "action-flow get_chunk_flow must return exactly one of frames or latents"
        )
    if frames is not None and (
        isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence)
    ):
        raise TypeError("action-flow frames must be a sequence of PIL images")
    if not isinstance(provenance, Mapping):
        raise TypeError("action-flow chunk provenance must be a mapping")
    json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    return frames, latents, dict(provenance)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_chunk_seed(base_seed: int, episode_id: int, chunk_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:episode{episode_id}:chunk{chunk_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def find_ffmpeg(executable: str) -> str:
    explicit = Path(executable)
    if explicit.parent != Path(".") or explicit.is_absolute():
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return str(explicit.resolve())
    resolved = shutil.which(executable)
    if resolved is None and executable == "ffmpeg":
        try:
            import imageio_ffmpeg

            bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if bundled.is_file():
                resolved = str(bundled.resolve())
        except (ImportError, RuntimeError):
            resolved = None
    if resolved is None:
        raise FileNotFoundError(
            f"could not find {executable!r}; pass --ffmpeg with an FFmpeg binary"
        )
    return resolved


def ffmpeg_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=15,
        check=False,
    )
    first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    return first_line.strip()


def write_h264_mp4(
    *,
    destination: Path,
    frames: Sequence[Any],
    width: int,
    height: int,
    fps: int,
    ffmpeg: str,
    crf: int,
    preset: str,
    first_frame: Any,
) -> None:
    from PIL import Image

    if not frames:
        raise ValueError("cannot encode an empty video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.mp4.part"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-frames:v",
        str(len(frames)),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temporary),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("failed to open FFmpeg pipes")
        try:
            for frame_index, frame in enumerate(frames):
                source = first_frame if frame_index == 0 else frame
                encoded_frame = source.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                )
                process.stdin.write(encoded_frame.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        error_text = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg exited with status {return_code}: {error_text.strip()}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced no MP4 output")
        os.replace(temporary, destination)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()


def rollout_episode(
    *,
    model: Stage1WorldModel,
    first_frame: Any,
    instruction: str,
    target_frames: int,
    native_width: int,
    num_inference_steps: int,
    sigma_shift: float,
    base_seed: int,
    episode_id: int,
    interpolator: LinearFrameInterpolator,
    conditioning_mode: str,
    action_conditioner: Any,
    hdf5_path: Path | None,
) -> tuple[
    list[Any],
    list[dict[str, Any]],
    tuple[int, int],
    tuple[int, int],
]:
    chunk_horizon = (KEYFRAMES_PER_CHUNK - 1) * VISUAL_STRIDE
    chunk_starts = list(range(0, target_frames - 1, chunk_horizon))
    prepared = None
    output: list[Any] = []
    chunk_records: list[dict[str, Any]] = []
    current = first_frame.convert("RGB")
    source_width, source_height = first_frame.size
    flow_source_size = (
        native_width,
        max(1, round(source_height * native_width / source_width)),
    )
    try:
        prepared = model.prepare_episode(
            first_frame=first_frame,
            instruction=instruction,
            num_frames=KEYFRAMES_PER_CHUNK,
            native_width=native_width,
            conditioning_mode=conditioning_mode,
        )
        for chunk_index, chunk_start in enumerate(chunk_starts):
            seed = deterministic_chunk_seed(base_seed, episode_id, chunk_index)
            flow_frames = None
            flow_latents = None
            if conditioning_mode == "action_flow":
                if action_conditioner is None or hdf5_path is None:
                    raise RuntimeError(
                        "action_flow requires a conditioner and episode HDF5"
                    )
                flow_result = action_conditioner.get_chunk_flow(
                    episode_id=episode_id,
                    hdf5_path=hdf5_path,
                    first_frame=first_frame,
                    frame_count=target_frames,
                    chunk_start=chunk_start,
                    keyframe_count=KEYFRAMES_PER_CHUNK,
                    visual_stride=VISUAL_STRIDE,
                    target_size=flow_source_size,
                )
                flow_frames, flow_latents, flow_provenance = (
                    normalize_chunk_flow_result(flow_result)
                )
                model_size = (prepared.width, prepared.height)
                resize_policy = "identity"
                if flow_source_size != model_size:
                    if flow_frames is None:
                        raise ValueError(
                            "pre-encoded action-flow latents cannot be resized from "
                            f"provider size {flow_source_size} to model size {model_size}"
                        )
                    # Match the formal online training path exactly:
                    # flow_action_train.py calls RGB PIL frame.resize((flow_w,
                    # flow_h)) after Wan's 32-pixel alignment check.  Keep the
                    # manifested 320x240 flow bytes immutable and adapt only
                    # the in-memory model input to 320x256.
                    flow_frames = tuple(
                        frame.resize(model_size) for frame in flow_frames
                    )
                    resize_policy = FLOW_MODEL_RESIZE_POLICY
                flow_model_adapter = {
                    "source_size": [flow_source_size[0], flow_source_size[1]],
                    "target_size": [model_size[0], model_size[1]],
                    "resize_policy": resize_policy,
                }
            else:
                flow_provenance = {
                    "mode": "zero_flow",
                    "control_type": "text_driven",
                    "explicit_baseline": True,
                }
                flow_source_size = (prepared.width, prepared.height)
                flow_model_adapter = {
                    "source_size": [prepared.width, prepared.height],
                    "target_size": [prepared.width, prepared.height],
                    "resize_policy": "identity",
                }
            keyframes = model.generate_keyframe_chunk(
                conditioning_frame=current,
                prepared=prepared,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                flow_frames=flow_frames,
                flow_clean_latents=flow_latents,
            )
            if len(keyframes) != KEYFRAMES_PER_CHUNK:
                raise RuntimeError(
                    f"chunk {chunk_index} returned {len(keyframes)} keyframes"
                )
            expanded = interpolator.expand(keyframes, VISUAL_STRIDE)
            expected_chunk_frames = chunk_horizon + 1
            if len(expanded) != expected_chunk_frames:
                raise AssertionError((len(expanded), expected_chunk_frames))
            if chunk_index == 0:
                output.extend(expanded)
            else:
                output.extend(expanded[1:])
            current = keyframes[-1]
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "seed": seed,
                    "keyframe_count": len(keyframes),
                    "generated_frame_range": [
                        chunk_start,
                        chunk_start + chunk_horizon,
                    ],
                    "consumed_frame_range": [
                        chunk_start,
                        min(chunk_start + chunk_horizon, target_frames - 1),
                    ],
                    "flow_conditioning": flow_provenance,
                    "flow_model_adapter": flow_model_adapter,
                }
            )
            LOGGER.info(
                "episode%d chunk %d/%d consumed frames %d..%d",
                episode_id,
                chunk_index + 1,
                len(chunk_starts),
                chunk_start,
                min(chunk_start + chunk_horizon, target_frames - 1),
            )
        output = output[:target_frames]
        if len(output) != target_frames:
            raise RuntimeError(
                f"rollout produced {len(output)} frames; expected {target_frames}"
            )
        return (
            output,
            chunk_records,
            (prepared.width, prepared.height),
            flow_source_size,
        )
    finally:
        try:
            model.release_episode(prepared)
        finally:
            if action_conditioner is not None:
                release = getattr(action_conditioner, "release_episode", None)
                if callable(release):
                    release(episode_id=episode_id)


def existing_output_state(
    *,
    video_path: Path,
    record_path: Path,
    run_signature: str,
    expected_frame_count: int,
    overwrite: bool,
) -> str:
    record: Mapping[str, Any] | None = None
    if record_path.is_file():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                record = loaded
        except (OSError, json.JSONDecodeError):
            record = None
    record_source = record.get("source") if record is not None else None
    record_video = record.get("video") if record is not None else None
    if (
        video_path.is_file()
        and record is not None
        and record.get("status") == "complete"
        and record.get("run_signature") == run_signature
        and isinstance(record_source, Mapping)
        and record_source.get("trajectory_frame_count") == expected_frame_count
        and isinstance(record_video, Mapping)
        and record_video.get("frames") == expected_frame_count
    ):
        return "skip"
    if video_path.exists() and not overwrite:
        raise FileExistsError(
            f"existing output has missing/mismatched provenance: {video_path}; "
            "pass --overwrite to replace it"
        )
    if record_path.exists() and not overwrite and record is not None:
        previous_signature = record.get("run_signature")
        if previous_signature not in (None, run_signature):
            raise FileExistsError(
                f"existing provenance belongs to another run: {record_path}; "
                "pass --overwrite to replace it"
            )
    return "generate"


def write_status(
    path: Path,
    *,
    status: str,
    run_signature: str,
    args: argparse.Namespace,
    assigned_count: int,
    completed: int,
    skipped: int,
    current_episode: int | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_signature": run_signature,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_episode_count": assigned_count,
        "completed_episode_count": completed,
        "skipped_episode_count": skipped,
        "updated_unix_time": time.time(),
    }
    if current_episode is not None:
        payload["current_episode"] = current_episode
    if error is not None:
        payload["error"] = error
    atomic_json(path, payload)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate_args(args)
    dataset_root = resolve_dataset_root(args.dataset_root)
    episodes = discover_episodes(
        dataset_root,
        allow_missing_hdf5=(
            args.conditioning_mode == "zero_flow"
            and args.default_frame_count is not None
        ),
    )
    selected = [
        item
        for item in episodes
        if args.episode_start <= item.episode_id <= args.episode_end
        and (item.episode_id - args.episode_start) % args.num_shards
        == args.shard_index
    ]
    assignment = {
        "event": "dataset_discovered",
        "dataset_root": str(dataset_root),
        "official_episode_count": len(episodes),
        "episode_range": [args.episode_start, args.episode_end],
        "shard": [args.shard_index, args.num_shards],
        "assigned_episode_count": len(selected),
        "assigned_first": selected[0].episode_id if selected else None,
        "assigned_last": selected[-1].episode_id if selected else None,
        "missing_hdf5_count": sum(item.hdf5 is None for item in episodes),
        "default_frame_count": args.default_frame_count,
        "conditioning_mode": args.conditioning_mode,
    }
    print(json.dumps(assignment, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    # Fail on missing/invalid official trajectory lengths before downloading
    # multi-gigabyte checkpoints or constructing the GPU pipeline.
    frame_counts: dict[int, tuple[int, str]] = {}
    for item in selected:
        frame_counts[item.episode_id] = source_trajectory_frame_count(
            item.hdf5,
            default_frame_count=args.default_frame_count,
            require_action_14d=args.conditioning_mode == "action_flow",
        )

    action_conditioner = build_action_conditioner(args)
    action_conditioner_provenance = conditioner_description(action_conditioner)

    ffmpeg = find_ffmpeg(args.ffmpeg)
    checkpoint_path, checkpoint_source = resolve_checkpoint(
        checkpoint_path=args.checkpoint_path,
        checkpoint_repo=args.checkpoint_repo,
        checkpoint_file=args.checkpoint_file,
        checkpoint_revision=args.checkpoint_revision,
        cache_dir=args.model_cache_dir.expanduser().resolve() / "huggingface",
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    signature_inputs = {
        "adapter_version": ADAPTER_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "base_model_id": args.base_model_id,
        "tokenizer_model_id": args.tokenizer_model_id,
        "world_model_mode": "clean_flow_condition_rgb_only_denoising",
        "prompt_policy": "plain_instruction_no_t_shape_prefix",
        "conditioning_mode": args.conditioning_mode,
        "control_type": (
            "action_driven"
            if args.conditioning_mode == "action_flow"
            else "text_driven"
        ),
        "action_flow_conditioner": action_conditioner_provenance,
        "action_flow_provider_spec": args.action_flow_provider,
        "precomputed_flow_root": (
            str(args.precomputed_flow_root.expanduser().resolve())
            if args.precomputed_flow_root is not None
            else None
        ),
        "action_flow_provider_kwargs_sha256": canonical_signature(
            args.action_flow_provider_kwargs_parsed
        ),
        "keyframes_per_chunk": KEYFRAMES_PER_CHUNK,
        "visual_stride": VISUAL_STRIDE,
        "frame_count_source": "/joint_action/vector",
        "default_frame_count": args.default_frame_count,
        "output_width": args.output_width,
        "output_height": args.output_height,
        "fps": args.fps,
        "native_width": args.native_width,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "base_seed": args.seed,
        "interpolator": LinearFrameInterpolator.name,
        "flow_model_resize_policy": FLOW_MODEL_RESIZE_POLICY,
        "video_codec": "h264/libx264/yuv420p",
        "video_crf": args.video_crf,
        "video_preset": args.video_preset,
    }
    run_signature = canonical_signature(signature_inputs)

    output_root = args.output_root.expanduser().resolve()
    videos_dir = output_root / "videos"
    records_dir = output_root / "per_episode"
    status_dir = output_root / "status"
    for directory in (videos_dir, records_dir, status_dir):
        directory.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / (
        f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    run_config_path = output_root / "run_config.json"
    if run_config_path.is_file():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous.get("run_signature") != run_signature and not args.overwrite:
            raise FileExistsError(
                f"{run_config_path} belongs to a different run; pass --overwrite"
            )
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "signature_inputs": signature_inputs,
        "dataset_root": str(dataset_root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "source": checkpoint_source,
            "sha256": checkpoint_sha256,
        },
        "conditioning": {
            "mode": args.conditioning_mode,
            "control_type": signature_inputs["control_type"],
            "conditioner": action_conditioner_provenance,
            "zero_flow_is_explicit_baseline": args.conditioning_mode == "zero_flow",
        },
        "ffmpeg": {"path": ffmpeg, "version": ffmpeg_version(ffmpeg)},
    }
    atomic_json(run_config_path, run_config)

    pending: list[EpisodeFiles] = []
    skipped = 0
    try:
        for item in selected:
            frame_count, _frame_count_source = frame_counts[item.episode_id]
            state = existing_output_state(
                video_path=videos_dir / f"{item.stem}.mp4",
                record_path=records_dir / f"{item.stem}.json",
                run_signature=run_signature,
                expected_frame_count=frame_count,
                overwrite=args.overwrite,
            )
            if state == "skip":
                skipped += 1
            else:
                pending.append(item)
    except BaseException:
        close_action_conditioner(action_conditioner)
        write_status(
            status_path,
            status="failed",
            run_signature=run_signature,
            args=args,
            assigned_count=len(selected),
            completed=0,
            skipped=skipped,
            error=traceback.format_exc(),
        )
        raise

    write_status(
        status_path,
        status="running" if pending else "complete",
        run_signature=run_signature,
        args=args,
        assigned_count=len(selected),
        completed=0,
        skipped=skipped,
    )
    if not pending:
        LOGGER.info("All %d assigned episodes already complete", len(selected))
        close_action_conditioner(action_conditioner)
        return

    try:
        model = Stage1WorldModel(
            checkpoint_path=checkpoint_path,
            base_model_id=args.base_model_id,
            tokenizer_model_id=args.tokenizer_model_id,
            model_cache_dir=args.model_cache_dir,
            device=args.device,
            checkpoint_sha256=checkpoint_sha256,
        )
    except BaseException:
        close_action_conditioner(action_conditioner)
        write_status(
            status_path,
            status="failed",
            run_signature=run_signature,
            args=args,
            assigned_count=len(selected),
            completed=0,
            skipped=skipped,
            error=traceback.format_exc(),
        )
        raise

    interpolator = LinearFrameInterpolator()
    completed = 0
    current_episode: int | None = None
    try:
        from PIL import Image

        for local_index, item in enumerate(pending):
            current_episode = item.episode_id
            episode_started = time.time()
            instruction = load_instruction(item.instruction)
            source_frames, frame_count_source = frame_counts[item.episode_id]
            with Image.open(item.png) as image:
                first_frame = image.convert("RGB").copy()
            source_width, source_height = first_frame.size

            (
                native_frames,
                chunk_records,
                native_size,
                flow_source_size,
            ) = rollout_episode(
                model=model,
                first_frame=first_frame,
                instruction=instruction,
                target_frames=source_frames,
                native_width=args.native_width,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                base_seed=args.seed,
                episode_id=item.episode_id,
                interpolator=interpolator,
                conditioning_mode=args.conditioning_mode,
                action_conditioner=action_conditioner,
                hdf5_path=item.hdf5,
            )
            video_path = videos_dir / f"{item.stem}.mp4"
            write_h264_mp4(
                destination=video_path,
                frames=native_frames,
                width=args.output_width,
                height=args.output_height,
                fps=args.fps,
                ffmpeg=ffmpeg,
                crf=args.video_crf,
                preset=args.video_preset,
                first_frame=first_frame,
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "run_signature": run_signature,
                "adapter_version": ADAPTER_VERSION,
                "episode_id": item.episode_id,
                "episode_stem": item.stem,
                "instruction": instruction,
                "prompt_policy": "plain_instruction_no_t_shape_prefix",
                "conditioning_mode": args.conditioning_mode,
                "control_type": signature_inputs["control_type"],
                "world_model_mode": {
                    "flow_condition": (
                        "per_chunk_action_flow"
                        if args.conditioning_mode == "action_flow"
                        else "clean_white_zero_flow_sentinel"
                    ),
                    "flow_denoised": False,
                    "rgb_denoised": True,
                    "rgb_prefix_clamped_each_step": True,
                    "zero_flow_is_explicit_baseline": (
                        args.conditioning_mode == "zero_flow"
                    ),
                    "conditioner": action_conditioner_provenance,
                    "flow_model_resize_policy": FLOW_MODEL_RESIZE_POLICY,
                },
                "source": {
                    "hdf5": str(item.hdf5) if item.hdf5 is not None else None,
                    "hdf5_sha256": (
                        sha256_file(item.hdf5) if item.hdf5 is not None else None
                    ),
                    "trajectory_frame_count": source_frames,
                    "frame_count_source": frame_count_source,
                    "instruction_json": str(item.instruction),
                    "instruction_json_sha256": sha256_file(item.instruction),
                    "first_frame_png": str(item.png),
                    "first_frame_png_sha256": sha256_file(item.png),
                    "first_frame_width": source_width,
                    "first_frame_height": source_height,
                },
                "rollout": {
                    "keyframes_per_chunk": KEYFRAMES_PER_CHUNK,
                    "visual_stride": VISUAL_STRIDE,
                    "chunk_count": len(chunk_records),
                    "chunks": chunk_records,
                    "interpolator": interpolator.name,
                    "target_frame_count": source_frames,
                    "native_width": native_size[0],
                    "native_height": native_size[1],
                    "flow_source_width": flow_source_size[0],
                    "flow_source_height": flow_source_size[1],
                    "flow_model_resize_policy": FLOW_MODEL_RESIZE_POLICY,
                    "num_inference_steps": args.num_inference_steps,
                    "sigma_shift": args.sigma_shift,
                },
                "first_frame": {
                    "official_png_pinned_before_video_encoding": True,
                    "output_frame_index": 0,
                },
                "video": {
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                    "bytes": video_path.stat().st_size,
                    "frames": source_frames,
                    "fps": args.fps,
                    "width": args.output_width,
                    "height": args.output_height,
                    "codec": "h264",
                    "encoder": "libx264",
                    "pixel_format": "yuv420p",
                    "atomic_write": True,
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "source": checkpoint_source,
                    "sha256": checkpoint_sha256,
                    "load_report": model.load_report,
                },
                "elapsed_seconds": round(time.time() - episode_started, 3),
            }
            atomic_json(records_dir / f"{item.stem}.json", result)
            completed += 1
            write_status(
                status_path,
                status="running",
                run_signature=run_signature,
                args=args,
                assigned_count=len(selected),
                completed=completed,
                skipped=skipped,
                current_episode=item.episode_id,
            )
            print(
                json.dumps(
                    {
                        "event": "episode_complete",
                        "episode": item.episode_id,
                        "shard": args.shard_index,
                        "local_progress": f"{local_index + 1}/{len(pending)}",
                        "elapsed_seconds": result["elapsed_seconds"],
                        "video": str(video_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            # Do not retain a long episode's decoded frame list while the next
            # episode begins its model rollout.
            del native_frames, first_frame

        write_status(
            status_path,
            status="complete",
            run_signature=run_signature,
            args=args,
            assigned_count=len(selected),
            completed=completed,
            skipped=skipped,
        )
        close_action_conditioner(action_conditioner)
    except BaseException:
        error_text = traceback.format_exc()
        close_action_conditioner(action_conditioner)
        write_status(
            status_path,
            status="failed",
            run_signature=run_signature,
            args=args,
            assigned_count=len(selected),
            completed=completed,
            skipped=skipped,
            current_episode=current_episode,
            error=error_text,
        )
        raise


if __name__ == "__main__":
    main()
