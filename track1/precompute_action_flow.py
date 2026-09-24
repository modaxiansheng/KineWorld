#!/usr/bin/env python3
"""Precompute strict action-driven KineWorld conditions for Track-1.

The output layout and manifest are deliberately identical to the contract
consumed by :class:`action_flow_conditioning.PrecomputedActionFlowConditioner`.
There is no zero-flow mode.  A manifest is the commit record for a chunk: all
PNG files are atomically replaced first and the manifest is atomically written
last, so an interrupted chunk can never be mistaken for a completed one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

try:
    from .action_flow_conditioning import (
        ACTION_DATASET,
        ACTION_DIM,
        BUILTIN_PROVIDER_NAMES,
        DEFAULT_KEYFRAME_COUNT,
        DEFAULT_VISUAL_STRIDE,
        PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
        PrecomputedActionFlowConditioner,
        _action_chunk_sha256,
        _atomic_json,
        _json_mapping,
        _read_action_chunk,
        _sha256_file,
        build_action_flow_conditioner,
        chunk_source_indices,
    )
except ImportError:  # Direct ``python track1/precompute_action_flow.py`` use.
    from action_flow_conditioning import (  # type: ignore[no-redef]
        ACTION_DATASET,
        ACTION_DIM,
        BUILTIN_PROVIDER_NAMES,
        DEFAULT_KEYFRAME_COUNT,
        DEFAULT_VISUAL_STRIDE,
        PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
        PrecomputedActionFlowConditioner,
        _action_chunk_sha256,
        _atomic_json,
        _json_mapping,
        _read_action_chunk,
        _sha256_file,
        build_action_flow_conditioner,
        chunk_source_indices,
    )


EXPECTED_EPISODES = 1000


class PrecomputeError(RuntimeError):
    """Fail-closed error raised before an output manifest is committed."""


def resolve_dataset_root(path: Path) -> Path:
    """Resolve either ``dataset_track1`` or its parent directory."""

    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "dataset_track1"):
        if all((candidate / kind).is_dir() for kind in ("data", "first_frame", "instructions")):
            try:
                resolve_task_directory(candidate)
            except (FileNotFoundError, PrecomputeError):
                continue
            return candidate
    raise FileNotFoundError(
        f"expected data/, first_frame/, instructions/ under {path}"
    )


def resolve_task_directory(dataset_root: Path) -> str:
    """Return the one task directory shared by every official input kind.

    Track-1 releases have used names such as ``fixed_scene_task``; the name is
    not part of the model contract and must not be guessed.  Requiring the
    three directory sets to match also fails closed on a partial extraction.
    """

    directory_sets: dict[str, set[str]] = {}
    for kind in ("data", "first_frame", "instructions"):
        root = Path(dataset_root) / kind
        if not root.is_dir():
            raise FileNotFoundError(root)
        directory_sets[kind] = {entry.name for entry in root.iterdir() if entry.is_dir()}
    values = list(directory_sets.values())
    if not values[0] or any(value != values[0] for value in values[1:]):
        raise PrecomputeError(
            "Track-1 task directories disagree across data/first_frame/instructions: "
            f"{directory_sets}"
        )
    if len(values[0]) != 1:
        raise PrecomputeError(
            f"expected exactly one Track-1 task directory, got {sorted(values[0])}"
        )
    return next(iter(values[0]))


def episode_paths(
    dataset_root: Path, episode_id: int, *, task_directory: str | None = None
) -> tuple[Path, Path]:
    if task_directory is None:
        task_directory = resolve_task_directory(dataset_root)
    hdf5_path = (
        dataset_root / "data" / task_directory / f"episode{episode_id}.hdf5"
    ).resolve()
    first_frame_path = (
        dataset_root
        / "first_frame"
        / task_directory
        / f"episode{episode_id}.png"
    ).resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    if not first_frame_path.is_file():
        raise FileNotFoundError(first_frame_path)
    return hdf5_path, first_frame_path


def select_episode_ids(
    *, start: int, end: int, shard_index: int, num_shards: int
) -> list[int]:
    if not 1 <= start <= end <= EXPECTED_EPISODES:
        raise ValueError(
            f"episode range must satisfy 1 <= start <= end <= {EXPECTED_EPISODES}"
        )
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    # Sharding is stable across invocations and balanced to within one episode.
    return [
        episode_id
        for episode_id in range(start, end + 1)
        if (episode_id - start) % num_shards == shard_index
    ]


def trajectory_frame_count(hdf5_path: Path) -> int:
    try:
        import h5py
    except ImportError as error:
        raise PrecomputeError("h5py is required to inspect Track-1 actions") from error
    with h5py.File(hdf5_path, "r") as handle:
        if ACTION_DATASET not in handle:
            raise PrecomputeError(f"missing {ACTION_DATASET} in {hdf5_path}")
        shape = tuple(handle[ACTION_DATASET].shape)
    if len(shape) != 2 or shape[1] != ACTION_DIM or shape[0] < 2:
        raise PrecomputeError(
            f"{ACTION_DATASET} must be [N,{ACTION_DIM}] with N>=2, got {shape}"
        )
    return int(shape[0])


def chunk_starts(
    frame_count: int,
    *,
    keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
    visual_stride: int = DEFAULT_VISUAL_STRIDE,
) -> list[int]:
    if keyframe_count < 2 or visual_stride < 1:
        raise ValueError("keyframe_count must be >=2 and visual_stride must be positive")
    if frame_count < 2:
        raise ValueError("frame_count must be >=2")
    horizon = (keyframe_count - 1) * visual_stride
    return list(range(0, frame_count - 1, horizon))


def parse_provider_kwargs(raw: str | None) -> dict[str, Any]:
    """Parse a JSON object, accepting ``@path.json`` for shell convenience."""

    if raw is None:
        return {}
    if raw.startswith("@"):
        source = Path(raw[1:]).expanduser().resolve()
        raw = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("--provider-kwargs-json must be a JSON object or @file") from error
    if not isinstance(payload, Mapping):
        raise ValueError("--provider-kwargs-json must decode to an object")
    return _json_mapping(payload, what="provider kwargs")


def _atomic_png(path: Path, image: Any) -> None:
    """Write one RGB PNG durably and atomically within its destination folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        rgb = image.convert("RGB")
        with temporary.open("wb") as stream:
            rgb.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_action_digest(
    *,
    hdf5_path: Path,
    frame_count: int,
    chunk_start: int,
    keyframe_count: int,
    visual_stride: int,
) -> tuple[list[int], str]:
    indices = list(
        chunk_source_indices(
            chunk_start=chunk_start,
            frame_count=frame_count,
            keyframe_count=keyframe_count,
            visual_stride=visual_stride,
        )
    )
    vectors = _read_action_chunk(
        hdf5_path, frame_count=frame_count, source_indices=indices
    )
    return indices, _action_chunk_sha256(vectors)


def write_chunk_atomic(
    *,
    output_root: Path,
    episode_id: int,
    hdf5_path: Path,
    frame_count: int,
    chunk_start: int,
    keyframe_count: int,
    visual_stride: int,
    target_size: tuple[int, int],
    result: Mapping[str, Any],
) -> Path:
    """Commit a provider result using the exact precomputed-reader schema."""

    frames = result.get("frames")
    if (
        isinstance(frames, (str, bytes))
        or not isinstance(frames, Sequence)
        or len(frames) != keyframe_count
        or "latents" in result
    ):
        raise PrecomputeError(
            f"provider must return exactly {keyframe_count} frames and no latents"
        )
    provenance = _json_mapping(result.get("provenance"), what="chunk provenance")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise PrecomputeError("chunk provenance lacks source actions")
    indices, expected_digest = _expected_action_digest(
        hdf5_path=hdf5_path,
        frame_count=frame_count,
        chunk_start=chunk_start,
        keyframe_count=keyframe_count,
        visual_stride=visual_stride,
    )
    if source.get("selected_action_sha256_float32_le") != expected_digest:
        raise PrecomputeError(
            "provider provenance action SHA-256 disagrees with official HDF5 rows"
        )

    width, height = target_size
    checked_frames = []
    for index, frame in enumerate(frames):
        convert = getattr(frame, "convert", None)
        if not callable(convert):
            raise PrecomputeError(f"flow frame {index} is not PIL-like")
        rgb = convert("RGB")
        if tuple(rgb.size) != (width, height):
            raise PrecomputeError(
                f"flow frame {index} has size {rgb.size}, expected {(width, height)}"
            )
        checked_frames.append(rgb)
    if checked_frames[0].getextrema() != ((255, 255), (255, 255), (255, 255)):
        raise PrecomputeError("flow frame zero is not the mandatory white sentinel")

    chunk_dir = output_root / f"episode{episode_id}" / f"chunk_{chunk_start:06d}"
    manifest_path = chunk_dir / "manifest.json"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous manifest before touching its payload.  If anything
    # below fails, the reader cannot mistake the partial replacement as done.
    if manifest_path.exists():
        manifest_path.unlink()
    frame_names = [f"flow_{index:02d}.png" for index in range(keyframe_count)]
    for name, frame in zip(frame_names, checked_frames):
        _atomic_png(chunk_dir / name, frame)
    frame_sha256 = {
        name: _sha256_file(chunk_dir / name)
        for name in frame_names
    }
    manifest = {
        "schema_version": PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "chunk_start": chunk_start,
        "frame_count": frame_count,
        "keyframe_count": keyframe_count,
        "visual_stride": visual_stride,
        "source_indices": indices,
        "target_size": [width, height],
        "action_sha256_float32_le": expected_digest,
        "frames": frame_names,
        "frame_sha256": frame_sha256,
        "provenance": provenance,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def validate_completed_chunk(
    *,
    output_root: Path,
    episode_id: int,
    hdf5_path: Path,
    first_frame: Any,
    frame_count: int,
    chunk_start: int,
    keyframe_count: int,
    visual_stride: int,
    target_size: tuple[int, int],
) -> bool:
    """Return true only when the production reader accepts the whole chunk."""

    manifest_path = (
        output_root
        / f"episode{episode_id}"
        / f"chunk_{chunk_start:06d}"
        / "manifest.json"
    )
    if not manifest_path.is_file():
        return False
    reader = PrecomputedActionFlowConditioner(output_root)
    try:
        result = reader.get_chunk_flow(
            episode_id=episode_id,
            hdf5_path=hdf5_path,
            first_frame=first_frame,
            frame_count=frame_count,
            chunk_start=chunk_start,
            keyframe_count=keyframe_count,
            visual_stride=visual_stride,
            target_size=target_size,
        )
        frames = result.get("frames")
        return isinstance(frames, Sequence) and len(frames) == keyframe_count
    except Exception:
        # Resume is conservative: only a completely successful production
        # reader call is allowed to skip rendering.  Decode errors, stale
        # provenance, schema drift, and provider-contract errors all rebuild.
        return False


def invalidate_chunk_manifest(
    output_root: Path, *, episode_id: int, chunk_start: int
) -> None:
    """Remove only the chunk commit record before a rebuild begins."""

    manifest = (
        output_root
        / f"episode{episode_id}"
        / f"chunk_{chunk_start:06d}"
        / "manifest.json"
    )
    if manifest.exists():
        manifest.unlink()


def _install_sigterm_handler() -> Any:
    old_handler = signal.getsignal(signal.SIGTERM)

    def stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("received SIGTERM")

    signal.signal(signal.SIGTERM, stop)
    return old_handler


def _progress(
    *, event: str, episode_id: int, chunk_start: int, done: int, total: int, begun: float
) -> None:
    elapsed = max(time.monotonic() - begun, 1e-9)
    print(
        json.dumps(
            {
                "event": event,
                "episode_id": episode_id,
                "chunk_start": chunk_start,
                "completed_chunks": done,
                "total_chunks": total,
                "elapsed_seconds": round(elapsed, 3),
                "chunks_per_second": round(done / elapsed, 6),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute official Track-1 Aloha-AgileX action flow (never zero flow)"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=1)
    parser.add_argument("--episode-end", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", default="kineworld_robot_only")
    parser.add_argument(
        "--provider-kwargs-json",
        help="JSON object (or @file.json) forwarded to the action-flow provider",
    )
    parser.add_argument("--robotwin-assets-root", type=Path)
    parser.add_argument(
        "--kineworld-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    # These defaults must match infer_track1.py's native Stage-1 size.  The
    # renderer may stay at 640x480; flow is encoded at the model-native size.
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--target-height", type=int, default=240)
    parser.add_argument("--keyframes", type=int, default=DEFAULT_KEYFRAME_COUNT)
    parser.add_argument("--visual-stride", type=int, default=DEFAULT_VISUAL_STRIDE)
    parser.add_argument("--flow-method", choices=("raft", "farneback"), default="raft")
    parser.add_argument("--flow-device", default="cuda")
    return parser


def run(args: argparse.Namespace, *, conditioner: Any = None) -> int:
    try:
        from PIL import Image
    except ImportError as error:
        raise PrecomputeError("Pillow is required for Track-1 flow output") from error

    dataset_root = resolve_dataset_root(args.dataset_root)
    task_directory = resolve_task_directory(dataset_root)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ids = select_episode_ids(
        start=args.episode_start,
        end=args.episode_end,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    target_size = (int(args.target_width), int(args.target_height))
    render_size = (int(args.render_width), int(args.render_height))
    if min(*target_size, *render_size) < 1:
        raise ValueError("render and target dimensions must be positive")
    if args.keyframes < 2 or args.visual_stride < 1:
        raise ValueError("keyframes must be >=2 and visual_stride must be positive")

    # Validate and plan all selected input episodes before initializing SAPIEN
    # or loading RAFT, so an input error cannot leak heavy provider resources.
    plans: list[tuple[int, Path, Path, int, list[int]]] = []
    for episode_id in ids:
        hdf5_path, first_frame_path = episode_paths(
            dataset_root, episode_id, task_directory=task_directory
        )
        count = trajectory_frame_count(hdf5_path)
        plans.append(
            (
                episode_id,
                hdf5_path,
                first_frame_path,
                count,
                chunk_starts(
                    count,
                    keyframe_count=args.keyframes,
                    visual_stride=args.visual_stride,
                ),
            )
        )
    total = sum(len(item[4]) for item in plans)

    owned_conditioner = conditioner is None
    if conditioner is None:
        kwargs = parse_provider_kwargs(args.provider_kwargs_json)
        normalized_provider = args.provider.strip()
        if normalized_provider in BUILTIN_PROVIDER_NAMES:
            if args.robotwin_assets_root is None and "robotwin_assets_root" not in kwargs:
                raise ValueError("built-in provider requires --robotwin-assets-root")
            if args.robotwin_assets_root is not None:
                kwargs.setdefault("robotwin_assets_root", str(args.robotwin_assets_root))
            kwargs.setdefault("kineworld_root", str(args.kineworld_root))
            kwargs.setdefault("render_size", list(render_size))
            kwargs.setdefault("flow_method", args.flow_method)
            kwargs.setdefault("flow_device", args.flow_device)
        conditioner = build_action_flow_conditioner(
            provider_spec=normalized_provider,
            precomputed_root=None,
            provider_kwargs=kwargs,
        )
    if not callable(getattr(conditioner, "get_chunk_flow", None)):
        raise TypeError("action-flow provider lacks get_chunk_flow")

    done = 0
    begun = time.monotonic()
    old_sigterm = _install_sigterm_handler()
    try:
        description = getattr(conditioner, "describe", lambda: {})()
        print(json.dumps(_json_mapping(description, what="provider description"), sort_keys=True))
        for episode_id, hdf5_path, first_frame_path, count, starts in plans:
            try:
                with Image.open(first_frame_path) as image:
                    first_frame = image.convert("RGB").copy()
                for start in starts:
                    if args.resume and validate_completed_chunk(
                        output_root=output_root,
                        episode_id=episode_id,
                        hdf5_path=hdf5_path,
                        first_frame=first_frame,
                        frame_count=count,
                        chunk_start=start,
                        keyframe_count=args.keyframes,
                        visual_stride=args.visual_stride,
                        target_size=target_size,
                    ):
                        done += 1
                        _progress(
                            event="skip_valid",
                            episode_id=episode_id,
                            chunk_start=start,
                            done=done,
                            total=total,
                            begun=begun,
                        )
                        continue
                    # Manifest is the only completion record.  Remove it before
                    # invoking the provider as well as before PNG replacement;
                    # a renderer/RAFT failure therefore cannot leave an older
                    # chunk looking like this invocation completed it.
                    invalidate_chunk_manifest(
                        output_root, episode_id=episode_id, chunk_start=start
                    )
                    result = conditioner.get_chunk_flow(
                        episode_id=episode_id,
                        hdf5_path=hdf5_path,
                        first_frame=first_frame,
                        frame_count=count,
                        chunk_start=start,
                        keyframe_count=args.keyframes,
                        visual_stride=args.visual_stride,
                        target_size=target_size,
                    )
                    write_chunk_atomic(
                        output_root=output_root,
                        episode_id=episode_id,
                        hdf5_path=hdf5_path,
                        frame_count=count,
                        chunk_start=start,
                        keyframe_count=args.keyframes,
                        visual_stride=args.visual_stride,
                        target_size=target_size,
                        result=result,
                    )
                    done += 1
                    _progress(
                        event="written",
                        episode_id=episode_id,
                        chunk_start=start,
                        done=done,
                        total=total,
                        begun=begun,
                    )
            finally:
                release = getattr(conditioner, "release_episode", None)
                if callable(release):
                    release(episode_id=episode_id)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        if owned_conditioner:
            close = getattr(conditioner, "close", None)
            if callable(close):
                close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("precompute interrupted; no in-progress manifest was committed", file=sys.stderr)
        raise SystemExit(130)
