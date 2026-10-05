#!/usr/bin/env python3
"""Export one real RoboTwin Aloha-AgileX episode for custom offline inference.

The source HDF5 is copied intact. Only a missing /joint_action/vector is added,
assembled from existing joint fields without temporal resampling. Source frame
zero becomes the PNG and an explicitly selected source instruction becomes the
JSON. This adapter does not download data, produce sample data, or run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

try:
    from .robotwin_data_io import decode_rgb_frame, read_joint_vectors
except ImportError:
    from robotwin_data_io import decode_rgb_frame, read_joint_vectors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_instruction(path: Path, *, key: str = "auto", index: int = 0) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("instruction JSON must be an object")
    if key == "auto":
        key = "instruction" if "instruction" in payload else "seen"
    value = payload.get(key)
    if isinstance(value, list):
        if not 0 <= index < len(value):
            raise ValueError(f"instruction index {index} is outside {key!r}")
        value = value[index]
    elif index != 0:
        raise ValueError("--instruction-index only selects entries from a list")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"no nonempty source instruction at key {key!r}, index {index}")
    return value, {"key": key, "index": index, "text": value}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True, help="real raw RoboTwin episode HDF5")
    parser.add_argument("--instruction-json", type=Path, required=True)
    parser.add_argument("--instruction-key", default="auto", help="default: instruction, otherwise seen")
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True, help="custom input collection root")
    parser.add_argument("--episode-id", type=int, default=1, help="output ID in 1..1000 (default: 1)")
    parser.add_argument(
        "--image-encoding", choices=("auto", "robotwin-legacy", "standard-rgb"), default="auto",
        help="auto: marked standard or legacy RoboTwin JPEG; decoded arrays are RGB",
    )
    return parser


def prepare_episode(args: argparse.Namespace) -> dict:
    import h5py
    import numpy as np

    if not 1 <= args.episode_id <= 1000:
        raise ValueError("--episode-id must be in 1..1000")
    if args.instruction_index < 0:
        raise ValueError("--instruction-index must be nonnegative")
    source = args.hdf5.expanduser().resolve(strict=True)
    instruction_path = args.instruction_json.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    stem = f"episode{args.episode_id}"
    relative = {
        "hdf5": Path("data/fixed_scene_task") / f"{stem}.hdf5",
        "first_frame": Path("first_frame/fixed_scene_task") / f"{stem}.png",
        "instruction": Path("instructions/fixed_scene_task") / f"{stem}.json",
        "provenance": Path("preparation") / f"{stem}.json",
    }
    for path in relative.values():
        if (output_root / path).exists():
            raise FileExistsError(f"refusing to overwrite {output_root / path}; use a new output root or ID")
    instruction_hash = sha256_file(instruction_path)
    instruction, selection = select_instruction(
        instruction_path, key=args.instruction_key, index=args.instruction_index,
    )
    if sha256_file(instruction_path) != instruction_hash:
        raise RuntimeError("source instruction changed during preparation; retry from a stable source")
    source_hash = sha256_file(source)
    with h5py.File(source, "r") as handle:
        vectors, action_source = read_joint_vectors(handle)
        key = "observation/head_camera/rgb"
        if key not in handle:
            raise ValueError(f"missing /{key}")
        frames = handle[key]
        if not frames.shape or frames.shape[0] != len(vectors):
            raise ValueError("head_camera RGB and action frame counts must match exactly")
        image, color = decode_rgb_frame(frames[0], image_encoding=args.image_encoding)
    record = {
        "schema_version": 1,
        "input_profile": "custom",
        "official_submission_input": False,
        "episode_id": args.episode_id,
        "source_hdf5": str(source),
        "source_hdf5_sha256": source_hash,
        "source_instruction_json": str(instruction_path),
        "source_instruction_json_sha256": instruction_hash,
        "instruction_selection": selection,
        "action_source": action_source,
        "action_order": ["left_arm[0:6]", "left_gripper", "right_arm[0:6]", "right_gripper"],
        "trajectory_frame_count": len(vectors),
        "temporal_resampling": False,
        "first_frame": {"dataset": "/observation/head_camera/rgb", "index": 0, **color},
        "output_paths": {kind: path.as_posix() for kind, path in relative.items()},
    }
    # Stage and validate before publishing. Existing outputs are never replaced;
    # exclusive creation also handles two exporters racing on the same ID.
    with tempfile.TemporaryDirectory(prefix="kineworld-input-") as temporary:
        staging = Path(temporary)
        staged = {kind: staging / path.name for kind, path in relative.items()}
        staged["provenance"] = staging / "provenance.json"
        shutil.copyfile(source, staged["hdf5"])
        if sha256_file(staged["hdf5"]) != source_hash:
            raise RuntimeError("source HDF5 changed during preparation; retry from a stable source")
        if action_source != "/joint_action/vector":
            with h5py.File(staged["hdf5"], "r+") as handle:
                handle.create_dataset("joint_action/vector", data=vectors)
        with h5py.File(staged["hdf5"], "r") as handle:
            actual, _ = read_joint_vectors(handle)
            if not np.array_equal(vectors, actual):
                raise RuntimeError("exported action trajectory failed exact readback")
        image.save(staged["first_frame"], format="PNG")
        staged["instruction"].write_text(
            json.dumps({"instruction": instruction}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        record["output_sha256"] = {
            kind: sha256_file(staged[kind]) for kind in ("hdf5", "first_frame", "instruction")
        }
        staged["provenance"].write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        created = []
        try:
            for kind, relative_path in relative.items():
                destination = output_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as target:
                    created.append(destination)
                    with staged[kind].open("rb") as source_stream:
                        shutil.copyfileobj(source_stream, target)
        except BaseException:
            for destination in reversed(created):
                destination.unlink(missing_ok=True)
            raise
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = prepare_episode(args)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
