#!/usr/bin/env python3
"""Delegate KineWorld results to the repository's Track-1 validators.

Keeping validation in one place prevents this adapter from drifting away from
the submission contract enforced by ``track1_submission``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_ROOT = WORKSPACE_ROOT / "track1_submission"
TASK_DIRECTORY = "fixed_scene_task"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KineWorld WorldArena2 Track-1 outputs or archive"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    outputs = subparsers.add_parser(
        "outputs", help="decode and structurally validate generated MP4 files"
    )
    outputs.add_argument("--dataset-root", type=Path, required=True)
    outputs.add_argument("--videos-dir", type=Path, required=True)
    outputs.add_argument("--records-dir", type=Path)
    outputs.add_argument("--run-config", type=Path)
    outputs.add_argument("--episode-start", type=int, default=1)
    outputs.add_argument("--episode-end", type=int, default=1000)
    outputs.add_argument(
        "--default-frame-count",
        type=int,
        help="explicit fallback only when an episode HDF5 file is absent",
    )
    outputs.add_argument("--expected-fps", type=float, default=24.0)
    outputs.add_argument("--expected-width", type=int, default=640)
    outputs.add_argument("--expected-height", type=int, default=480)
    outputs.add_argument("--minimum-first-frame-psnr", type=float, default=30.0)
    outputs.add_argument(
        "--required-control-type",
        choices=("action_driven", "text_driven", "none"),
        default="action_driven",
    )
    outputs.add_argument("--workers", type=int, default=1)
    outputs.add_argument("--output-json", type=Path, required=True)

    archive = subparsers.add_parser(
        "archive", help="validate final tar.gz member names and metadata"
    )
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "outputs":
        validator = Path(__file__).with_name("validate_track1_outputs.py")
        command = [
            sys.executable,
            str(validator),
            "--dataset-root",
            str(resolve_dataset_root(args.dataset_root)),
            "--videos-dir",
            str(args.videos_dir.expanduser().resolve()),
            "--episode-start",
            str(args.episode_start),
            "--episode-end",
            str(args.episode_end),
            "--expected-fps",
            str(args.expected_fps),
            "--expected-width",
            str(args.expected_width),
            "--expected-height",
            str(args.expected_height),
            "--minimum-first-frame-psnr",
            str(args.minimum_first_frame_psnr),
            "--required-control-type",
            args.required_control_type,
            "--workers",
            str(args.workers),
            "--output-json",
            str(args.output_json.expanduser().resolve()),
        ]
        if args.default_frame_count is not None:
            command.extend(
                ["--default-frame-count", str(args.default_frame_count)]
            )
        if args.records_dir is not None:
            command.extend(
                ["--records-dir", str(args.records_dir.expanduser().resolve())]
            )
        if args.run_config is not None:
            command.extend(
                ["--run-config", str(args.run_config.expanduser().resolve())]
            )
    else:
        validator = VALIDATOR_ROOT / "validate_track1_archive.py"
        command = [
            sys.executable,
            str(validator),
            "--archive",
            str(args.archive.expanduser().resolve()),
            "--output-json",
            str(args.output_json.expanduser().resolve()),
        ]

    if not validator.is_file():
        raise FileNotFoundError(validator)
    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
