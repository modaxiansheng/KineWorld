#!/usr/bin/env python3
"""Build separate, pinned public manifests from explicitly selected local episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.public_manifest import (
    CAMERAS, PUBLIC_SCHEMA, PUBLIC_VARIANT, inspect_episode,
    parse_episode_selection, validate_data_root, validate_public_rows,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "No downloads, simulation, or inferred episode records. Every selected file must exist. "
        "Minimal schema smoke subset: --tasks TASK --train-episodes 0 --val-episodes none --test-episodes none. "
        "This validates files only; it is not a training run or research result."
    ))
    parser.add_argument("--data-root", required=True, help="Root containing TASK/aloha-agilex_clean_50/")
    parser.add_argument("--output-dir", required=True, help="New output directory; existing paths are never overwritten")
    parser.add_argument("--tasks", nargs="+", required=True, help="Explicit task names (no implicit all-task discovery)")
    parser.add_argument("--variant", choices=[PUBLIC_VARIANT], default=PUBLIC_VARIANT)
    parser.add_argument("--train-episodes", default="0-35", help="Indices/ranges within 0-35")
    parser.add_argument("--val-episodes", default="36-39", help="Indices/ranges within 36-39, or none")
    parser.add_argument("--test-episodes", default="40-49", help="Quarantined indices/ranges within 40-49, or none")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=["head_camera"])
    parser.add_argument("--prompt-index", type=int, default=0, help="Pinned entry in each instructions JSON 'seen' list")
    parser.add_argument("--source-revision", default="local-files-sha256", help="Known downloaded/collected source revision if available; files are always SHA-256 pinned")
    return parser


def prepare(args):
    root = validate_data_root(args.data_root)
    output = Path(args.output_dir).absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest directory: {output}")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("duplicate task selections are forbidden")
    selections = {split: parse_episode_selection(getattr(args, f"{split}_episodes"), split)
                  for split in ("train", "val", "test")}
    if not selections["train"]:
        raise ValueError("at least one train episode must be explicitly selected")
    manifests = {}
    # Verify everything first. Missing/corrupt selected data never yields a
    # successful-looking partial allowlist or an output directory.
    for split, episodes in selections.items():
        rows = []
        for task in sorted(args.tasks):
            for episode in episodes:
                print(f"Validating {split}: {task}/episode{episode}", file=sys.stderr, flush=True)
                rows.append(inspect_episode(root, task, episode, split, cameras=args.cameras,
                                            source_revision=args.source_revision, prompt_index=args.prompt_index))
        rows = validate_public_rows(rows, split, allow_empty=split != "train")
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
        manifests[split] = (rows, payload, hashlib.sha256(payload).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": PUBLIC_SCHEMA, "manifest_profile": "public", "data_root": str(root),
               "variant": args.variant, "tasks": sorted(args.tasks), "source_revision": args.source_revision,
               "claim_boundary": "Local file/schema validation only; not proof of provenance, training, or benchmark results.",
               "splits": {}}
    for split, (rows, payload, digest) in manifests.items():
        name = f"{split}.jsonl"
        with (output / name).open("xb") as stream:
            stream.write(payload)
        with (output / f"{split}.sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{digest}  {name}\n")
        summary["splits"][split] = {"path": str(output / name), "sha256": digest,
                                    "episode_count": len(rows), "frame_count": sum(row["frame_count"] for row in rows),
                                    "episode_indices_per_task": selections[split]}
    with (output / "manifest_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = prepare(args)
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
