#!/usr/bin/env python3
"""Pure CPU contract tests for precompute_action_flow.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image

import precompute_action_flow as precompute
from action_flow_conditioning import (
    PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
    PrecomputedActionFlowConditioner,
    PrecomputedFlowError,
    _action_chunk_sha256,
    _read_action_chunk,
    _sha256_file,
    chunk_source_indices,
)


class FakeActionFlowProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.released: list[int] = []

    def describe(self):
        return {"conditioner": "fake_action_flow_for_contract_test"}

    def get_chunk_flow(
        self,
        *,
        episode_id,
        hdf5_path,
        first_frame,
        frame_count,
        chunk_start,
        keyframe_count,
        visual_stride,
        target_size,
    ):
        del first_frame
        indices = chunk_source_indices(
            chunk_start=chunk_start,
            frame_count=frame_count,
            keyframe_count=keyframe_count,
            visual_stride=visual_stride,
        )
        vectors = _read_action_chunk(
            hdf5_path, frame_count=frame_count, source_indices=indices
        )
        digest = _action_chunk_sha256(vectors)
        self.calls.append((episode_id, chunk_start))
        frames = [Image.new("RGB", target_size, "white")]
        frames.extend(
            Image.new("RGB", target_size, (index, 20, 30))
            for index in range(1, keyframe_count)
        )
        return {
            "frames": tuple(frames),
            "provenance": {
                "source": {"selected_action_sha256_float32_le": digest},
                "renderer": {
                    "embodiment": "aloha-agilex",
                    "camera": {"name": "head_camera"},
                    "urdf": {"path": "/fake/aloha.urdf", "sha256": "f" * 64},
                },
            },
        }

    def release_episode(self, *, episode_id):
        self.released.append(episode_id)


class FailingProvider(FakeActionFlowProvider):
    def get_chunk_flow(self, **kwargs):
        raise RuntimeError("synthetic renderer failure")


def make_args(dataset: Path, output: Path, *, resume: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=dataset,
        output_root=output,
        episode_start=1,
        episode_end=1,
        shard_index=0,
        num_shards=1,
        resume=resume,
        provider="unused.fake:provider",
        provider_kwargs_json=None,
        robotwin_assets_root=None,
        kineworld_root=Path(__file__).resolve().parents[1],
        render_width=640,
        render_height=480,
        target_width=64,
        target_height=48,
        keyframes=9,
        visual_stride=4,
        flow_method="raft",
        flow_device="cpu",
    )


class PrecomputeActionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset_track1"
        for kind in ("data", "first_frame", "instructions"):
            (self.dataset / kind / "unknown_task").mkdir(parents=True)
        actions = np.zeros((35, 14), dtype=np.float32)
        actions[:, :6] = np.arange(35, dtype=np.float32)[:, None] / 100.0
        actions[:, 7:13] = -actions[:, :6]
        actions[:, 6] = 0.25
        actions[:, 13] = 0.75
        with h5py.File(
            self.dataset / "data" / "unknown_task" / "episode1.hdf5", "w"
        ) as handle:
            handle.create_dataset("/joint_action/vector", data=actions)
        Image.new("RGB", (64, 48), (1, 2, 3)).save(
            self.dataset / "first_frame" / "unknown_task" / "episode1.png"
        )
        (self.dataset / "instructions" / "unknown_task" / "episode1.json").write_text(
            '{"instruction":"test"}\n', encoding="utf-8"
        )
        self.output = self.root / "flows"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_chunk_starts_and_sharding(self) -> None:
        self.assertEqual(precompute.chunk_starts(35), [0, 32])
        self.assertEqual(
            precompute.select_episode_ids(start=1, end=8, shard_index=1, num_shards=3),
            [2, 5, 8],
        )

    def test_cli_defaults_match_inference_native_size(self) -> None:
        args = precompute.build_parser().parse_args(
            [
                "--dataset-root",
                str(self.dataset),
                "--output-root",
                str(self.output),
            ]
        )
        self.assertEqual((args.render_width, args.render_height), (640, 480))
        self.assertEqual((args.target_width, args.target_height), (320, 240))

    def test_task_directory_is_discovered_not_guessed(self) -> None:
        self.assertEqual(precompute.resolve_task_directory(self.dataset), "unknown_task")
        for kind in ("data", "first_frame", "instructions"):
            (self.dataset / kind / "unknown_task").rename(
                self.dataset / kind / "fixed_scene_task"
            )
        self.assertEqual(
            precompute.resolve_task_directory(self.dataset), "fixed_scene_task"
        )
        hdf5, first_frame = precompute.episode_paths(self.dataset, 1)
        self.assertEqual(hdf5.parent.name, "fixed_scene_task")
        self.assertEqual(first_frame.parent.name, "fixed_scene_task")

    def test_write_then_strict_resume(self) -> None:
        first = FakeActionFlowProvider()
        self.assertEqual(
            precompute.run(make_args(self.dataset, self.output, resume=False), conditioner=first),
            0,
        )
        self.assertEqual(first.calls, [(1, 0), (1, 32)])
        for start in (0, 32):
            manifest = (
                self.output / "episode1" / f"chunk_{start:06d}" / "manifest.json"
            )
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"], PRECOMPUTED_MANIFEST_SCHEMA_VERSION
            )
            self.assertEqual(set(payload["frame_sha256"]), set(payload["frames"]))
            for name in payload["frames"]:
                self.assertEqual(
                    payload["frame_sha256"][name],
                    _sha256_file(manifest.parent / name),
                )

        resumed = FakeActionFlowProvider()
        self.assertEqual(
            precompute.run(make_args(self.dataset, self.output, resume=True), conditioner=resumed),
            0,
        )
        self.assertEqual(resumed.calls, [])
        self.assertEqual(resumed.released, [1])

    def test_corrupt_png_is_not_skipped(self) -> None:
        precompute.run(
            make_args(self.dataset, self.output, resume=False),
            conditioner=FakeActionFlowProvider(),
        )
        Image.new("RGB", (7, 7), "black").save(
            self.output / "episode1" / "chunk_000032" / "flow_01.png"
        )
        resumed = FakeActionFlowProvider()
        precompute.run(
            make_args(self.dataset, self.output, resume=True), conditioner=resumed
        )
        self.assertEqual(resumed.calls, [(1, 32)])

    def test_same_size_white_png_replacement_fails_closed(self) -> None:
        precompute.run(
            make_args(self.dataset, self.output, resume=False),
            conditioner=FakeActionFlowProvider(),
        )
        tampered = self.output / "episode1" / "chunk_000032" / "flow_01.png"
        Image.new("RGB", (64, 48), "white").save(tampered, format="PNG")

        hdf5, png = precompute.episode_paths(self.dataset, 1)
        with Image.open(png) as image:
            first_frame = image.convert("RGB").copy()
        reader = PrecomputedActionFlowConditioner(self.output)
        with self.assertRaisesRegex(PrecomputedFlowError, "SHA-256 mismatch"):
            reader.get_chunk_flow(
                episode_id=1,
                hdf5_path=hdf5,
                first_frame=first_frame,
                frame_count=35,
                chunk_start=32,
                keyframe_count=9,
                visual_stride=4,
                target_size=(64, 48),
            )

        resumed = FakeActionFlowProvider()
        precompute.run(
            make_args(self.dataset, self.output, resume=True), conditioner=resumed
        )
        self.assertEqual(resumed.calls, [(1, 32)])

    def test_legacy_v1_manifest_requires_rebuild(self) -> None:
        precompute.run(
            make_args(self.dataset, self.output, resume=False),
            conditioner=FakeActionFlowProvider(),
        )
        manifest = self.output / "episode1" / "chunk_000000" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        payload.pop("frame_sha256")
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        hdf5, png = precompute.episode_paths(self.dataset, 1)
        with Image.open(png) as image:
            first_frame = image.convert("RGB").copy()
        with self.assertRaisesRegex(PrecomputedFlowError, "legacy.*schema v1"):
            PrecomputedActionFlowConditioner(self.output).get_chunk_flow(
                episode_id=1,
                hdf5_path=hdf5,
                first_frame=first_frame,
                frame_count=35,
                chunk_start=0,
                keyframe_count=9,
                visual_stride=4,
                target_size=(64, 48),
            )

    def test_bad_action_digest_never_commits_manifest(self) -> None:
        provider = FakeActionFlowProvider()
        hdf5, png = precompute.episode_paths(self.dataset, 1)
        with Image.open(png) as image:
            first_frame = image.convert("RGB").copy()
        result = provider.get_chunk_flow(
            episode_id=1,
            hdf5_path=hdf5,
            first_frame=first_frame,
            frame_count=35,
            chunk_start=0,
            keyframe_count=9,
            visual_stride=4,
            target_size=(64, 48),
        )
        result["provenance"]["source"]["selected_action_sha256_float32_le"] = "0" * 64
        with self.assertRaises(precompute.PrecomputeError):
            precompute.write_chunk_atomic(
                output_root=self.output,
                episode_id=1,
                hdf5_path=hdf5,
                frame_count=35,
                chunk_start=0,
                keyframe_count=9,
                visual_stride=4,
                target_size=(64, 48),
                result=result,
            )
        self.assertFalse(
            (self.output / "episode1" / "chunk_000000" / "manifest.json").exists()
        )

    def test_provider_failure_invalidates_old_manifest(self) -> None:
        precompute.run(
            make_args(self.dataset, self.output, resume=False),
            conditioner=FakeActionFlowProvider(),
        )
        manifest = self.output / "episode1" / "chunk_000000" / "manifest.json"
        self.assertTrue(manifest.is_file())
        with self.assertRaisesRegex(RuntimeError, "synthetic renderer failure"):
            precompute.run(
                make_args(self.dataset, self.output, resume=False),
                conditioner=FailingProvider(),
            )
        self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
