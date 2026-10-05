"""CPU-only synthetic fixtures for explicit custom inference input validation."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

import infer_track1 as inference


class CustomInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset_track1"
        for kind in ("data", "first_frame", "instructions"):
            (self.dataset / kind / "fixed_scene_task").mkdir(parents=True)
        self.make_episode(1)

    def tearDown(self):
        self.temporary.cleanup()

    def make_episode(self, episode_id):
        stem = f"episode{episode_id}"
        with h5py.File(self.dataset / "data/fixed_scene_task" / f"{stem}.hdf5", "w") as handle:
            handle.create_dataset("joint_action/vector", data=np.zeros((3, 14), dtype=np.float32))
        Image.new("RGB", (16, 12), "green").save(self.dataset / "first_frame/fixed_scene_task" / f"{stem}.png")
        (self.dataset / "instructions/fixed_scene_task" / f"{stem}.json").write_text('{"instruction":"fixture instruction"}', encoding="utf-8")

    def cli(self, *extra):
        return ["--dataset-root", str(self.dataset), "--output-root", str(self.root / "output"), "--dry-run", *extra]

    def test_default_is_official_and_one_episode_is_rejected(self):
        self.assertEqual(inference.parse_args(self.cli()).input_profile, "official")
        with self.assertRaisesRegex(ValueError, "episode set mismatch"):
            inference.discover_episodes(self.dataset)

    def test_official_still_requires_exact_1000_even_for_subset_selection(self):
        with mock.patch.object(inference, "EXPECTED_EPISODES", 2):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                inference.discover_episodes(self.dataset, episode_end=1)
            self.make_episode(2)
            self.assertEqual(len(inference.discover_episodes(self.dataset, episode_end=1)), 2)
            self.make_episode(3)
            with self.assertRaisesRegex(ValueError, "extra"):
                inference.discover_episodes(self.dataset, episode_end=1)

    def test_custom_discovers_exact_selected_range(self):
        self.make_episode(2)
        selected = inference.discover_episodes(self.dataset, input_profile="custom", episode_start=2, episode_end=2)
        self.assertEqual([episode.episode_id for episode in selected], [2])

    def test_custom_missing_range_member_or_hdf5_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            inference.discover_episodes(self.dataset, input_profile="custom", episode_end=2)
        (self.dataset / "data/fixed_scene_task/episode1.hdf5").unlink()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            inference.discover_episodes(self.dataset, input_profile="custom", episode_end=1)

    def test_custom_never_allows_frame_count_fallback_even_for_zero_flow(self):
        args = inference.parse_args(self.cli("--input-profile", "custom", "--episode-end", "1", "--conditioning-mode", "zero_flow", "--default-frame-count", "10"))
        with self.assertRaisesRegex(ValueError, "no frame-count fallback"):
            inference.validate_args(args)
        with self.assertRaisesRegex(ValueError, "real HDF5"):
            inference.discover_episodes(self.dataset, input_profile="custom", episode_end=1, allow_missing_hdf5=True)

    def test_custom_dry_run_validates_without_loading_weights(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["infer_track1.py", *self.cli("--input-profile", "custom", "--episode-end", "1")]), mock.patch.object(inference, "resolve_checkpoint", side_effect=AssertionError("must not load weights")), contextlib.redirect_stdout(stdout):
            inference.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["assigned_episode_count"], 1)
        self.assertIsNone(report["official_episode_count"])
        self.assertTrue(report["selected_contents_validated"])
        self.assertFalse((self.root / "output").exists())

    def test_custom_dry_run_rejects_nonfinite_actions(self):
        with h5py.File(self.dataset / "data/fixed_scene_task/episode1.hdf5", "r+") as handle:
            handle["joint_action/vector"][1, 1] = float("nan")
        with mock.patch.object(sys, "argv", ["infer_track1.py", *self.cli("--input-profile", "custom", "--episode-end", "1")]):
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                inference.main()

    def test_custom_dry_run_rejects_invalid_instruction(self):
        (self.dataset / "instructions/fixed_scene_task/episode1.json").write_text('{}', encoding="utf-8")
        with mock.patch.object(sys, "argv", ["infer_track1.py", *self.cli("--input-profile", "custom", "--episode-end", "1")]):
            with self.assertRaisesRegex(ValueError, "instruction"):
                inference.main()

    def test_custom_dry_run_rejects_invalid_png(self):
        (self.dataset / "first_frame/fixed_scene_task/episode1.png").write_bytes(b"not a PNG")
        with mock.patch.object(sys, "argv", ["infer_track1.py", *self.cli("--input-profile", "custom", "--episode-end", "1")]):
            with self.assertRaises(OSError):
                inference.main()

    def test_custom_resume_fingerprint_changes_when_input_bytes_change(self):
        episodes = inference.discover_episodes(self.dataset, input_profile="custom", episode_end=1)
        before = inference.custom_input_fingerprint(episodes)
        (self.dataset / "instructions/fixed_scene_task/episode1.json").write_text('{"instruction":"different text"}', encoding="utf-8")
        self.assertNotEqual(before, inference.custom_input_fingerprint(episodes))


if __name__ == "__main__":
    unittest.main()
