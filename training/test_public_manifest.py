"""Synthetic fixtures ONLY: schema/unit checks, never training/research evidence."""

from __future__ import annotations

import ast
import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training import public_manifest as pm
from scripts.prepare_robotwin_manifest import build_parser, prepare
sys.path.insert(0, str(REPO / "training"))

try:
    import h5py
    import numpy as np
    from PIL import Image
    HAS_DATA_DEPS = True
except ImportError:
    HAS_DATA_DEPS = False


def write_synthetic_fixture(root, episode=0, *, task="adjust_bottle", frames=3, ro_frames=None, marked=False):
    """Small authored image/action fixture, not a real RoboTwin recording."""
    base = root / task / pm.PUBLIC_VARIANT
    encoded = BytesIO()
    Image.new("RGB", (8, 6), color=(30, 50, 90)).save(encoded, format="JPEG")
    jpeg = encoded.getvalue()
    if marked:
        jpeg = jpeg[:2] + b"\xff\xfe\x00\x0aXPL-RGB1" + jpeg[2:]
    for folder, count in (("data", frames), ("robot_only/data", ro_frames or frames)):
        location = base / folder
        location.mkdir(parents=True, exist_ok=True)
        with h5py.File(location / f"episode{episode}.hdf5", "w") as stream:
            rgb = stream.create_dataset("observation/head_camera/rgb", (count,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for index in range(count):
                rgb[index] = np.frombuffer(jpeg, dtype=np.uint8)
            if folder == "data":
                for field, width in (("left_arm", 6), ("left_gripper", 1), ("right_arm", 6), ("right_gripper", 1)):
                    stream.create_dataset(f"joint_action/{field}", data=np.arange(count * width, dtype=np.float32).reshape(count, width))
    (base / "instructions").mkdir(exist_ok=True)
    (base / "instructions" / f"episode{episode}.json").write_text(
        json.dumps({"seen": ["Synthetic fixture instruction for schema testing only."]}), encoding="utf-8")
    return base


class PublicManifestPureTests(unittest.TestCase):
    def test_help_is_dependency_free(self):
        result = subprocess.run([sys.executable, "-S", str(REPO / "scripts/prepare_robotwin_manifest.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No downloads", result.stdout)

    def test_exact_split_boundaries_and_no_duplicates(self):
        self.assertEqual(pm.parse_episode_selection("0,2-3", "train"), [0, 2, 3])
        self.assertEqual(pm.parse_episode_selection("none", "test"), [])
        for text, split in (("36", "train"), ("40", "val"), ("39", "test"), ("0-2,2", "train"), ("3-1", "train"), ("", "train")):
            with self.subTest(text=text, split=split), self.assertRaises(ValueError):
                pm.parse_episode_selection(text, split)

    def test_task_filter_cannot_silently_change_allowlist(self):
        rows = [{"task": "adjust_bottle"}, {"task": "beat_block_hammer"}]
        self.assertEqual(pm.select_public_training_tasks(rows), ["adjust_bottle", "beat_block_hammer"])
        with self.assertRaises(ValueError):
            pm.select_public_training_tasks(rows, ["adjust_bottle"])
        with self.assertRaises(ValueError):
            pm.select_public_training_tasks(rows, [])

    def test_audited_default_hash_and_exact_2000_gate_are_preserved(self):
        # Execute only the manifest loader's stdlib AST, not torch/RAFT imports.
        tree = ast.parse((REPO / "training/dataset_action_robotwin.py").read_text(encoding="utf-8"))
        names = {"TRACK1_FINAL_TRAIN_MANIFEST_SHA256", "TRACK1_ROBOTWIN_SOURCE_REVISION", "ROBOTWIN_ALL_TASKS"}
        nodes = [node for node in tree.body if
                 isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
                 or isinstance(node, ast.FunctionDef) and node.name in {"_sha256_file", "load_track1_training_manifest"}]
        scope = {"hashlib": hashlib, "json": json}
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), "dataset_manifest_contract", "exec"), scope)
        expected = "b21baa6a699e8972a8a3f590c676bcf1471214f4f56a2ecad5d2c9e78e302497"
        self.assertEqual(scope["TRACK1_FINAL_TRAIN_MANIFEST_SHA256"], expected)
        with self.assertRaisesRegex(ValueError, "only the audited"):
            scope["load_track1_training_manifest"]("unused", "0" * 64)
        with tempfile.TemporaryDirectory(prefix="kw-fixture-") as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            scope["_sha256_file"] = lambda _: expected
            with self.assertRaisesRegex(ValueError, "must contain 2000"):
                scope["load_track1_training_manifest"](str(path))


@unittest.skipUnless(HAS_DATA_DEPS, "install h5py numpy Pillow for explicitly synthetic fixture tests")
class PublicManifestFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kw-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.base = write_synthetic_fixture(self.root)
        self.row = pm.inspect_episode(self.root, "adjust_bottle", 0, "train")

    def _manifest(self, rows):
        path = Path(self.temp.name) / "train.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path, pm.sha256_file(path)

    def test_roundtrip_pins_all_files_and_actions(self):
        path, digest = self._manifest([self.row])
        rows = pm.load_public_training_manifest(path, digest)
        pm.validate_public_manifest_files(self.root, rows, ["head_camera"])
        self.assertEqual(rows[0]["frame_count"], 3)
        self.assertEqual(rows[0]["action_dim"], 14)

    def test_wrong_hash_and_missing_hash_are_rejected(self):
        path, _ = self._manifest([self.row])
        for digest in ("0" * 64, "", None):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                pm.load_public_training_manifest(path, digest)

    def test_mix_duplicates_and_invalid_paths_are_rejected(self):
        for key, value in (("split", "val"), ("episode_index", 40), ("hdf5_path", "../episode0.hdf5"), ("prompt_seen", ""), ("schema", "unknown")):
            row = copy.deepcopy(self.row)
            row[key] = value
            path, digest = self._manifest([row])
            with self.subTest(key=key), self.assertRaises(ValueError):
                pm.load_public_training_manifest(path, digest)
        path, digest = self._manifest([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pm.load_public_training_manifest(path, digest)

    def test_byte_mutation_and_prompt_mutation_are_rejected(self):
        path = self.base / "instructions/episode0.json"
        path.write_text(json.dumps({"seen": ["Different synthetic fixture prompt"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/manifest mismatch"):
            pm.validate_public_manifest_files(self.root, [self.row])

    def test_missing_paired_render_is_never_filtered(self):
        (self.base / "robot_only/data/episode0.hdf5").unlink()
        with self.assertRaises(FileNotFoundError):
            pm.validate_public_manifest_files(self.root, [self.row])

    def test_paired_frame_counts_must_match(self):
        write_synthetic_fixture(self.root, ro_frames=2)
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            pm.inspect_episode(self.root, "adjust_bottle", 0, "train")

    def test_action_width_and_finiteness_are_strict(self):
        with h5py.File(self.base / "data/episode0.hdf5", "a") as stream:
            del stream["joint_action/left_arm"]
            stream.create_dataset("joint_action/left_arm", data=np.ones((3, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            pm.inspect_episode(self.root, "adjust_bottle", 0, "train")
        write_synthetic_fixture(self.root)
        with h5py.File(self.base / "data/episode0.hdf5", "a") as stream:
            stream["joint_action/left_arm"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            pm.inspect_episode(self.root, "adjust_bottle", 0, "train")

    def test_corrupt_rgb_and_new_encoding_fail_clearly(self):
        with h5py.File(self.base / "data/episode0.hdf5", "a") as stream:
            stream["observation/head_camera/rgb"][1] = np.array([0, 1, 2], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "invalid RGB frame"):
            pm.inspect_episode(self.root, "adjust_bottle", 0, "train")
        write_synthetic_fixture(self.root, marked=True)
        with self.assertRaisesRegex(ValueError, "XPL-RGB1"):
            pm.inspect_episode(self.root, "adjust_bottle", 0, "train")

    def test_test_shaped_root_is_rejected(self):
        for name in ("test", "track1_test", "hidden-test-data", "heldout", "dataset_track1", "evaluation_inputs"):
            root = Path(self.temp.name) / name
            root.mkdir()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "test/held-out"):
                pm.validate_data_root(root)

    def test_official_test_layout_is_rejected_even_under_neutral_root(self):
        (self.root / "first_frame/fixed_scene_task").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "official Track 1 test layout"):
            pm.validate_data_root(self.root)

    def test_unpinned_cameras_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cameras were not pinned"):
            pm.validate_public_manifest_files(self.root, [self.row], ["left_camera"])

    def test_cli_prepares_separate_manifests_without_overwrite(self):
        write_synthetic_fixture(self.root, 36)
        write_synthetic_fixture(self.root, 40)
        output = Path(self.temp.name) / "manifests"
        args = build_parser().parse_args(["--data-root", str(self.root), "--output-dir", str(output), "--tasks", "adjust_bottle", "--train-episodes", "0", "--val-episodes", "36", "--test-episodes", "40"])
        summary = prepare(args)
        self.assertEqual([summary["splits"][split]["episode_count"] for split in ("train", "val", "test")], [1, 1, 1])
        pm.load_public_training_manifest(output / "train.jsonl", summary["splits"]["train"]["sha256"])
        with self.assertRaisesRegex(ValueError, "split mixing"):
            pm.load_public_training_manifest(output / "val.jsonl", summary["splits"]["val"]["sha256"])
        with self.assertRaises(FileExistsError):
            prepare(args)

    def test_missing_selection_produces_no_partial_output(self):
        output = Path(self.temp.name) / "manifests"
        args = build_parser().parse_args(["--data-root", str(self.root), "--output-dir", str(output), "--tasks", "adjust_bottle", "--train-episodes", "0-1", "--val-episodes", "none", "--test-episodes", "none"])
        with self.assertRaises(FileNotFoundError):
            prepare(args)
        self.assertFalse(output.exists())

    def _cpu_dataset_scope(self):
        # Execute unchanged CPU dataset/normalization code, isolating unused
        # torch/RAFT imports. No model or numerical kernel is replaced here.
        source = ast.parse((REPO / "training/dataset_action_robotwin.py").read_text(encoding="utf-8"))
        functions = {"_normalize_variants", "_action_norm_workers", "_sha256_file", "load_track1_training_manifest",
                     "_manifest_episode_paths", "_assemble_joint_sequence", "_load_episode_qpos", "compute_global_action_norm_stats"}
        constants = {"ROBOTWIN_ALL_TASKS", "TRACK1_FINAL_TRAIN_MANIFEST_SHA256", "TRACK1_ROBOTWIN_SOURCE_REVISION", "TRACK1_TRAIN_VARIANT", "WRIST_MAG_RATIO", "FLOW_NOISE_THRESHOLD_PX"}
        nodes = [node for node in source.body if
                 isinstance(node, ast.FunctionDef) and node.name in functions
                 or isinstance(node, ast.ClassDef) and node.name in {"ActionNormStats", "RoboTwinActionFlowDataset"}
                 or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
                 or isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in constants]

        class Progress:
            def __init__(self, *args, **kwargs):
                pass
            def update(self, count):
                pass
            def close(self):
                pass

        scope = dict(globals(), tqdm=Progress, FlowCodec=object)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "dataset_cpu_contract", "exec"), scope)
        return scope

    def test_public_dataset_and_norm_use_exact_manifest(self):
        scope = self._cpu_dataset_scope()
        path, digest = self._manifest([self.row])
        # A fixed joint is legal in a one-episode subset, and must not force the
        # caller to fabricate variability or add unrelated episodes.
        with h5py.File(self.base / "data/episode0.hdf5", "a") as stream:
            stream["joint_action/left_gripper"][:] = 0
        self.row = pm.inspect_episode(self.root, "adjust_bottle", 0, "train")
        path, digest = self._manifest([self.row])
        norm = scope["compute_global_action_norm_stats"](
            str(self.root), training_manifest_path=str(path), training_manifest_sha256=digest,
            training_manifest_profile="public")
        self.assertEqual(norm.training_manifest_sha256, digest)
        self.assertTrue(np.isfinite(norm.std).all())
        self.assertGreater(norm.std[6], 0)
        dataset = scope["RoboTwinActionFlowDataset"](
            str(self.root), cameras=["head_camera"], training_manifest_path=str(path),
            training_manifest_sha256=digest, training_manifest_profile="public", action_norm_stats=norm)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0]["episode_name"], "episode0")
        self.assertEqual(dataset.task_names, ["adjust_bottle"])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            scope["compute_global_action_norm_stats"](
                str(self.root), task_names=["beat_block_hammer"], training_manifest_path=str(path),
                training_manifest_sha256=digest, training_manifest_profile="public")

    def test_public_cache_error_is_not_silently_skipped(self):
        scope = self._cpu_dataset_scope()
        dataset = scope["RoboTwinActionFlowDataset"].__new__(scope["RoboTwinActionFlowDataset"])
        dataset.training_manifest_profile = "public"
        dataset._cached_chunks = []
        with self.assertRaisesRegex(RuntimeError, "never skips"):
            dataset._load_chunk_from_cache_safe(0)


if __name__ == "__main__":
    unittest.main()
