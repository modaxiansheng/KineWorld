"""CPU-only tests: no torch, accelerator, public weight download or server access."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASH = os.environ.get("KINEWORLD_TEST_BASH") or (shutil.which("bash") if os.name != "nt" else None)
spec = importlib.util.spec_from_file_location("public_training_preflight", HERE / "public_training_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def pure_vram_helpers():
    tree = ast.parse((HERE / "flow_action_train.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_validate_vram_policy", "_public_device_vram_cap", "_vram_measurements_pass"}]
    namespace = {"math": math, "FORMAL_MIN_VRAM_GIB": 54.0, "FORMAL_MAX_VRAM_GIB": 63.0}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "vram_helpers", "exec"), namespace)
    return namespace


class PublicVramTests(unittest.TestCase):
    def setUp(self):
        self.helpers = pure_vram_helpers()

    def test_audited_defaults_and_gates_are_preserved(self):
        validate = self.helpers["_validate_vram_policy"]
        self.assertEqual(validate("audited", None, None, None), (54.0, 54.0, 63.0))
        for values in ((0, 0, 0), (53, 54, 63), (54, 53, 63), (54, 54, 64)):
            with self.assertRaises(ValueError):
                validate("audited", *values)

    def test_public_has_no_historical_floor_or_ceiling(self):
        validate = self.helpers["_validate_vram_policy"]
        self.assertEqual(validate("public", None, None, None), (0.0, 0.0, 0.0))
        self.assertEqual(validate("public", 0, 0, 120), (0, 0, 120))
        cap = self.helpers["_public_device_vram_cap"]
        self.assertAlmostEqual(cap(0, 24), 22.8)
        self.assertAlmostEqual(cap(120, 80), 76)
        self.assertAlmostEqual(cap(20, 24), 20)

    def test_public_requires_finite_positive_observations(self):
        passed = self.helpers["_vram_measurements_pass"]
        self.assertTrue(passed([10, 12, 11, 13], 0, 0, 22.8, "public"))
        for invalid in (0, -1, float("nan"), float("inf")):
            for index in range(4):
                values = [10, 12, 11, 13]
                values[index] = invalid
                self.assertFalse(passed(values, 0, 0, 22.8, "public"))
        self.assertFalse(passed([10, 12, 11, 24], 0, 0, 22.8, "public"))

    def test_invalid_configured_cap_and_device_total_fail_closed(self):
        validate = self.helpers["_validate_vram_policy"]
        for values in ((-1, 0, 0), (0, 0, -1), (0, 0, float("nan")), (0, 25, 24)):
            with self.assertRaises(ValueError):
                validate("public", *values)
        for total in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.helpers["_public_device_vram_cap"](0, total)


class CheckpointPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kineworld-preflight-")
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "tiny.safetensors"
        header = {key: {"dtype": "F32", "shape": [1], "data_offsets": [4 * index, 4 * (index + 1)]}
                  for index, key in enumerate(("blocks.0.weight", "flow_stream.weight", "action_expert.weight"))}
        raw = json.dumps(header).encode()
        self.checkpoint.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(12))
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def test_custom_derivative_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "released step-500"):
            preflight.inspect_checkpoint(self.checkpoint, self.digest)
        result = preflight.inspect_checkpoint(self.checkpoint, self.digest, allow_custom=True)
        self.assertEqual(result["profile"], "custom")
        self.assertEqual(result["tensor_groups"], {"dit": 1, "flow_stream": 1, "action_expert": 1})

    def test_wrong_digest_and_release_size_fail(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            preflight.inspect_checkpoint(self.checkpoint, "a" * 64, allow_custom=True)
        with self.assertRaisesRegex(ValueError, "byte-size mismatch"):
            preflight.inspect_checkpoint(self.checkpoint, preflight.RELEASE_SHA256)

    def test_invalid_header_offsets_are_rejected_before_hash(self):
        raw = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999]}}).encode()
        self.checkpoint.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
        with self.assertRaisesRegex(ValueError, "shape/offsets"):
            preflight.inspect_checkpoint(self.checkpoint, self.digest, allow_custom=True)

    def test_preflight_help_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-S", str(HERE / "public_training_preflight.py"), "--help"],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No torch", result.stdout)

    def test_raft_weight_missing_or_wrong_size_fails_before_training(self):
        missing = Path(self.temp.name) / "missing-raft.pth"
        with self.assertRaisesRegex(ValueError, "missing offline RAFT"):
            preflight.inspect_raft_weights({"KINEWORLD_RAFT_WEIGHTS_PATH": str(missing)})
        with self.assertRaisesRegex(ValueError, "RAFT weight byte-size mismatch"):
            preflight.inspect_raft_weights({"KINEWORLD_RAFT_WEIGHTS_PATH": str(self.checkpoint)})


class EntrySourceContractTests(unittest.TestCase):
    def test_profile_is_threaded_to_both_data_and_stats(self):
        tree = ast.parse((HERE / "flow_action_train.py").read_text(encoding="utf-8"))
        for name in ("RoboTwinActionFlowDataset", "compute_global_action_norm_stats"):
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == name]
            self.assertEqual(len(calls), 1)
            keywords = {item.arg: item.value for item in calls[0].keywords}
            self.assertEqual(ast.unparse(keywords["training_manifest_profile"]), "args.training_manifest_profile")

    def test_public_checkpoint_has_strict_module_coverage(self):
        source = (HERE / "flow_action_train.py").read_text(encoding="utf-8")
        self.assertIn('public_release = warm_start_profile == "public_step500"', source)
        self.assertIn("if official_robotwin or public_release:", source)
        self.assertIn("if actual_source_counts != expected_source_counts:", source)
        self.assertIn("(825, 3, 1191) if public_release", source)
        self.assertIn("self.video_objective != TRACK1_CONDITIONAL_RGB", source)

    def test_launcher_dry_run_precedes_gpu_probe_and_cd(self):
        source = (HERE / "train.sh").read_text(encoding="utf-8")
        self.assertLess(source.index('if is_true "$DRY_RUN"; then exit 0; fi'), source.index("nvidia-smi"))
        self.assertLess(source.index("os.path.abspath"), source.index('cd "$SCRIPT_DIR"'))
        self.assertIn('TRAINING_MANIFEST_PROFILE="${TRAINING_MANIFEST_PROFILE:-audited}"', source)
        self.assertIn('--training_manifest_profile "$TRAINING_MANIFEST_PROFILE"', source)

    @unittest.skipUnless(BASH, "set KINEWORLD_TEST_BASH to run Bash launcher subprocess tests")
    def test_launcher_help_requires_no_python_or_gpu(self):
        result = subprocess.run([BASH, str(HERE / "train.sh"), "--help"], env={**os.environ, "PYTHON": "/no/python"},
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile audited|public", result.stdout)


try:
    from test_public_manifest import HAS_DATA_DEPS, write_synthetic_fixture, pm
except ImportError:
    HAS_DATA_DEPS = False


@unittest.skipUnless(BASH and HAS_DATA_DEPS, "Bash and h5py/numpy/Pillow needed for synthetic end-to-end preflight")
class LauncherFixtureTests(unittest.TestCase):
    """Tiny authored fixture proves plumbing, never weights/model quality."""
    setUp = CheckpointPreflightTests.setUp

    def test_relative_paths_dry_run_without_gpu_or_output_creation(self):
        root = Path(self.temp.name)
        dataset = root / "dataset"
        write_synthetic_fixture(dataset, frames=33)
        row = pm.inspect_episode(dataset, "adjust_bottle", 0, "train")
        manifest = root / "train.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        base = root / "models/Wan-AI/Wan2.2-TI2V-5B"
        base.mkdir(parents=True)
        # These are merely file-presence fixtures; preflight explicitly reports
        # base-model checksums/GPU runtime unverified and never loads them.
        for name in ("diffusion_pytorch_model.safetensors", "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
            (base / name).write_bytes(b"fixture-not-model-weights")
        tokenizer = root / "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
        tokenizer.mkdir(parents=True)
        (tokenizer / "spiece.model").write_bytes(b"fixture-not-a-tokenizer")
        env = {**os.environ, "PYTHON": sys.executable, "ALLOW_CUSTOM_WARM_START": "true", "FLOW_METHOD": "farneback"}
        for name in ("DATASET_BASE_PATH", "TRAINING_MANIFEST", "TRAINING_MANIFEST_SHA256", "RESUME_CHECKPOINT",
                     "RESUME_CHECKPOINT_SHA256", "MODEL_CACHE_DIR", "OUTPUT_PATH", "NUM_GPUS", "RESUME_STATE_DIR"):
            env.pop(name, None)
        args = [BASH, str(HERE / "train.sh"), "--profile", "public", "--dry-run",
                "--dataset-root", "dataset", "--manifest", "train.jsonl", "--manifest-sha256", pm.sha256_file(manifest).upper(),
                "--checkpoint", "tiny.safetensors", "--checkpoint-sha256", self.digest,
                "--model-cache-dir", "models", "--output", "output"]
        result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "preflight_passed_not_trained")
        self.assertEqual(Path(report["dataset_root"]), dataset.resolve())
        self.assertEqual(Path(report["checkpoint"]["path"]), self.checkpoint.resolve())
        self.assertEqual(Path(report["output_path"]), root / "output")
        self.assertFalse((root / "output").exists())
        self.assertIn("base model checksums", report["unverified"])
        self.assertIn("explicit Farneback ablation", report["flow_scope"])
        # Changed dataset bytes cannot pass on a second invocation.
        instruction = dataset / "adjust_bottle/aloha-agilex_clean_50/instructions/episode0.json"
        instruction.write_text('{"seen":["mutated"]}', encoding="utf-8")
        rejected = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("source/manifest mismatch", rejected.stderr)


class ReleasedCheckpointCoverageTests(unittest.TestCase):
    """Execute the real loader's AST with module/tensor mocks, not torch."""
    def setUp(self):
        class FakeModule:
            def __init__(self, keys):
                self.weights = {key: SimpleNamespace(shape=(1,)) for key in keys}
            def state_dict(self):
                return self.weights
            def load_state_dict(self, source, strict=False):
                for key in source.keys() & self.weights.keys():
                    if source[key].shape != self.weights[key].shape:
                        raise RuntimeError(f"shape mismatch: {key}")
                return SimpleNamespace(missing_keys=sorted(self.weights.keys() - source.keys()),
                                       unexpected_keys=sorted(source.keys() - self.weights.keys()))

        tree = ast.parse((HERE / "flow_action_train.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlowActionTrainingModule")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_resume_checkpoint")
        self.flow_inputs = {"flow_patch_embedding.weight", "flow_patch_embedding.bias", "stream_embed"}
        self.head_keys = {"head.weight", "head.bias", "modulation"}
        dit = FakeModule({f"dit_weight_{index}" for index in range(825)})
        action = FakeModule({f"action_weight_{index}" for index in range(1191)})
        flow = FakeModule(self.flow_inputs | {f"flow_head.{key}" for key in self.head_keys})
        flow.flow_head = FakeModule(self.head_keys)
        self.model = SimpleNamespace(pipe=SimpleNamespace(dit=dit), action_expert=action,
                                     flow_stream=flow, video_objective="track1_conditional_rgb")
        self.source = {**dit.state_dict(),
                       **{f"flow_stream.{key}": flow.state_dict()[key] for key in self.flow_inputs},
                       **{f"action_expert.{key}": value for key, value in action.state_dict().items()}}
        scope = {"load_state_dict": lambda _: self.source, "os": os,
                 "TRACK1_CONDITIONAL_RGB": "track1_conditional_rgb"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "checkpoint_loader", "exec"), scope)
        self.load = scope["_load_resume_checkpoint"]

    def run_load(self, profile="public_step500"):
        with patch.dict(os.environ, {"KINEWORLD_WARM_START_PROFILE": profile}), contextlib.redirect_stdout(io.StringIO()):
            return self.load(self.model, "mocked-checkpoint-no-io.safetensors")

    def test_released_missing_only_unused_flow_head_is_accepted(self):
        self.run_load()

    def test_missing_patch_embedding_or_stream_is_rejected(self):
        original = self.source.copy()
        for key in sorted(self.flow_inputs):
            self.source = original.copy()
            del self.source[f"flow_stream.{key}"]
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "exact module coverage"):
                self.run_load()

    def test_unexpected_flow_key_is_rejected_even_with_correct_count(self):
        del self.source["flow_stream.stream_embed"]
        self.source["flow_stream.unexpected"] = SimpleNamespace(shape=(1,))
        with self.assertRaisesRegex(RuntimeError, "flow missing/unexpected"):
            self.run_load()

    def test_public_omission_is_not_allowed_for_joint_objective(self):
        self.model.video_objective = "joint_dual_stream"
        with self.assertRaisesRegex(RuntimeError, "requires track1_conditional_rgb"):
            self.run_load()

    def test_public_dit_and_action_coverage_are_still_strict(self):
        original = self.source.copy()
        for key in ("dit_weight_0", "action_expert.action_weight_0"):
            self.source = original.copy()
            del self.source[key]
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "exact module coverage"):
                self.run_load()
        self.source = original.copy()
        self.source["action_expert.action_weight_0"] = SimpleNamespace(shape=(2,))
        with self.assertRaisesRegex(RuntimeError, "action missing/unexpected/incompatible"):
            self.run_load()

    def test_historical_robotwin_and_stage1_coverage_is_preserved(self):
        self.source.update({f"flow_stream.flow_head.{key}": value
                            for key, value in self.model.flow_stream.flow_head.state_dict().items()})
        self.run_load("robotwin_pretrained")
        del self.source["flow_stream.flow_head.modulation"]
        with self.assertRaisesRegex(RuntimeError, "exact module coverage"):
            self.run_load("robotwin_pretrained")
        self.source = {**self.model.pipe.dit.state_dict(),
                       **{f"flow_stream.{key}": value for key, value in self.model.flow_stream.state_dict().items()
                          if key != "stream_embed"}}
        self.run_load("worldarena_stage1")
        del self.source["flow_stream.flow_patch_embedding.bias"]
        with self.assertRaisesRegex(RuntimeError, "exact module coverage"):
            self.run_load("worldarena_stage1")


if __name__ == "__main__":
    unittest.main()
