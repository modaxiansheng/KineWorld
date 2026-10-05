"""CPU-only public launch checks. Imports no torch and never downloads files.

The launcher runs this exactly once before importing the training stack. A
successful preflight proves input integrity only, not GPU capacity or training.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

RELEASE_SHA256 = "86294739c54073c836a0dcb3f9114c6cf2bf83d1a8698423b71698e5f88460a3"
RELEASE_BYTES = 13113498528


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value, name):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value or ""):
        raise ValueError(f"{name} must be an explicit 64-character SHA-256")
    return value.lower()


def inspect_checkpoint(path, expected_sha256, allow_custom=False):
    """Check metadata before one streaming hash; no tensor deserialization."""
    path = Path(path)
    expected_sha256 = _digest(expected_sha256, "RESUME_CHECKPOINT_SHA256")
    released = expected_sha256 == RELEASE_SHA256
    if not released and not allow_custom:
        raise ValueError("public training requires the released step-500 SHA-256; "
                         "set ALLOW_CUSTOM_WARM_START=true only for a verified derivative")
    if not path.is_file():
        raise ValueError(f"missing checkpoint: {path}")
    size = path.stat().st_size
    if released and size != RELEASE_BYTES:
        raise ValueError(f"released checkpoint byte-size mismatch: {size} != {RELEASE_BYTES}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("checkpoint lacks Safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(size - 8, 64 * 1024 * 1024):
            raise ValueError("invalid or excessive Safetensors header length")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    keys = [key for key in header if key != "__metadata__"]
    groups = {"dit": 0, "flow_stream": 0, "action_expert": 0}
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                   "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
                   "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    ranges = []
    for key in keys:
        entry = header[key]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid Safetensors tensor metadata: {key}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - header_size):
            raise ValueError(f"invalid Safetensors shape/offsets: {key}")
        element_size = dtype_bytes.get(entry.get("dtype"))
        if element_size is None or math.prod(shape) * element_size != offsets[1] - offsets[0]:
            raise ValueError(f"invalid Safetensors dtype/tensor length: {key}")
        ranges.append(tuple(offsets))
        group = "action_expert" if key.startswith("action_expert.") else (
            "flow_stream" if key.startswith("flow_stream.") else "dit")
        groups[group] += 1
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Safetensors payload has overlapping ranges or gaps")
        cursor = end
    if cursor != size - 8 - header_size or not all(groups.values()):
        raise ValueError("checkpoint must contain complete DiT, flow_stream and action_expert tensors")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual} != {expected_sha256}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": size,
            "profile": "public_step500" if released else "custom", "tensor_groups": groups}


def _truth(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def inspect_raft_weights(env):
    # Read literal constants only; importing the extractor would import cv2 and
    # numpy. Keep this preflight aligned with the actual offline RAFT loader.
    names = {"RAFT_LARGE_DEFAULT_FILENAME", "RAFT_LARGE_DEFAULT_BYTES", "RAFT_LARGE_DEFAULT_SHA256"}
    tree = ast.parse(Path(__file__).with_name("raft_flow_extractor.py").read_text(encoding="utf-8"))
    contract = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name) and target.id in names}
    explicit = env.get("KINEWORLD_RAFT_WEIGHTS_PATH", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        cache_root = Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        torch_home = Path(env.get("TORCH_HOME", str(cache_root / "torch"))).expanduser()
        path = torch_home / "hub/checkpoints" / contract["RAFT_LARGE_DEFAULT_FILENAME"]
    if not path.is_file():
        raise ValueError(f"missing offline RAFT weights: {path}; set KINEWORLD_RAFT_WEIGHTS_PATH to the official {contract['RAFT_LARGE_DEFAULT_FILENAME']}")
    if path.stat().st_size != contract["RAFT_LARGE_DEFAULT_BYTES"]:
        raise ValueError(f"RAFT weight byte-size mismatch: {path.stat().st_size} != {contract['RAFT_LARGE_DEFAULT_BYTES']}")
    digest = sha256_file(path)
    if digest != contract["RAFT_LARGE_DEFAULT_SHA256"]:
        raise ValueError("RAFT weight SHA-256 mismatch")
    return {"path": str(path.resolve()), "sha256": digest}


def preflight(env):
    if __package__:
        from .public_manifest import load_public_training_manifest, validate_public_manifest_files
    else:
        from public_manifest import load_public_training_manifest, validate_public_manifest_files
    required = ("DATASET_BASE_PATH", "TRAINING_MANIFEST", "TRAINING_MANIFEST_SHA256",
                "RESUME_CHECKPOINT", "RESUME_CHECKPOINT_SHA256", "OUTPUT_PATH")
    for name in required:
        if not env.get(name):
            raise ValueError(f"set {name}; see configs/train_public.example.env")
    root = Path(env["DATASET_BASE_PATH"]).resolve()
    if not root.is_dir():
        raise ValueError(f"missing RoboTwin training root: {root}")
    if any(marker in root.as_posix().lower() for marker in
           ("dataset_track1", "current_track1", "evaluation_inputs", "/track1_data", "/track1/test")):
        raise ValueError("official Track 1 test inputs cannot be training data")
    for directory in ("data", "first_frame", "instructions"):
        for scene in ("fixed_scene_task", "random_scene_task"):
            if (root / directory / scene).is_dir():
                raise ValueError("official Track 1 test layout cannot be training data")
    expected = {"VARIANTS": "aloha-agilex_clean_50", "CAMERAS": "head_camera",
                "NUM_FRAMES": "33", "NUM_VIDEO_FRAMES": "9", "VISUAL_STRIDE": "4",
                "SIZE_W": "320", "SIZE_H": "240", "COND_LAYER_STRIDE": "2",
                "ACTION_DIM": "14", "NUM_ACTION_LAYERS": "30",
                "ACTION_PRED_TARGET": "velocity", "ACTION_POS_MODE": "rope", "PROPRIO_MODE": "text",
                "FLOW_MODE": "robot_only", "VIDEO_OBJECTIVE": "track1_conditional_rgb"}
    for name, value in expected.items():
        if env.get(name, value).strip() != value:
            raise ValueError(f"public step-500 recipe requires {name}={value}")
    if not _truth(env.get("COND_DETACH", "true")):
        raise ValueError("public step-500 recipe requires COND_DETACH=true")
    if float(env.get("FLOW_LOSS_WEIGHT", "0")) != 0:
        raise ValueError("public conditional-RGB objective requires FLOW_LOSS_WEIGHT=0")
    if not _truth(env.get("TRACK1_PROMPT_TEMPLATE", "true")) or env.get("CAMERA_PREFIX", ""):
        raise ValueError("public recipe requires TRACK1_PROMPT_TEMPLATE=true and empty CAMERA_PREFIX")
    if _truth(env.get("LOAD_FROM_CACHE", "false")):
        raise ValueError("public preflight supports online paired data; cache provenance is not audited")
    flow_method = env.get("FLOW_METHOD", "raft")
    if flow_method not in {"raft", "farneback"}:
        raise ValueError("FLOW_METHOD must be raft or farneback")
    raft = inspect_raft_weights(env) if flow_method == "raft" else None
    if env.get("RESUME_STATE_DIR"):
        state = Path(env["RESUME_STATE_DIR"])
        if not state.is_dir() or not (state / "trainer_state.json").is_file():
            raise ValueError("RESUME_STATE_DIR needs an existing trainer_state.json; runtime exact-resume binding is still enforced")
    output = Path(env["OUTPUT_PATH"]).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"OUTPUT_PATH is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not env.get("RESUME_STATE_DIR"):
        raise ValueError("OUTPUT_PATH must be new/empty for a weights-only warm-start; use a new output directory")
    floors = [float(env.get(name, "0")) for name in ("MIN_ALLOCATED_GIB", "MIN_VRAM_GIB")]
    cap = float(env.get("MAX_VRAM_GIB", "0"))
    if not all(math.isfinite(x) and x >= 0 for x in [*floors, cap]):
        raise ValueError("public VRAM floors/cap must be finite non-negative numbers")
    if cap and max(floors) > cap:
        raise ValueError("public VRAM floors cannot exceed MAX_VRAM_GIB")
    manifest_sha = _digest(env["TRAINING_MANIFEST_SHA256"], "TRAINING_MANIFEST_SHA256")
    rows = load_public_training_manifest(env["TRAINING_MANIFEST"], manifest_sha)
    validate_public_manifest_files(str(root), rows, cameras=["head_camera"])
    cache = Path(env.get("MODEL_CACHE_DIR", "models")).resolve()
    base = cache / "Wan-AI/Wan2.2-TI2V-5B"
    if env.get("MODEL_PATHS_JSON"):
        paths = json.loads(env["MODEL_PATHS_JSON"])
        def check_paths(items):
            if not isinstance(items, list) or not items:
                raise ValueError("MODEL_PATHS_JSON must be a non-empty JSON list of model paths/shard lists")
            for item in items:
                if isinstance(item, list):
                    check_paths(item)
                elif not isinstance(item, str) or not Path(item).is_file():
                    raise ValueError(f"missing base model file: {item}")
        check_paths(paths)
    else:
        model_ids = {"MODEL_PATHS_DIT": "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors",
                     "MODEL_PATHS_T5": "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth",
                     "MODEL_PATHS_VAE": "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"}
        for name, value in model_ids.items():
            if env.get(name, value) != value:
                raise ValueError(f"public recipe requires {name}={value}; use MODEL_PATHS_JSON for explicit local files")
        needed = (base / "models_t5_umt5-xxl-enc-bf16.pth", base / "Wan2.2_VAE.pth")
        for path in needed:
            if not path.is_file():
                raise ValueError(f"missing base model file: {path}; download base models before launch")
        shards = sorted(base.glob("diffusion_pytorch_model*.safetensors"))
        if not shards:
            raise ValueError(f"missing DiT shards in {base}")
        for shard in shards:
            match = re.search(r"-(\d+)-of-(\d+)\.safetensors$", shard.name)
            if match and not all((base / re.sub(r"-\d+-of-", f"-{index:05d}-of-", shard.name)).is_file()
                                 for index in range(1, int(match.group(2)) + 1)):
                raise ValueError(f"incomplete DiT shard collection in {base}")
    tokenizer = cache / env.get("TOKENIZER_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B") / "google/umt5-xxl"
    if not tokenizer.is_dir() or not any((tokenizer / name).is_file() for name in ("tokenizer.json", "spiece.model")):
        raise ValueError(f"missing tokenizer files in {tokenizer}")
    checkpoint = inspect_checkpoint(env["RESUME_CHECKPOINT"], env["RESUME_CHECKPOINT_SHA256"],
                                    allow_custom=_truth(env.get("ALLOW_CUSTOM_WARM_START", "false")))
    return {"status": "preflight_passed_not_trained", "training_manifest_profile": "public",
            "manifest_sha256": manifest_sha, "episodes": len(rows), "dataset_root": str(root),
            "model_cache_dir": str(cache), "checkpoint": checkpoint, "output_path": str(output),
            "flow_method": flow_method, "raft_weights": raft,
            "flow_scope": "released recipe" if flow_method == "raft" else "explicit Farneback ablation, not released RAFT recipe",
            "scope": "continued fine-tuning; uniform spatial RGB loss; not TAWD or manuscript reproduction",
            "vram_cap": cap or "95% of each device total, checked at runtime",
            "unverified": ["base model checksums", "GPU compatibility/capacity", "optimizer updates", "model quality"]}


def main():
    if len(sys.argv) > 1:
        if sys.argv[1:] in (["--help"], ["-h"]):
            print("CPU-only environment-driven preflight; use bash training/train.sh --profile public --dry-run. "
                  "No torch, GPU, downloads, or training. Requires data inspection dependencies h5py/numpy/Pillow.")
            return 0
        print("Unknown arguments; use --help", file=sys.stderr)
        return 2
    try:
        print(json.dumps(preflight(os.environ), indent=2, ensure_ascii=False))
    except (OSError, ValueError, ImportError, KeyError, RuntimeError) as error:
        print(f"[public preflight] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
