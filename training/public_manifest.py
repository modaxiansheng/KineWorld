"""Explicit, content-pinned RoboTwin subsets; no torch import and no downloads.

This is a file/schema preflight, not a certification of real-world provenance or
research results. The public profile deliberately supports the legacy RoboTwin
JPEG encoding consumed by this repository, not newer XPL-RGB1-marked images.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path, PurePosixPath
import re


PUBLIC_SCHEMA = "kineworld_robotwin_public_v1"
PUBLIC_DATASET_ID = "robotwin2_public_subset"
PUBLIC_VARIANT = "aloha-agilex_clean_50"
SPLIT_BOUNDS = {"train": (0, 35), "val": (36, 39), "test": (40, 49)}
CAMERAS = ("head_camera", "left_camera", "right_camera")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_TASK_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_TEST_COMPONENT_RE = re.compile(r"(^|[^a-z0-9])(test|testing|testset|testdata|heldout|holdout|quarantine)([^a-z0-9]|$)")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_data_root(data_root) -> Path:
    """Reject explicitly test-shaped roots, including symlink destinations."""
    original = Path(data_root).absolute()
    root = original.resolve(strict=True)
    for path in (original, root):
        if any(marker in path.as_posix().lower() for marker in
               ("dataset_track1", "current_track1", "evaluation_inputs", "/track1_data")):
            raise ValueError(f"test/held-out-shaped data root is forbidden: {path}")
        for part in path.parts:
            if _TEST_COMPONENT_RE.search(part.lower()):
                raise ValueError(f"test/held-out-shaped data root is forbidden: {path}")
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    for directory in ("data", "first_frame", "instructions"):
        for scene in ("fixed_scene_task", "random_scene_task"):
            if (root / directory / scene).is_dir():
                raise ValueError(f"official Track 1 test layout cannot be training data: {root}")
    return root


def _relative_paths(task: str, episode_index: int) -> dict:
    base = PurePosixPath(task) / PUBLIC_VARIANT
    episode = f"episode{episode_index}"
    return {
        "hdf5_path": str(base / "data" / f"{episode}.hdf5"),
        "robot_only_hdf5_path": str(base / "robot_only" / "data" / f"{episode}.hdf5"),
        "instruction_path": str(base / "instructions" / f"{episode}.json"),
    }


def _checked_file(root: Path, relative: str) -> Path:
    expected = root / relative
    path = expected.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path != expected:
        raise ValueError(f"episode files must remain at their declared paths inside data root (no per-episode symlink remapping): {relative}")
    return path


def parse_episode_selection(text: str, split: str) -> list[int]:
    """Comma-separated indices/inclusive ranges; 'none' explicitly disables a split."""
    if split not in SPLIT_BOUNDS:
        raise ValueError(f"unknown split: {split}")
    if text.strip().lower() == "none":
        return []
    selected = []
    for item in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", item)
        if not match:
            raise ValueError(f"invalid {split} episode selection: {text!r}")
        first, last = int(match[1]), int(match[2] or match[1])
        lower, upper = SPLIT_BOUNDS[split]
        if first > last or first < lower or last > upper:
            raise ValueError(f"{split} episodes must stay in {lower}-{upper}: {item!r}")
        selected.extend(range(first, last + 1))
    if len(selected) != len(set(selected)):
        raise ValueError(f"duplicate episode in {split} selection: {text!r}")
    return sorted(selected)


def validate_public_rows(rows: list[dict], expected_split: str = "train", *, allow_empty=False):
    """Validate the complete selected split, never filtering rows."""
    if expected_split not in SPLIT_BOUNDS:
        raise ValueError(f"unknown split: {expected_split}")
    if not rows and not allow_empty:
        raise ValueError(f"public {expected_split} manifest must not be empty")
    seen = set()
    revisions = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("public manifest rows must be JSON objects")
        task, index = row.get("task"), row.get("episode_index")
        if not isinstance(task, str) or not _TASK_RE.fullmatch(task):
            raise ValueError(f"unsafe or missing task identity: {task!r}")
        if type(index) is not int:
            raise ValueError(f"episode_index must be an integer: {index!r}")
        episode_id = f"{task}/episode{index}"
        if row.get("episode_id") != episode_id or episode_id in seen:
            raise ValueError(f"invalid or duplicate episode identity: {episode_id}")
        seen.add(episode_id)
        if row.get("split") != expected_split:
            raise ValueError(f"split mixing forbidden: expected {expected_split}, found {row.get('split')} for {episode_id}")
        lower, upper = SPLIT_BOUNDS[expected_split]
        if not lower <= index <= upper:
            raise ValueError(f"{expected_split} episode outside {lower}-{upper}: {episode_id}")
        if row.get("schema") != PUBLIC_SCHEMA or row.get("manifest_profile") != "public":
            raise ValueError(f"unsupported public manifest schema/profile for {episode_id}")
        if row.get("dataset_id") != PUBLIC_DATASET_ID or row.get("variant") != PUBLIC_VARIANT:
            raise ValueError(f"unsupported dataset/variant for {episode_id}")
        revision = row.get("source_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"missing source_revision for {episode_id}")
        revisions.add(revision)
        if row.get("image_encoding") != "robotwin_legacy_channel_reversed_jpeg":
            raise ValueError(f"unsupported image_encoding for {episode_id}")
        if type(row.get("frame_count")) is not int or row["frame_count"] < 2:
            raise ValueError(f"invalid frame_count for {episode_id}")
        if row.get("action_dim") != 14:
            raise ValueError(f"action_dim must be 14 for {episode_id}")
        if not isinstance(row.get("prompt_seen"), str) or not row["prompt_seen"].strip():
            raise ValueError(f"missing pinned prompt_seen for {episode_id}")
        if type(row.get("prompt_index")) is not int or row["prompt_index"] < 0:
            raise ValueError(f"invalid prompt_index for {episode_id}")
        for key, value in _relative_paths(task, index).items():
            if row.get(key) != value:
                raise ValueError(f"{key} must equal {value!r} for {episode_id}")
        for key in ("hdf5_sha256", "robot_only_hdf5_sha256", "instruction_sha256", "action_sha256"):
            if not isinstance(row.get(key), str) or not _SHA_RE.fullmatch(row[key]):
                raise ValueError(f"invalid {key} for {episode_id}")
        cameras = row.get("cameras")
        if not isinstance(cameras, dict) or "head_camera" not in cameras or set(cameras) - set(CAMERAS):
            raise ValueError(f"invalid camera records for {episode_id}")
        for camera, record in cameras.items():
            if not isinstance(record, dict) or record.get("rgb_key") != f"observation/{camera}/rgb":
                raise ValueError(f"invalid {camera} record for {episode_id}")
            if record.get("frame_count") != row["frame_count"]:
                raise ValueError(f"RGB/action frame count mismatch for {episode_id}")
            if any(type(record.get(key)) is not int or record[key] < 1 for key in ("width", "height")):
                raise ValueError(f"invalid image dimensions for {episode_id}")
        ro = row.get("robot_only_camera")
        if not isinstance(ro, dict) or ro != cameras["head_camera"]:
            raise ValueError(f"robot_only/head camera mismatch for {episode_id}")
    if len(revisions) > 1:
        raise ValueError("source revision mixing is forbidden within a public manifest")
    return sorted(rows, key=lambda row: (row["task"], row["episode_index"]))


def load_public_training_manifest(manifest_path, expected_sha256: str) -> list[dict]:
    """Load only a caller-pinned, train-only public manifest using stdlib."""
    if not manifest_path:
        raise ValueError("training_manifest_path is required for public training")
    if not isinstance(expected_sha256, str) or not _SHA_RE.fullmatch(expected_sha256):
        raise ValueError("public training requires an explicit 64-character SHA-256 from preparation")
    # Parse exactly the bytes that were hashed, avoiding a hash/reopen race.
    data = Path(manifest_path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"training manifest SHA-256 mismatch: {actual} != {expected_sha256}")
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line in training manifest {manifest_path}:{number}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in training manifest {manifest_path}:{number}") from error
    return validate_public_rows(rows)


def _data_dependencies():
    try:
        import h5py
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("RoboTwin file validation requires h5py, numpy and Pillow; install these in your environment") from error
    return h5py, np, Image


def _inspect_rgb(stream, camera: str, expected_frames: int, np, Image) -> dict:
    key = f"observation/{camera}/rgb"
    if key not in stream or stream[key].shape[0] != expected_frames:
        raise ValueError(f"missing RGB or RGB/action frame count mismatch: {stream.filename}:{key}")
    dimensions = None
    for index in range(expected_frames):
        value = stream[key][index]
        if isinstance(value, np.ndarray) and (value.ndim != 1 or value.dtype != np.uint8):
            raise ValueError(f"expected encoded JPEG bytes at {stream.filename}:{key}[{index}]")
        try:
            encoded = bytes(value)
            with Image.open(BytesIO(encoded)) as frame:
                if frame.format != "JPEG" or frame.mode != "RGB":
                    raise ValueError("only three-channel RoboTwin legacy JPEG frames are supported")
                # New official XPolicyLab encoders put this marker in a JPEG COM
                # segment. The existing training/precompute reader swaps R/B.
                if b"XPL-RGB1" in encoded:
                    raise ValueError("XPL-RGB1 standard RGB JPEG is unsupported by this legacy training reader; use the pinned legacy RoboTwin collector, do not strip/relabel the marker")
                frame.load()
                current = frame.size
        except Exception as error:
            raise ValueError(f"invalid RGB frame {stream.filename}:{key}[{index}]: {error}") from error
        if dimensions is not None and current != dimensions:
            raise ValueError(f"inconsistent frame dimensions in {stream.filename}:{key}")
        dimensions = current
    return {"rgb_key": key, "frame_count": expected_frames, "width": dimensions[0], "height": dimensions[1]}


def inspect_episode(data_root, task: str, episode_index: int, split: str, *,
                    cameras=("head_camera",), source_revision="local-files-sha256",
                    prompt_index=0) -> dict:
    """Inspect every selected RGB frame and finite 14D action; pin source bytes."""
    if not isinstance(task, str) or not _TASK_RE.fullmatch(task):
        raise ValueError(f"unsafe task name: {task!r}")
    if split not in SPLIT_BOUNDS or type(episode_index) is not int or not SPLIT_BOUNDS[split][0] <= episode_index <= SPLIT_BOUNDS[split][1]:
        raise ValueError(f"invalid episode {episode_index} for split {split}")
    if "head_camera" not in cameras or len(set(cameras)) != len(cameras) or set(cameras) - set(CAMERAS):
        raise ValueError(f"cameras must include head_camera and be unique members of {CAMERAS}")
    root = validate_data_root(data_root)
    relative = _relative_paths(task, episode_index)
    paths = {key: _checked_file(root, value) for key, value in relative.items()}
    h5py, np, Image = _data_dependencies()
    before = {key: sha256_file(path) for key, path in paths.items()}
    with h5py.File(paths["hdf5_path"], "r") as stream:
        arrays = []
        length = None
        for field, width in (("left_arm", 6), ("left_gripper", 1), ("right_arm", 6), ("right_gripper", 1)):
            key = f"joint_action/{field}"
            if key not in stream:
                raise ValueError(f"missing {key}: {paths['hdf5_path']}")
            raw = np.asarray(stream[key][:])
            if raw.dtype.kind not in "fiu" or raw.ndim not in (1, 2):
                raise ValueError(f"invalid numeric action shape/dtype for {key}: {raw.shape}/{raw.dtype}")
            array = np.asarray(raw, dtype=np.float32)
            if width == 1 and array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 2 or not np.isfinite(array).all():
                raise ValueError(f"expected finite (T,{width}) actions with T>=2: {key}")
            if length is not None and array.shape[0] != length:
                raise ValueError(f"joint action frame count mismatch: {key}")
            length = array.shape[0]
            arrays.append(array)
        qpos = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype="<f4")
        camera_records = {camera: _inspect_rgb(stream, camera, length, np, Image) for camera in cameras}
    with h5py.File(paths["robot_only_hdf5_path"], "r") as stream:
        ro_record = _inspect_rgb(stream, "head_camera", length, np, Image)
    if ro_record != camera_records["head_camera"]:
        raise ValueError("paired robot_only RGB dimensions/frame counts must match scene head RGB")
    instruction = json.loads(paths["instruction_path"].read_text(encoding="utf-8"))
    seen = instruction.get("seen") if isinstance(instruction, dict) else None
    if not isinstance(seen, list) or not seen or any(not isinstance(item, str) or not item.strip() for item in seen):
        raise ValueError(f"instruction JSON requires a nonempty 'seen' string list: {paths['instruction_path']}")
    if type(prompt_index) is not int or not 0 <= prompt_index < len(seen):
        raise ValueError(f"prompt index {prompt_index} unavailable: {paths['instruction_path']}")
    after = {key: sha256_file(path) for key, path in paths.items()}
    if before != after:
        raise ValueError(f"episode source files changed during validation: {task}/episode{episode_index}")
    row = {
        "schema": PUBLIC_SCHEMA, "manifest_profile": "public", "dataset_id": PUBLIC_DATASET_ID,
        "source_revision": source_revision, "variant": PUBLIC_VARIANT, "split": split,
        "task": task, "episode_index": episode_index, "episode_id": f"{task}/episode{episode_index}",
        "frame_count": int(length), "action_dim": 14,
        "action_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
        "image_encoding": "robotwin_legacy_channel_reversed_jpeg",
        "cameras": camera_records, "robot_only_camera": ro_record,
        "prompt_seen": seen[prompt_index], "prompt_index": prompt_index,
        **relative,
        "hdf5_sha256": after["hdf5_path"],
        "robot_only_hdf5_sha256": after["robot_only_hdf5_path"],
        "instruction_sha256": after["instruction_path"],
    }
    validate_public_rows([row], split)
    return row


def validate_public_manifest_files(data_root, rows: list[dict], cameras=None) -> None:
    """Revalidate train-only source bytes and schema before normalization/training."""
    validate_public_rows(rows)
    root = validate_data_root(data_root)
    for row in rows:
        if cameras is not None and not set(cameras).issubset(row["cameras"]):
            raise ValueError(f"requested cameras were not pinned for {row['episode_id']}: {cameras}")
        actual = inspect_episode(root, row["task"], row["episode_index"], "train",
                                 cameras=tuple(row["cameras"]), source_revision=row["source_revision"],
                                 prompt_index=row["prompt_index"])
        for key in actual:
            if actual[key] != row.get(key):
                raise ValueError(f"public source/manifest mismatch for {row['episode_id']}: {key}")


def select_public_training_tasks(rows: list[dict], task_names=None) -> list[str]:
    """The manifest itself is the subset: reject a silently narrower task filter."""
    tasks = sorted({row["task"] for row in rows})
    if task_names is not None and (len(task_names) != len(set(task_names)) or set(task_names) != set(tasks)):
        raise ValueError(f"public task_names must exactly match manifest tasks {tasks}; prepare a new manifest for another subset")
    return tasks
