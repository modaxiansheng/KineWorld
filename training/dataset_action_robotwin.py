"""
Multi-camera dataset for joint RGB+Flow video + action training on RoboTwin.

Reads per-episode HDF5 files from the RoboTwin dataset (aloha-agilex_clean_50
variant across all 50 tasks) with 3 cameras (head, left, right).

Each sample provides:
  - Per-camera RGB video  (from observation/{camera}/rgb)
  - Per-camera Flow video (computed on-the-fly via real RAFT; full scene by default)
  - 14D absolute target qpos with global z-score normalization.

Temporal sampling: random start frame + num_frames consecutive frames per episode.
"""

import os
import sys
import json
import hashlib
import time
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def _normalize_variants(variants: Union[str, List[str]]) -> List[str]:
    """Accept a single string or a list/tuple of strings; return a list."""
    if isinstance(variants, str):
        return [variants]
    return list(variants)


def _cache_manifest_workers() -> int:
    env = os.environ.get("ROBOTWIN_CACHE_DISCOVERY_WORKERS")
    if env:
        return max(1, int(env))
    return min(32, max(1, (os.cpu_count() or 8)))


def _cache_chunk_stride() -> int:
    """Expected precompute chunk stride used to derive start_indices quickly."""
    return max(1, int(os.environ.get("ROBOTWIN_CACHE_CHUNK_STRIDE", "1")))


def _action_norm_workers() -> int:
    env = os.environ.get("ROBOTWIN_ACTION_NORM_WORKERS")
    if env:
        return max(1, int(env))
    return min(32, max(1, (os.cpu_count() or 8)))

# All flow / codec / shared-utils modules live next to this file, so no
# sys.path mutation is needed. The flow-prefix pipeline here is the single
# source of truth; training and inference must stay bit-aligned.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from reversible_flow_codec import FlowCodec
from raft_flow_extractor import RAFTFlowExtractor
from flow_prefix_utils import (
    add_bg_texture as _shared_add_bg_texture,
    mask_flows_by_robot as _shared_mask_flows_by_robot,
    process_camera_flow as _shared_process_camera_flow,
    process_camera_flow_full_scene as _shared_process_camera_flow_full_scene,
    tile_flow_with_white_wrist as _shared_tile_flow_with_white_wrist,
    tile_flow_t_shape as _shared_tile_flow_t_shape,
)



WRIST_MAG_RATIO: float = 1.5

FLOW_NOISE_THRESHOLD_PX: float = 0.5


# ---------------------------------------------------------------------------
# Multi-camera T-shape tiling
# ---------------------------------------------------------------------------

def tshape_tile(head_img, left_img, right_img):
    """T-shape spatial concatenation: head on top, left+right half-res below.

    Layout (pixels):
        +-----------+
        |   head    |  (orig_h x orig_w)
        +-----+-----+
        |left | right|  (orig_h//2 x orig_w)
        +-----+-----+

    Output shape: (orig_h + orig_h//2, orig_w, 3)
    """
    orig_h, orig_w = head_img.shape[:2]
    half_h, half_w = orig_h // 2, orig_w // 2
    left_half = cv2.resize(left_img, (half_w, half_h))
    right_half = cv2.resize(right_img, (half_w, half_h))
    bottom = np.hstack([left_half, right_half])
    return np.vstack([head_img, bottom])


ROBOTWIN_ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
    "blocks_ranking_size", "click_alarmclock", "click_bell",
    "dump_bin_bigbin", "grab_roller", "handover_block", "handover_mic",
    "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
    "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch",
]

TRACK1_FINAL_TRAIN_MANIFEST_SHA256 = (
    "b21baa6a699e8972a8a3f590c676bcf1471214f4f56a2ecad5d2c9e78e302497"
)
TRACK1_ROBOTWIN_SOURCE_REVISION = "9dc9299c163db059931898a9f0852098a61155a1"
TRACK1_TRAIN_VARIANT = "aloha-agilex_clean_50"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_track1_training_manifest(
    manifest_path: str,
    expected_sha256: str = TRACK1_FINAL_TRAIN_MANIFEST_SHA256,
) -> List[Dict]:
    """Load and strictly validate the 2,000-episode final-train allowlist."""
    if not manifest_path:
        raise ValueError(
            "training_manifest_path is required; the extracted RoboTwin tree "
            "must never be globbed for Track1 training"
        )
    if expected_sha256 != TRACK1_FINAL_TRAIN_MANIFEST_SHA256:
        raise ValueError(
            "only the audited final-train manifest SHA-256 is accepted: "
            f"{TRACK1_FINAL_TRAIN_MANIFEST_SHA256}"
        )
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"training manifest SHA-256 mismatch: {actual_sha256} != "
            f"{expected_sha256} ({manifest_path})"
        )

    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(
                    f"blank line in training manifest {manifest_path}:{line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in training manifest {manifest_path}:"
                    f"{line_number}: {error}"
                ) from error
            rows.append(row)

    if len(rows) != 2000:
        raise ValueError(
            f"final-train manifest must contain 2000 episodes, got {len(rows)}"
        )
    allowed_tasks = set(ROBOTWIN_ALL_TASKS)
    seen_ids = set()
    task_counts = {task: 0 for task in ROBOTWIN_ALL_TASKS}
    for row in rows:
        task = str(row.get("task", ""))
        try:
            episode_index = int(row["episode_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid episode_index in manifest row: {row!r}") from error
        episode_id = str(row.get("episode_id", ""))
        expected_id = f"{task}/episode{episode_index}"
        if task not in allowed_tasks or episode_id != expected_id:
            raise ValueError(f"invalid task/episode identity in manifest: {episode_id!r}")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode_id in training manifest: {episode_id}")
        if not 0 <= episode_index <= 39:
            raise ValueError(f"quarantined episode entered training manifest: {episode_id}")
        if row.get("dataset_id") != "robotwin2_clean50_official":
            raise ValueError(f"unexpected dataset_id for {episode_id}")
        if row.get("source_revision") != TRACK1_ROBOTWIN_SOURCE_REVISION:
            raise ValueError(f"unexpected source revision for {episode_id}")
        if row.get("split") != "train":
            raise ValueError(f"non-train split in final allowlist: {episode_id}")
        action_sha256 = str(row.get("action_sha256", ""))
        if len(action_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in action_sha256
        ):
            raise ValueError(f"invalid action_sha256 for {episode_id}")
        if int(row.get("frame_count", 0)) < 2:
            raise ValueError(f"invalid frame_count for {episode_id}")
        if not isinstance(row.get("prompt_seen"), str) or not row["prompt_seen"].strip():
            raise ValueError(f"missing manifest-pinned prompt_seen for {episode_id}")
        cameras = row.get("cameras")
        if not isinstance(cameras, dict) or "head_camera" not in cameras:
            raise ValueError(f"missing manifest camera records for {episode_id}")
        expected_basename = f"episode{episode_index}"
        if os.path.splitext(os.path.basename(str(row.get("hdf5_path", ""))))[0] != expected_basename:
            raise ValueError(f"hdf5_path identity mismatch for {episode_id}")
        if os.path.splitext(os.path.basename(str(row.get("instruction_path", ""))))[0] != expected_basename:
            raise ValueError(f"instruction_path identity mismatch for {episode_id}")
        seen_ids.add(episode_id)
        task_counts[task] += 1
    if set(task_counts.values()) != {40}:
        raise ValueError(
            "final-train manifest must contain exactly 40 episodes per task; "
            f"counts={task_counts}"
        )
    return sorted(rows, key=lambda row: (row["task"], int(row["episode_index"])))


def _manifest_episode_paths(data_root: str, row: Dict) -> Tuple[str, str, str]:
    task = str(row["task"])
    episode_name = f"episode{int(row['episode_index'])}"
    variant_dir = os.path.join(data_root, task, TRACK1_TRAIN_VARIANT)
    return (
        variant_dir,
        os.path.join(variant_dir, "data", f"{episode_name}.hdf5"),
        os.path.join(variant_dir, "instructions", f"{episode_name}.json"),
    )

# ---------------------------------------------------------------------------
# Background texture helpers — re-exported from the shared module so external
# callers that historically imported from this file keep working.
# ---------------------------------------------------------------------------

add_bg_texture = _shared_add_bg_texture
mask_flows_by_robot = _shared_mask_flows_by_robot


# ---------------------------------------------------------------------------
# Action normalization (global z-score)
# ---------------------------------------------------------------------------

@dataclass
class ActionNormStats:
    """Global mean/std statistics for z-score normalization of absolute qpos."""
    mean: np.ndarray
    std: np.ndarray
    training_manifest_sha256: Optional[str] = None

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        if self.mean.shape != (14,) or self.std.shape != (14,):
            raise ValueError(
                "action norm mean/std must each have shape (14,), got "
                f"{self.mean.shape}/{self.std.shape}"
            )
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("action norm mean/std must contain only finite values")
        if not (self.std > 0).all():
            raise ValueError("action norm std must be strictly positive in all 14 dimensions")

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.mean) / (self.std + 1e-6)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * (self.std + 1e-6) + self.mean

    def save(self, path: str):
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            training_manifest_sha256=np.asarray(
                self.training_manifest_sha256 or "", dtype=np.str_
            ),
        )

    @classmethod
    def load(
        cls,
        path: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> "ActionNormStats":
        with np.load(path, allow_pickle=False) as data:
            stored_sha256 = (
                str(data["training_manifest_sha256"].item())
                if "training_manifest_sha256" in data.files else ""
            )
            mean = np.array(data["mean"], dtype=np.float32, copy=True)
            std = np.array(data["std"], dtype=np.float32, copy=True)
        if expected_manifest_sha256 and stored_sha256 != expected_manifest_sha256:
            raise ValueError(
                f"action norm manifest mismatch: {stored_sha256 or '<missing>'} != "
                f"{expected_manifest_sha256} ({path})"
            )
        return cls(
            mean=mean,
            std=std,
            training_manifest_sha256=stored_sha256 or None,
        )


def _assemble_joint_vector(hdf5_file, frame_idx: int) -> np.ndarray:
    """Assemble 14D joint vector from HDF5 joint_action fields for one frame."""
    la = hdf5_file["joint_action/left_arm"][frame_idx]
    lg = hdf5_file["joint_action/left_gripper"][frame_idx]
    ra = hdf5_file["joint_action/right_arm"][frame_idx]
    rg = hdf5_file["joint_action/right_gripper"][frame_idx]
    lg_val = np.atleast_1d(lg).astype(np.float32)[:1]
    rg_val = np.atleast_1d(rg).astype(np.float32)[:1]
    return np.concatenate([
        np.asarray(la, dtype=np.float32),
        lg_val,
        np.asarray(ra, dtype=np.float32),
        rg_val,
    ])


def _assemble_joint_sequence(hdf5_file, frame_indices=None) -> np.ndarray:
    """Vectorized 14D joint sequence assembly from HDF5 joint_action fields."""
    la = np.asarray(hdf5_file["joint_action/left_arm"][:], dtype=np.float32)
    ra = np.asarray(hdf5_file["joint_action/right_arm"][:], dtype=np.float32)
    lg = np.asarray(hdf5_file["joint_action/left_gripper"][:], dtype=np.float32)
    rg = np.asarray(hdf5_file["joint_action/right_gripper"][:], dtype=np.float32)

    T = la.shape[0]
    lg = lg.reshape(T, -1)[:, :1]
    rg = rg.reshape(T, -1)[:, :1]
    qpos = np.concatenate([la, lg, ra, rg], axis=1).astype(np.float32, copy=False)
    if frame_indices is not None:
        qpos = qpos[np.asarray(frame_indices, dtype=np.int64)]
    return qpos


def _load_episode_qpos(hdf5_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        with h5py.File(hdf5_path, "r") as f:
            if f["joint_action/left_arm"].shape[0] < 1:
                return None, None
            return _assemble_joint_sequence(f), None
    except Exception as exc:
        return None, f"{hdf5_path}: {type(exc).__name__}: {exc}"


def compute_global_action_norm_stats(
    data_root: str,
    variants: Union[str, List[str]] = "aloha-agilex_clean_50",
    task_names: Optional[List[str]] = None,
    training_manifest_path: Optional[str] = None,
    training_manifest_sha256: str = TRACK1_FINAL_TRAIN_MANIFEST_SHA256,
) -> ActionNormStats:
    """Compute qpos statistics from the audited final-train allowlist only."""
    if task_names is None or len(task_names) == 0:
        task_names = ROBOTWIN_ALL_TASKS
    variants_list = _normalize_variants(variants)
    if variants_list != [TRACK1_TRAIN_VARIANT]:
        raise ValueError(
            "Track1 server training contains only aloha-agilex_clean_50; "
            f"requested variants={variants_list}"
        )
    manifest_rows = load_track1_training_manifest(
        training_manifest_path, training_manifest_sha256
    )

    task_names_sorted = sorted(task_names)
    requested_tasks = set(task_names_sorted)
    selected_rows = [row for row in manifest_rows if row["task"] in requested_tasks]
    all_episode_paths = []
    for row in selected_rows:
        _, hdf5_path, _ = _manifest_episode_paths(data_root, row)
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(
                f"manifest-allowed HDF5 is missing: {hdf5_path}"
            )
        all_episode_paths.append(hdf5_path)
    total_episodes = len(all_episode_paths)
    expected_episodes = 40 * len(requested_tasks)
    if total_episodes != expected_episodes:
        raise ValueError(
            f"manifest selected {total_episodes} episodes, expected "
            f"{expected_episodes} for {len(requested_tasks)} tasks"
        )

    print(
        f"[ActionNorm] Reading audited manifest allowlist: {total_episodes} "
        f"episodes across {len(requested_tasks)} tasks; sha256="
        f"{training_manifest_sha256}"
    )

    all_qpos = []
    errors = []
    workers = _action_norm_workers()
    print(f"[ActionNorm] Loading qpos with {workers} workers")
    pbar = tqdm(
        total=total_episodes,
        desc="[ActionNorm] Computing qpos stats",
        unit="ep",
        dynamic_ncols=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_load_episode_qpos, p) for p in all_episode_paths]
        for fut in as_completed(futures):
            qpos, err = fut.result()
            if qpos is not None:
                all_qpos.append(qpos)
            if err is not None:
                errors.append(err)
            pbar.update(1)
    pbar.close()

    for err in errors[:20]:
        print(f"\n  Warning: failed to read {err}")
    if len(errors) > 20:
        print(f"\n  Warning: suppressed {len(errors) - 20} additional read errors")
    if errors or len(all_qpos) != total_episodes:
        raise RuntimeError(
            f"action normalization requires all {total_episodes} allowlisted "
            f"episodes; loaded={len(all_qpos)} errors={len(errors)}"
        )

    if not all_qpos:
        raise ValueError(
            f"No action data found in {data_root} for variants={variants_list}"
        )

    all_qpos = np.concatenate(all_qpos, axis=0)
    stats = ActionNormStats(
        mean=all_qpos.mean(axis=0).astype(np.float32),
        std=all_qpos.std(axis=0).astype(np.float32),
        training_manifest_sha256=training_manifest_sha256,
    )
    print(f"[ActionNorm] Computed global z-score stats from "
          f"{all_qpos.shape[0]} qpos steps, "
          f"action_dim={all_qpos.shape[1]}")

    print(f"[ActionNorm] mean={stats.mean}")
    print(f"[ActionNorm] std={stats.std}")
    return stats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RoboTwinActionFlowDataset:
    """Multi-camera RGB+Flow+Action dataset for RoboTwin.

    Parameters
    ----------
    data_root : str
        Path to ``robotwin/dataset/`` containing per-task subdirectories.
    variant : str
        Robot variant directory name.
    cameras : list of str
        Camera names to use.
    size : tuple of (width, height)
        Target resolution for output frames.
    num_frames : int
        Number of output video frames (must satisfy Wan 4N+1).
    task_names : list of str, optional
        Subset of tasks. None = all 50 tasks.
    flow_method : str
        ``"raft"`` or ``"farneback"``.
    flow_device : str
        Device for RAFT model.
    flow_max_magnitude : float or None
        Fixed magnitude cap for FlowCodec on the *head* camera. The wrist
        cameras use ``flow_max_magnitude * WRIST_MAG_RATIO`` automatically
        (see the module-level ``WRIST_MAG_RATIO`` constant). Pass ``None``
        for percentile-adaptive normalization (debug only).
    flow_mode : str
        ``"full_scene"`` (default) — RAFT runs on the raw RGB of all three
        cameras and the resulting per-camera flow is T-shape tiled to
        match the RGB stream. Drops the dependency on the ``robot_only``
        HDF5 render.
        ``"robot_only"`` (legacy) — head flow on robot-only frames, white
        wrist placeholder; kept for backward compatibility / debugging.
        Requires the ``robot_only/data/*.hdf5`` companion files.
    action_norm_stats : ActionNormStats or None
        Pre-computed normalization stats.
    action_norm_path : str or None
        Path to save/load ``action_norm_stats.npz``.
    """

    CAMERA_PREFIX = (
        "A multi-view video of an aloha robot in T-shape layout: "
        "the top row shows the full-size rear camera view, "
        "the bottom-left shows the half-size left arm camera view, "
        "and the bottom-right shows the half-size right arm camera view. "
        "The robot is performing the following task: "
    )
    TRACK1_PROMPT_PREFIX = (
        "In a fixed robotic workspace, generate a rigid, physically consistent "
        "embodied robotic arm. The arm maintains high stability with no "
        "deformation and enters the frame to "
    )

    def __init__(
        self,
        data_root: str,
        variants: Union[str, List[str]] = "aloha-agilex_clean_50",
        cameras: Optional[List[str]] = None,
        flow_cameras: Optional[List[str]] = None,
        size: Tuple[int, int] = (320, 240),
        num_frames: int = 49,
        visual_stride: int = 1,
        num_video_frames: Optional[int] = None,
        task_names: Optional[List[str]] = None,
        flow_method: str = "raft",
        flow_device: str = "cuda",
        flow_max_magnitude: Optional[float] = None,
        flow_mode: str = "robot_only",
        action_norm_stats: Optional[ActionNormStats] = None,
        action_norm_path: Optional[str] = None,
        training_manifest_path: Optional[str] = None,
        training_manifest_sha256: str = TRACK1_FINAL_TRAIN_MANIFEST_SHA256,
        camera_prefix: Optional[str] = None,
        track1_prompt_template: bool = True,
        load_from_cache: bool = False,
        cache_root: Optional[str] = None,
        cache_episode_lru_size: int = 0,
    ):
        self.data_root = data_root
        self.variants = _normalize_variants(variants)
        if self.variants != [TRACK1_TRAIN_VARIANT]:
            raise ValueError(
                "Track1 training is manifest-locked to aloha-agilex_clean_50; "
                f"requested variants={self.variants}"
            )
        self.training_manifest_path = training_manifest_path
        self.training_manifest_sha256 = training_manifest_sha256
        self.training_manifest_rows = load_track1_training_manifest(
            training_manifest_path, training_manifest_sha256
        )
        self._manifest_rows_by_episode = {
            (str(row["task"]), f"episode{int(row['episode_index'])}"): row
            for row in self.training_manifest_rows
        }
        self.cameras = cameras or ["head_camera", "left_camera", "right_camera"]
        self.num_cameras = len(self.cameras)
        self.flow_cameras = flow_cameras or [self.cameras[0]]
        self.flow_camera_set = set(self.flow_cameras)
        self.size = size
        # ``num_frames`` is the ACTION sequence length (kept high-frequency).
        self.num_frames = num_frames
        # --- Visual temporal downsampling (decouple video FPS from action FPS) ---
        # Default (visual_stride=1, num_video_frames=None) reproduces the legacy
        # 1:1 behavior (one video frame == one action step). Setting e.g.
        # visual_stride=4, num_video_frames=13 makes 13 stride-4 video frames span
        # the same 49-step horizon as the high-freq action chunk -> 1 video frame
        # : 4 actions. RAFT flow is then computed across the stride-4 frames, so
        # ``flow_max_magnitude`` must be re-profiled (≈21 for head at stride 4).
        self.visual_stride = max(1, int(visual_stride))
        self.num_video_frames = (
            int(num_video_frames) if num_video_frames is not None else num_frames
        )
        self.flow_method = flow_method
        self.flow_device = flow_device

        assert flow_mode in ("full_scene", "robot_only"), (
            f"flow_mode must be 'full_scene' or 'robot_only', got {flow_mode!r}"
        )
        self.flow_mode = flow_mode
        self.flow_max_magnitude = flow_max_magnitude
        self.flow_max_magnitude_wrist = (
            flow_max_magnitude * WRIST_MAG_RATIO
            if flow_max_magnitude is not None else None
        )

        self.camera_prefix = camera_prefix if camera_prefix is not None else self.CAMERA_PREFIX
        self.track1_prompt_template = bool(track1_prompt_template)
        self.load_from_cache = load_from_cache
        self.cache_root = cache_root
        self.cache_episode_lru_size = max(0, int(cache_episode_lru_size))
        self._episode_cache = OrderedDict()
        self._episode_cache_hits = 0
        self._episode_cache_misses = 0
        self._episode_cache_log_every = max(
            0, int(os.environ.get("ROBOTWIN_CACHE_EPISODE_LRU_LOG_EVERY", "0"))
        )
        self._cache_skip_count = 0
        self._cache_skip_warn_limit = 20
        self._cache_skip_warn_interval = 100

        self.codec = FlowCodec()
        self._flow_extractor = None

        if task_names is None or len(task_names) == 0:
            task_names = ROBOTWIN_ALL_TASKS
        unknown_tasks = sorted(set(task_names) - set(ROBOTWIN_ALL_TASKS))
        if unknown_tasks:
            raise ValueError(
                f"requested tasks are absent from the audited manifest: {unknown_tasks}"
            )
        self.task_names = task_names

        if self.load_from_cache:
            assert cache_root is not None, "cache_root required when load_from_cache=True"
            self._cached_chunks = self._discover_cached_chunks()
            print(f"[RoboTwinActionFlow] CACHE MODE: {len(self._cached_chunks)} chunks "
                  f"from cache_root={cache_root}, "
                  f"episode_lru_size={self.cache_episode_lru_size}")
        else:
            self.samples = self._discover_episodes()
            print(f"[RoboTwinActionFlow] {len(self.samples)} episodes from "
                  f"{len(set(s['task'] for s in self.samples))} tasks x "
                  f"{len(self.variants)} variants, "
                  f"cameras={self.cameras}, flow_cameras={self.flow_cameras}, "
                  f"variants={self.variants}, size={size}, num_frames={num_frames}, "
                  f"flow_mode={flow_mode}, "
                  f"flow_max_mag(head)={flow_max_magnitude}, "
                  f"flow_max_mag(wrist)={self.flow_max_magnitude_wrist} "
                  f"(wrist = head * {WRIST_MAG_RATIO}), "
                  f"flow_noise_threshold_px={FLOW_NOISE_THRESHOLD_PX}")

        if action_norm_stats is not None:
            if (
                action_norm_stats.training_manifest_sha256
                != self.training_manifest_sha256
            ):
                raise ValueError(
                    "provided action_norm_stats is not bound to the audited "
                    "training manifest"
                )
            self.action_norm = action_norm_stats
        elif action_norm_path is not None and os.path.exists(action_norm_path):
            self.action_norm = ActionNormStats.load(
                action_norm_path,
                expected_manifest_sha256=self.training_manifest_sha256,
            )
            print(f"[ActionNorm] Loaded from {action_norm_path}")
        else:
            self.action_norm = compute_global_action_norm_stats(
                data_root,
                self.variants,
                task_names,
                training_manifest_path=self.training_manifest_path,
                training_manifest_sha256=self.training_manifest_sha256,
            )
            if action_norm_path is not None:
                os.makedirs(os.path.dirname(action_norm_path) or ".", exist_ok=True)
                self.action_norm.save(action_norm_path)
                print(f"[ActionNorm] Saved to {action_norm_path}")

    def _discover_episodes(self) -> List[Dict]:
        samples = []
        require_robot_only = self.flow_mode == "robot_only"
        requested_tasks = set(self.task_names)
        for row in self.training_manifest_rows:
            task = row["task"]
            if task not in requested_tasks:
                continue
            variant_dir, hdf5_path, instruction_path = _manifest_episode_paths(
                self.data_root, row
            )
            ep_name = f"episode{int(row['episode_index'])}"
            robot_only_dir = os.path.join(variant_dir, "robot_only", "data")
            ro_path = os.path.join(robot_only_dir, f"{ep_name}.hdf5")
            if not os.path.isfile(hdf5_path):
                raise FileNotFoundError(
                    f"manifest-allowed HDF5 is missing: {hdf5_path}"
                )
            if not os.path.isfile(instruction_path):
                raise FileNotFoundError(
                    f"manifest-allowed instruction is missing: {instruction_path}"
                )
            if require_robot_only and not os.path.isfile(ro_path):
                raise FileNotFoundError(
                    "flow_mode='robot_only' requires a paired allowlisted render: "
                    f"{ro_path}"
                )

            samples.append({
                "task": task,
                "variant": TRACK1_TRAIN_VARIANT,
                "episode_name": ep_name,
                "data_hdf5": hdf5_path,
                "robot_only_hdf5": ro_path if os.path.isfile(ro_path) else None,
                "variant_dir": variant_dir,
                "prompt_seen": row["prompt_seen"],
            })

        if not samples:
            raise ValueError(
                f"No valid episodes found in {self.data_root} "
                f"for variants={self.variants}"
            )
        return samples

    # -- Cache discovery and loading --

    def _cache_manifest_path(self) -> str:
        """Manifest path for this cache_root/tasks/variants view.

        Default location: ``${cache_root}/.manifests/cached_chunks_<digest>.json``.

        Multi-node override (``ROBOTWIN_CACHE_MANIFEST_DIR``): when each
        rank has its own LOCAL copy of ``cache_root`` (e.g. node-local
        ext4 mounts), rank 0's manifest write is invisible to ranks
        1..N-1, who would block forever in ``_load_or_build_cache_manifest``
        waiting for the JSON to appear. Set this env var to a cross-node-
        visible path (PFS / NFS / shared FS) and all ranks will read/write
        the SAME manifest file. The hashed digest still encodes
        ``cache_root`` so different cache roots written to the same
        shared dir do not collide.
        """
        payload = {
            "cache_root": os.path.abspath(self.cache_root),
            "data_root": os.path.abspath(self.data_root),
            "tasks": sorted(self.task_names),
            "variants": sorted(self.variants),
            "training_manifest_sha256": self.training_manifest_sha256,
            "num_frames": int(self.num_frames),
            "chunk_stride": _cache_chunk_stride(),
            "discovery_mode": os.environ.get("ROBOTWIN_CACHE_DISCOVERY_MODE", "hdf5"),
            # v3: tail-padded chunk starts (n_chunks = max(1, T - 1)) — bumped
            # in lockstep with precompute_latents.py PHASE2_VERSION=3.
            # Old v2 manifests are simply not reused (the hash changes); old
            # v2 .pt latents are invalidated by _phase2_done and rebuilt on
            # the next precompute run.
            "version": 3,
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        manifest_dir_override = os.environ.get("ROBOTWIN_CACHE_MANIFEST_DIR")
        if manifest_dir_override:
            manifest_dir = os.path.abspath(manifest_dir_override)
        else:
            manifest_dir = os.path.join(self.cache_root, ".manifests")
        return os.path.join(manifest_dir, f"cached_chunks_{digest}.json")

    def _read_cache_episode_meta(self, item: Tuple[str, str, str, str]) -> Dict:
        """Read cheap episode-level metadata for one cached episode.

        Fast path derives chunk starts from the original HDF5 episode length,
        avoiding ``torch.load`` on large .pt files during startup. This is
        equivalent for caches produced by ``precompute_latents.py``
        v3+ (tail-padded): ``n_chunks = max(1, T - 1)`` and
        ``start_indices = range(0, n_chunks, chunk_stride)``. Trailing chunks
        whose 49-frame window runs past the episode end are padded by
        clamping frame indices to ``T-1`` (the on-disk latents already bake
        this in; ``_load_actions`` does the same clamp on the read side, so
        RGB / flow / qpos stay aligned).

        Set ``ROBOTWIN_CACHE_DISCOVERY_MODE=pt`` to fall back to reading
        ``num_chunks/start_indices`` from the .pt files for debugging.
        """
        task, variant, ep_name, pt_path = item
        hdf5_path = os.path.join(
            self.data_root, task, variant, "data", f"{ep_name}.hdf5",
        )
        try:
            mode = os.environ.get("ROBOTWIN_CACHE_DISCOVERY_MODE", "hdf5")
            if mode == "pt":
                try:
                    meta = torch.load(
                        pt_path, map_location="cpu", weights_only=False, mmap=True,
                    )
                except TypeError:
                    meta = torch.load(pt_path, map_location="cpu", weights_only=False)

                n_chunks = int(meta["num_chunks"])
                start_indices = meta.get("start_indices", list(range(n_chunks)))
                if len(start_indices) != n_chunks:
                    raise ValueError(
                        f"start_indices len {len(start_indices)} != num_chunks {n_chunks}"
                    )
                start_indices = [int(x) for x in start_indices]
            else:
                with h5py.File(hdf5_path, "r") as f:
                    head_cam = self.cameras[0]
                    if f"observation/{head_cam}/rgb" in f:
                        T = f[f"observation/{head_cam}/rgb"].shape[0]
                    else:
                        T = f["joint_action/left_arm"].shape[0]
                n_chunks = max(1, int(T) - 1)
                stride = _cache_chunk_stride()
                start_indices = list(range(0, n_chunks, stride))
                if not start_indices:
                    start_indices = [0]

            if not os.path.exists(pt_path):
                raise FileNotFoundError(pt_path)
            if not os.path.exists(hdf5_path):
                raise FileNotFoundError(hdf5_path)

            return {
                "ok": True,
                "episode": {
                    "pt_path": pt_path,
                    "task": task,
                    "variant": variant,
                    "episode_name": ep_name,
                    "hdf5_path": hdf5_path,
                    "start_indices": start_indices,
                },
            }
        except Exception as exc:
            return {
                "ok": False,
                "pt_path": pt_path,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _build_cache_episode_manifest(self, records: List[Tuple[str, str, str, str]]) -> Dict:
        workers = _cache_manifest_workers()
        mode = os.environ.get("ROBOTWIN_CACHE_DISCOVERY_MODE", "hdf5")
        stride = _cache_chunk_stride()
        episodes = []
        errors = []
        print(
            f"[CacheManifest] Building cache manifest from {len(records)} files "
            f"with {workers} workers "
            f"(mode={mode}, chunk_stride={stride}, num_frames={self.num_frames})"
        )
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._read_cache_episode_meta, item) for item in records]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[CacheManifest] reading episode metadata",
                unit="ep",
                dynamic_ncols=True,
            ):
                result = fut.result()
                if result["ok"]:
                    episodes.append(result["episode"])
                else:
                    errors.append(result)

        episodes.sort(key=lambda e: (e["task"], e["variant"], e["episode_name"]))
        for err in errors[:20]:
            print(f"[Cache] WARN failed to read {err['pt_path']}: {err['error']}")
        if len(errors) > 20:
            print(f"[Cache] WARN suppressed {len(errors) - 20} additional cache-read errors")
        if errors:
            raise RuntimeError(
                f"failed to read {len(errors)} manifest-allowed cache episodes; "
                "partial cache training is forbidden"
            )
        return {
            "version": 3,
            "episodes": episodes,
            "num_errors": 0,
            "training_manifest_sha256": self.training_manifest_sha256,
        }

    @staticmethod
    def _write_json_atomic(path: str, obj: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    def _load_or_build_cache_manifest(self, records: List[Tuple[str, str, str, str]]) -> Dict:
        """Build once on global rank0, all other ranks wait and reuse it."""
        manifest_path = self._cache_manifest_path()
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        refresh = os.environ.get("ROBOTWIN_CACHE_MANIFEST_REFRESH", "0") == "1"

        if rank == 0 and (refresh or not os.path.exists(manifest_path)):
            manifest = self._build_cache_episode_manifest(records)
            self._write_json_atomic(manifest_path, manifest)
            print(
                f"[CacheManifest] Wrote {len(manifest['episodes'])} episodes to "
                f"{manifest_path}"
            )
            return manifest

        if not os.path.exists(manifest_path):
            if world_size <= 1:
                manifest = self._build_cache_episode_manifest(records)
                self._write_json_atomic(manifest_path, manifest)
                return manifest

            timeout_s = int(os.environ.get("ROBOTWIN_CACHE_MANIFEST_TIMEOUT", "7200"))
            t0 = time.time()
            print(f"[CacheManifest][rank {rank}] waiting for {manifest_path}")
            while not os.path.exists(manifest_path):
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(
                        f"Timed out waiting for cache manifest {manifest_path}"
                    )
                time.sleep(2)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("training_manifest_sha256") != self.training_manifest_sha256:
            raise ValueError(
                f"cache manifest allowlist mismatch: {manifest_path}"
            )
        if manifest.get("num_errors") != 0:
            raise ValueError(f"cache manifest contains read errors: {manifest_path}")
        if len(manifest.get("episodes", [])) != len(records):
            raise ValueError(
                f"cache manifest episode count mismatch: "
                f"{len(manifest.get('episodes', []))} != {len(records)}"
            )
        print(
            f"[CacheManifest][rank {rank}] Loaded {len(manifest['episodes'])} "
            f"episodes from {manifest_path}"
        )
        return manifest

    def _discover_cached_chunks(self) -> List[Dict]:
        """Read only allowlisted cache files, then flatten them into chunks.

        Loads ``num_chunks``, ``start_indices``, and ``instructions`` once per
        episode (we skip loading the heavy latent tensors here — those are
        read lazily in ``_load_chunk_from_cache``). Iterates all
        ``task x variant`` combinations in ``self.task_names x self.variants``.
        """
        records = []
        requested_tasks = set(self.task_names)
        for row in self.training_manifest_rows:
            task = row["task"]
            if task not in requested_tasks:
                continue
            ep_name = f"episode{int(row['episode_index'])}"
            pt_path = os.path.join(
                self.cache_root, task, TRACK1_TRAIN_VARIANT, f"{ep_name}.pt"
            )
            if not os.path.isfile(pt_path):
                raise FileNotFoundError(
                    f"manifest-allowed cache episode is missing: {pt_path}"
                )
            records.append((task, TRACK1_TRAIN_VARIANT, ep_name, pt_path))

        manifest = self._load_or_build_cache_manifest(records)

        chunks = []
        episodes = manifest.get("episodes", [])
        # --- Guard: cache MUST match this dataset's downsample config ---
        # The manifest fast-path derives chunk starts from HDF5 length and does
        # NOT read latent metadata, so a stale/mismatched cache (a legacy 1:1
        # cache, wrong visual_stride, or a different chunk_stride) would be
        # loaded silently with wrong-shaped latents. Probe ONE .pt and assert.
        if episodes:
            _probe = torch.load(
                episodes[0]["pt_path"], map_location="cpu", weights_only=False,
            )
            _cvs = int(_probe.get("visual_stride", 1))
            _cnvf = int(_probe.get("num_video_frames",
                                   _probe.get("num_frames", self.num_frames)))
            _ccs = int(_probe.get("chunk_stride", 1))
            _cver = int(_probe.get("phase2_version", 1))
            _mstride = _cache_chunk_stride()
            if _cvs != self.visual_stride or _cnvf != self.num_video_frames:
                raise ValueError(
                    f"[Cache config MISMATCH] {episodes[0]['pt_path']}: cache "
                    f"visual_stride={_cvs}/num_video_frames={_cnvf} != dataset "
                    f"visual_stride={self.visual_stride}/"
                    f"num_video_frames={self.num_video_frames}. This CACHE_ROOT was "
                    f"built with a DIFFERENT (e.g. legacy 1:1) config; point it at a "
                    f"matching cache or re-run precompute with these settings."
                )
            if _ccs != _mstride:
                raise ValueError(
                    f"[Cache stride MISMATCH] cache chunk_stride={_ccs} != manifest "
                    f"stride={_mstride}; set ROBOTWIN_CACHE_CHUNK_STRIDE={_ccs} so "
                    f"start_indices are derived correctly."
                )
            print(
                f"[RoboTwinActionFlow] Cache config OK: visual_stride={_cvs}, "
                f"num_video_frames={_cnvf}, chunk_stride={_ccs}, phase2_version={_cver}"
            )
        for ep in episodes:
            for ci, start in enumerate(ep["start_indices"]):
                chunks.append({
                    "pt_path": ep["pt_path"],
                    "chunk_idx": ci,
                    "start": int(start),
                    "task": ep["task"],
                    "variant": ep["variant"],
                    "episode_name": ep["episode_name"],
                    "hdf5_path": ep["hdf5_path"],
                    "prompt_seen": self._manifest_rows_by_episode[
                        (ep["task"], ep["episode_name"])
                    ]["prompt_seen"],
                })
        if not chunks:
            raise ValueError(
                f"No cached chunks found in {self.cache_root} "
                f"for variants={self.variants}"
            )
        print(f"[RoboTwinActionFlow] Cache: {len(episodes)} episodes across "
              f"{len(self.variants)} variants, {len(chunks)} total chunks")
        return chunks

    @staticmethod
    def _pad_to(t: torch.Tensor, n: int) -> torch.Tensor:
        """Zero-pad a (seq_len, D) tensor on dim 0 to length n. No-op if equal."""
        L = t.shape[0]
        if L == n:
            return t
        if L > n:
            raise ValueError(f"_pad_to: tensor length {L} > target {n}")
        D = t.shape[1]
        out = torch.zeros((n, D), dtype=t.dtype, device=t.device)
        out[:L] = t
        return out

    def _load_cached_episode_pt(self, pt_path: str) -> Dict:
        """Load an episode cache file, optionally reusing a per-process LRU."""
        if self.cache_episode_lru_size <= 0:
            return torch.load(pt_path, map_location="cpu", weights_only=False)

        cached = self._episode_cache.get(pt_path)
        if cached is not None:
            self._episode_cache_hits += 1
            self._episode_cache.move_to_end(pt_path)
            return cached

        self._episode_cache_misses += 1
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self._episode_cache[pt_path] = data
        self._episode_cache.move_to_end(pt_path)
        while len(self._episode_cache) > self.cache_episode_lru_size:
            self._episode_cache.popitem(last=False)

        log_every = self._episode_cache_log_every
        total = self._episode_cache_hits + self._episode_cache_misses
        if log_every > 0 and total % log_every == 0:
            hit_rate = self._episode_cache_hits / max(total, 1)
            print(
                f"[RoboTwinActionFlow][pid {os.getpid()}] episode LRU "
                f"hits={self._episode_cache_hits} "
                f"misses={self._episode_cache_misses} "
                f"hit_rate={hit_rate:.3f} "
                f"size={len(self._episode_cache)}/{self.cache_episode_lru_size}"
            )

        return data

    def  _load_chunk_from_cache(self, idx: int) -> Dict:
        """Load one pre-encoded chunk by global index.

        Latents come from the cached .pt; actions are read live from HDF5
        and normalized via the dataset's action_norm. If the cache also
        contains pre-encoded T5 embeddings (``video_contexts`` /
        ``action_contexts`` produced by precompute_latents.py
        Phase 1), a matching random instruction is picked and the per-item
        trimmed tensors are zero-padded to (512, 4096) so the collate
        function can stack them directly, letting the trainer skip
        ``text_encoder`` entirely.
        """
        info = self._cached_chunks[idx]
        data = self._load_cached_episode_pt(info["pt_path"])
        ci = info["chunk_idx"]

        instructions = data.get("instructions") or []
        n_instr = len(instructions)
        has_cached_text = (
            "video_contexts" in data and "action_contexts" in data
        )
        if self.track1_prompt_template:
            # Old upstream caches contain T-shape video text embeddings. They
            # are intentionally ignored for Track1; the raw cached latents are
            # still valid and the trainer re-encodes this head-only prompt.
            has_cached_text = (
                has_cached_text
                and data.get("track1_prompt_prefix") == self.TRACK1_PROMPT_PREFIX
            )
        if has_cached_text:
            # Fail loudly on schema drift so we don't silently train on stale
            # embeddings if CAMERA_PREFIX changes in the codebase.
            assert data.get("camera_prefix") == self.camera_prefix, (
                f"cache camera_prefix mismatch for {info['pt_path']} — "
                f"recompute the text cache (Phase 1) or override "
                f"camera_prefix to match."
            )
            assert len(data["video_contexts"]) == n_instr, (
                f"video_contexts len {len(data['video_contexts'])} != "
                f"instructions len {n_instr} in {info['pt_path']}"
            )
            assert len(data["action_contexts"]) == n_instr, (
                f"action_contexts len {len(data['action_contexts'])} != "
                f"instructions len {n_instr} in {info['pt_path']}"
            )

        # The audited manifest pins one seen prompt per episode. Never sample
        # the mutable local instruction JSON/cache list for Track1 training.
        instruction = info["prompt_seen"]
        if has_cached_text and instruction in instructions:
            i = instructions.index(instruction)
        else:
            has_cached_text = False
            i = -1

        with h5py.File(info["hdf5_path"], "r") as f_data:
            actions = self._load_actions(f_data, info["start"], self.num_frames)

        out = {
            "rgb_input_latents":  data["rgb_input_latents"][ci],
            "flow_input_latents": data["flow_input_latents"][ci],
            "actions":            actions,
            "video_prompt":       self._format_video_prompt(instruction),
            "action_prompt":      (
                self._format_video_prompt(instruction)
                if self.track1_prompt_template else instruction
            ),
            "task":               info["task"],
            "variant":            info.get("variant", ""),
            "num_frames_pixel":   int(data.get("num_frames", self.num_frames)),
        }

        if has_cached_text:
            vctx = data["video_contexts"][i]      # (seq_len_v, 4096) bf16
            actx = data["action_contexts"][i]     # (seq_len_a, 4096) bf16
            out["video_context"] = self._pad_to(vctx, 512)
            out["action_context"] = self._pad_to(actx, 512)

        self._validate_cached_chunk(out, info, ci)
        return out

    @staticmethod
    def _validate_cached_chunk(out: Dict, info: Dict, chunk_idx: int):
        """Fail fast on malformed cache entries before collate/model code."""
        rgb = out["rgb_input_latents"]
        flow = out["flow_input_latents"]
        if not isinstance(rgb, torch.Tensor) or not isinstance(flow, torch.Tensor):
            raise TypeError("cached latents must be torch.Tensor")
        if rgb.shape != flow.shape:
            raise ValueError(
                f"rgb/flow latent shape mismatch at chunk {chunk_idx}: "
                f"{tuple(rgb.shape)} vs {tuple(flow.shape)}"
            )
        if rgb.ndim != 4:
            raise ValueError(
                f"expected latent shape (C,T,H,W), got {tuple(rgb.shape)} "
                f"for {info['pt_path']} chunk {chunk_idx}"
            )
        if out["actions"].shape[0] <= 0:
            raise ValueError(f"empty actions for {info['pt_path']} chunk {chunk_idx}")
        if "video_context" in out and tuple(out["video_context"].shape) != (512, 4096):
            raise ValueError(
                f"bad video_context shape {tuple(out['video_context'].shape)} "
                f"for {info['pt_path']}"
            )
        if "action_context" in out and tuple(out["action_context"].shape) != (512, 4096):
            raise ValueError(
                f"bad action_context shape {tuple(out['action_context'].shape)} "
                f"for {info['pt_path']}"
            )

    def _warn_cache_skip(self, idx: int, info: Optional[Dict], exc: Exception):
        self._cache_skip_count += 1
        n = self._cache_skip_count
        if n > self._cache_skip_warn_limit and n % self._cache_skip_warn_interval != 0:
            return

        rank = os.environ.get("RANK", "?")
        if info is None:
            where = f"idx={idx}"
        else:
            where = (
                f"idx={idx}, pt={info.get('pt_path')}, "
                f"chunk_idx={info.get('chunk_idx')}, start={info.get('start')}"
            )
        print(
            f"[CacheSkip][rank {rank}] skipped cached item #{n}: {where}; "
            f"{type(exc).__name__}: {exc}"
        )

    def _load_chunk_from_cache_safe(self, idx: int) -> Dict:
        info = None
        try:
            info = self._cached_chunks[idx]
            return self._load_chunk_from_cache(idx)
        except Exception as exc:
            self._warn_cache_skip(idx, info, exc)
            return {
                "__skip__": True,
                "error": f"{type(exc).__name__}: {exc}",
                "pt_path": info.get("pt_path", "") if info else "",
                "chunk_idx": info.get("chunk_idx", -1) if info else -1,
                "idx": idx,
            }

    # -- Frame loading helpers --

    def _load_hdf5_rgb_frame(self, hdf5_file, camera: str, idx: int) -> np.ndarray:
        jpeg_bytes = hdf5_file[f"observation/{camera}/rgb"][idx]
        img = Image.open(BytesIO(bytes(jpeg_bytes)))
        w, h = self.size
        img = img.resize((w, h), Image.BICUBIC)
        # hdf5 jpeg was cv2.imencode()-encoded from RGB frames (OpenCV treats
        # them as BGR), so PIL reads them back as BGR. Swap R/B to recover TRUE
        # RGB, matching pretrained WAN's colour space. MUST stay in sync with
        # precompute_latents.load_rgb_frame and the inference client.
        return np.ascontiguousarray(np.array(img, dtype=np.uint8)[..., ::-1])

    @staticmethod
    def _load_hdf5_rgb_frame_raw(hdf5_file, camera: str, idx: int) -> np.ndarray:
        jpeg_bytes = hdf5_file[f"observation/{camera}/rgb"][idx]
        img = Image.open(BytesIO(bytes(jpeg_bytes)))
        # See _load_hdf5_rgb_frame: undo the cv2.imencode BGR/RGB swap -> RGB.
        return np.ascontiguousarray(np.array(img.convert("RGB"), dtype=np.uint8)[..., ::-1])

    # -- Flow computation (delegates to the shared flow_prefix_utils module) --

    def _ensure_flow_extractor(self):
        if self.flow_method == "raft" and self._flow_extractor is None:
            self._flow_extractor = RAFTFlowExtractor(device=self.flow_device)
        return self._flow_extractor

    def _process_camera_flow(
        self, robot_only_frames_native: List[np.ndarray]
    ) -> Tuple[List[Image.Image], List[float]]:
        """Compute flow for one camera's robot-only frames (legacy pipeline)."""
        return _shared_process_camera_flow(
            robot_only_frames_native,
            target_size=self.size,
            codec=self.codec,
            flow_method=self.flow_method,
            raft_extractor=self._ensure_flow_extractor(),
            max_magnitude=self.flow_max_magnitude,
        )

    def _tile_flow_with_white_wrist(self, flow_pil_list):
        return _shared_tile_flow_with_white_wrist(flow_pil_list, target_size=self.size)

    def _process_camera_flow_full_scene(
        self,
        rgb_frames_native: List[np.ndarray],
        target_size: Tuple[int, int],
        max_magnitude: Optional[float],
    ) -> Tuple[List[Image.Image], List[float]]:
        """Full-scene flow for one camera (no bg-texture, no robot mask)."""
        return _shared_process_camera_flow_full_scene(
            rgb_frames_native,
            target_size=target_size,
            codec=self.codec,
            flow_method=self.flow_method,
            raft_extractor=self._ensure_flow_extractor(),
            max_magnitude=max_magnitude,
            noise_threshold=FLOW_NOISE_THRESHOLD_PX,
        )

    # -- Prompt --

    def _get_prompt(self, sample: Dict) -> str:
        """Build a text prompt from the task instruction.

        Uses the immutable ``prompt_seen`` value pinned in the audited JSONL.
        The mutable local instruction JSON is deliberately not consulted.
        """
        prompt = sample.get("prompt_seen")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"missing manifest-pinned prompt for {sample['task']}/"
                f"{sample['episode_name']}"
            )
        return prompt

    def _format_video_prompt(self, instruction: str) -> str:
        """Return an idempotent Track1 head-only prompt when enabled."""
        instruction = instruction.strip()
        if self.track1_prompt_template:
            if instruction.startswith(self.TRACK1_PROMPT_PREFIX):
                return instruction
            return self.TRACK1_PROMPT_PREFIX + instruction
        return self.camera_prefix + instruction

    # -- Action loading --

    def _load_actions(
        self, hdf5_file, start: int, num_frames: int
    ) -> np.ndarray:
        """Load absolute target qpos for frames [start, ..., start+num_frames-1].

        Frame indices >= T are right-clamped to T-1 for episodes shorter than
        ``num_frames``. Returns (num_frames, 14) float32 normalized absolute qpos.
        """
        T = hdf5_file["joint_action/left_arm"].shape[0]
        frame_indices = [min(T - 1, start + i) for i in range(num_frames)]
        targets = _assemble_joint_sequence(hdf5_file, frame_indices)
        return self.action_norm.normalize(targets).astype(np.float32)

    # -- Main --

    def __len__(self):
        if self.load_from_cache:
            return len(self._cached_chunks)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        if self.load_from_cache:
            return self._load_chunk_from_cache_safe(idx)

        sample = self.samples[idx]
        head_cam = self.cameras[0]
        multi_cam = len(self.cameras) >= 3

        f_data = h5py.File(sample["data_hdf5"], "r")
        f_ro = None
        if self.flow_mode == "robot_only":
            ro_path = sample.get("robot_only_hdf5")
            assert ro_path is not None, (
                f"flow_mode='robot_only' requires robot_only HDF5 but missing for "
                f"task={sample['task']} ep={sample['episode_name']}"
            )
            f_ro = h5py.File(ro_path, "r")

        try:
            T = f_data[f"observation/{head_cam}/rgb"].shape[0]
            # v3 tail-padded starts (match precompute / cache mode): every frame
            # except the very last (T-1) is a valid chunk start, so start spans
            # [0, T-2]. The OLD ``max_start = T - num_frames`` DROPPED the last
            # num_frames-1 start positions -> the model never saw near-end-of-
            # episode states -> OOD at inference once execution reaches the tail.
            max_start = max(0, T - 2)
            start = random.randint(0, max_start)

            # Visual / flow frames: low-FPS, downsampled by ``visual_stride``.
            # ``num_video_frames`` stride-``visual_stride`` frames span the same
            # horizon as the high-FPS action chunk below
            # (span = (num_video_frames-1)*visual_stride == num_frames-1 when
            # configured consistently). With the defaults (stride=1,
            # num_video_frames==num_frames) this is identical to the legacy
            # consecutive indexing. Trailing indices past the episode end are
            # right-clamped to T-1 (matching _load_actions), so the tail is
            # COVERED, not dropped.
            frame_indices = [
                min(T - 1, start + i * self.visual_stride)
                for i in range(self.num_video_frames)
            ]

            if multi_cam:
                head_frames = [
                    self._load_hdf5_rgb_frame(f_data, self.cameras[0], t)
                    for t in frame_indices
                ]
                left_frames = [
                    self._load_hdf5_rgb_frame(f_data, self.cameras[1], t)
                    for t in frame_indices
                ]
                right_frames = [
                    self._load_hdf5_rgb_frame(f_data, self.cameras[2], t)
                    for t in frame_indices
                ]
                tiled_frames = [
                    tshape_tile(h, l, r)
                    for h, l, r in zip(head_frames, left_frames, right_frames)
                ]
            else:
                tiled_frames = [
                    self._load_hdf5_rgb_frame(f_data, head_cam, t)
                    for t in frame_indices
                ]

            tiled_rgb_video = [Image.fromarray(f) for f in tiled_frames]

            if self.flow_mode == "full_scene":
                # Full-scene flow on all available cameras using the *native*
                # JPEG-decoded frames, then T-shape tile with real wrist
                # halves so the flow stream is spatially aligned with the
                # tiled RGB stream (no white placeholder bottom).
                head_size = self.size
                half_size = (self.size[0] // 2, self.size[1] // 2)

                head_native = [
                    self._load_hdf5_rgb_frame_raw(f_data, head_cam, t)
                    for t in frame_indices
                ]
                head_flow, _ = self._process_camera_flow_full_scene(
                    head_native,
                    target_size=head_size,
                    max_magnitude=self.flow_max_magnitude,
                )

                if multi_cam:
                    left_native = [
                        self._load_hdf5_rgb_frame_raw(f_data, self.cameras[1], t)
                        for t in frame_indices
                    ]
                    right_native = [
                        self._load_hdf5_rgb_frame_raw(f_data, self.cameras[2], t)
                        for t in frame_indices
                    ]
                    left_flow, _ = self._process_camera_flow_full_scene(
                        left_native,
                        target_size=half_size,
                        max_magnitude=self.flow_max_magnitude_wrist,
                    )
                    right_flow, _ = self._process_camera_flow_full_scene(
                        right_native,
                        target_size=half_size,
                        max_magnitude=self.flow_max_magnitude_wrist,
                    )
                    flow_pil = _shared_tile_flow_t_shape(
                        head_flow, left_flow, right_flow,
                        target_size=head_size,
                    )
                else:
                    flow_pil = head_flow
            else:
                # Legacy robot_only path: head flow on robot-only frames,
                # white placeholder for the wrist halves.
                robot_only_native = [
                    self._load_hdf5_rgb_frame_raw(f_ro, head_cam, t)
                    for t in frame_indices
                ]
                flow_pil, _ = self._process_camera_flow(robot_only_native)
                if multi_cam:
                    flow_pil = self._tile_flow_with_white_wrist(flow_pil)

            actions = self._load_actions(f_data, start, self.num_frames)
        finally:
            f_data.close()
            if f_ro is not None:
                f_ro.close()

        task_instruction = self._get_prompt(sample)

        return {
            "tiled_rgb_video": tiled_rgb_video,
            "flow_video": flow_pil,
            "actions": actions,
            "video_prompt": self._format_video_prompt(task_instruction),
            "action_prompt": (
                self._format_video_prompt(task_instruction)
                if self.track1_prompt_template else task_instruction
            ),
            "task": sample["task"],
        }
