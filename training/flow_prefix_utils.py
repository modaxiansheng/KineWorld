"""
Shared flow-prefix pipeline reused by training dataset and inference server.

The training dataset ([dataset_action_robotwin.py]) and the inference server
([flow_action_server.py]) MUST produce bit-identical flow latents for the
prefix frames; otherwise the action expert sees a different distribution at
deployment time. This module consolidates the entire pipeline so both call
sites use exactly the same code path.

Two pipelines are exposed:

* ``process_camera_flow`` (legacy ``robot_only`` mode):
    1. ``add_bg_texture``     — replace the (near-)solid background with a
                                static noise texture so RAFT can resolve
                                flow against it.
    2. RAFT optical flow      — pairwise flow between consecutive textured
                                frames (returns N-1 flows for N frames).
    3. ``mask_flows_by_robot`` — zero out flow on the original background
                                pixels of the *source* frame, so only
                                robot motion remains.
    4. Resize flow to target  — bilinear resize, rescale dx/dy.
    5. ``FlowCodec.encode``   — magnitude/angle -> RGB image, with a user-
                                supplied ``max_magnitude`` for normalization.
    6. White-flow sentinel    — prepend a fully white image at index 0.

* ``process_camera_flow_full_scene`` (new default, used when the dataset
    has ``flow_mode='full_scene'``):
    Same pipeline minus steps 1+3. RAFT runs on the raw (un-textured,
    un-masked) frames so object + robot motion is preserved. This is what
    the dual-stream world model also sees, removing the spatial
    inconsistency where the bottom region of the tiled image had RGB
    content but no flow. Optionally applies a magnitude threshold (the
    dataset uses 0.5 px at the encoded resolution — see
    ``FLOW_NOISE_THRESHOLD_PX`` in ``dataset_action_robotwin.py``)
    AFTER the resize step to suppress sub-pixel RAFT noise without
    touching real gripper / ego-motion signal.

For multi-camera T-shape tiling there are two helpers:

* ``tile_flow_with_white_wrist`` — head flow on top, white placeholders
    for the wrists (legacy, used with ``robot_only`` mode).
* ``tile_flow_t_shape`` — head flow on top, real wrist flow halves below
    (full-scene mode, mirrors ``tshape_tile`` for RGB).

NOTE: pipeline order, default tolerances, and the white-at-index-0
convention are intentionally identical to the original
``_process_camera_flow`` / ``_tile_flow_with_white_wrist`` methods of
``RoboTwinActionFlowDataset``. Any divergence breaks training/inference
alignment.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Background detection / texturing / robot-only masking
# ---------------------------------------------------------------------------

def detect_bg_color(frame: np.ndarray) -> np.ndarray:
    """Return the most-common quantized RGB color in ``frame`` (uint8 HxWx3)."""
    pixels = frame.reshape(-1, 3)
    quantized = (pixels // 4) * 4
    keys = (
        quantized[:, 0].astype(np.int32) * 65536
        + quantized[:, 1].astype(np.int32) * 256
        + quantized[:, 2].astype(np.int32)
    )
    counts = np.bincount(keys)
    mode_key = counts.argmax()
    r = (mode_key // 65536) & 0xFF
    g = (mode_key // 256) & 0xFF
    b = mode_key & 0xFF
    return np.array([r, g, b], dtype=np.uint8)


def make_static_texture(
    shape: Tuple[int, int, int],
    bg_color: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randint(-10, 11, size=shape, dtype=np.int16)
    return np.clip(bg_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_bg_texture(frames: List[np.ndarray], bg_tol: int = 10) -> List[np.ndarray]:
    """Replace solid-colored background pixels with a static noise texture.

    Background is detected from ``frames[0]``; the same texture is used for
    every frame so RAFT sees a stationary background pattern.
    """
    bg_color = detect_bg_color(frames[0])
    texture = make_static_texture(frames[0].shape, bg_color, seed=42)
    batch = np.stack(frames, axis=0)
    diff = np.max(np.abs(batch.astype(np.int16) - bg_color.astype(np.int16)), axis=-1)
    is_bg = diff <= bg_tol
    out = batch.copy()
    bg3 = np.broadcast_to(is_bg[..., None], out.shape)
    tex_broadcast = np.broadcast_to(texture[None], out.shape)
    np.copyto(out, tex_broadcast, where=bg3)
    return [out[i] for i in range(len(frames))]


def mask_flows_by_robot(
    flows: List[np.ndarray],
    frames: List[np.ndarray],
    bg_tol: int = 10,
) -> List[np.ndarray]:
    """Zero out flow on background pixels (as detected from ``frames[0]``).

    ``flows[i]`` is the flow whose source frame is ``frames[i]`` (i.e. flow
    from ``frames[i]`` to ``frames[i+1]``).
    """
    bg_color = detect_bg_color(frames[0])
    masked = []
    for i, flow in enumerate(flows):
        diff_src = np.max(
            np.abs(frames[i].astype(np.int16) - bg_color.astype(np.int16)), axis=-1
        )
        is_robot = diff_src > bg_tol
        flow_clean = np.zeros_like(flow)
        flow_clean[is_robot] = flow[is_robot]
        masked.append(flow_clean)
    return masked


# ---------------------------------------------------------------------------
# RAFT / Farneback flow + codec encoding
# ---------------------------------------------------------------------------

def resize_flow(flow: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize flow to ``(W, H)`` and rescale dx/dy proportionally."""
    h_orig, w_orig = flow.shape[:2]
    w_tgt, h_tgt = target_size
    if (w_orig, h_orig) == (w_tgt, h_tgt):
        return flow
    flow_resized = cv2.resize(flow, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
    flow_resized[..., 0] *= w_tgt / w_orig
    flow_resized[..., 1] *= h_tgt / h_orig
    return flow_resized


def compute_flows(
    frames: List[np.ndarray],
    flow_method: str = "raft",
    raft_extractor=None,
) -> List[np.ndarray]:
    """Pairwise flow between consecutive frames. Returns N-1 flows for N frames."""
    if flow_method == "raft":
        assert raft_extractor is not None, (
            "raft_extractor must be provided when flow_method='raft'"
        )
        return raft_extractor.batch_call(frames)
    return [
        cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        for i in range(len(frames) - 1)
    ]


def encode_flows_to_pil(
    codec,
    flows: List[np.ndarray],
    max_magnitude: Optional[float] = None,
) -> Tuple[List[Image.Image], List[float]]:
    pil_list = []
    max_mag_list = []
    for flow in flows:
        rgb, max_mag = codec.encode(flow, max_magnitude=max_magnitude)
        pil_list.append(Image.fromarray(rgb))
        max_mag_list.append(float(max_mag))
    return pil_list, max_mag_list


def threshold_flow_magnitude(
    flow: np.ndarray,
    noise_threshold: float,
) -> np.ndarray:
    """Zero out flow vectors with magnitude below ``noise_threshold`` (px).

    RAFT produces small but non-zero flow estimates on textureless
    regions (flat walls, table tops). A small threshold suppresses this
    sub-pixel noise without touching real manipulation/ego-motion flow.
    The dataset default (``FLOW_NOISE_THRESHOLD_PX = 0.5``) is set
    conservatively to preserve every pixel of gripper-finger / wrist
    ego-motion signal, even at the cost of leaving some weak background
    coloration for the model to learn to ignore. Larger values (1.0 +)
    clean the visualization more aggressively but start eroding small
    gripper motion, which matters for press-/click-style tasks.
    ``noise_threshold <= 0`` is a no-op.
    """
    if noise_threshold <= 0:
        return flow
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    keep = (mag >= noise_threshold)[..., None]
    return np.where(keep, flow, 0.0).astype(flow.dtype)


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------

def process_camera_flow(
    robot_only_frames_native: List[np.ndarray],
    target_size: Tuple[int, int],
    codec,
    flow_method: str = "raft",
    raft_extractor=None,
    max_magnitude: Optional[float] = None,
) -> Tuple[List[Image.Image], List[float]]:
    """Compute flow PIL images for one camera's robot-only frame sequence.

    Returns ``(flow_pil_list, max_mag_list)`` of the same length as the input
    frame list. ``flow_pil_list[0]`` is always the white sentinel; subsequent
    entries are RAFT/Farneback flows between consecutive robot-only frames,
    masked to robot pixels and codec-encoded.
    """
    textured = add_bg_texture(robot_only_frames_native)
    flows_native = compute_flows(textured, flow_method=flow_method,
                                 raft_extractor=raft_extractor)
    flows_native = mask_flows_by_robot(flows_native, robot_only_frames_native)
    flows_resized = [resize_flow(f, target_size) for f in flows_native]
    flow_pil_list, max_mag_list = encode_flows_to_pil(codec, flows_resized,
                                                      max_magnitude=max_magnitude)

    w, h = target_size
    zero_flow_pil = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
    flow_pil_list.insert(0, zero_flow_pil)
    max_mag_list.insert(0, 0.0)
    return flow_pil_list, max_mag_list


def tile_flow_with_white_wrist(
    flow_pil_list: List[Image.Image],
    target_size: Tuple[int, int],
) -> List[Image.Image]:
    """T-shape tile head-camera flow with white wrist placeholders.

    Mirrors ``tshape_tile`` so flow and RGB latents have identical
    spatial dimensions. ``target_size`` is the ``(W, H)`` of each input
    head-camera flow image; output is ``(W, H + H//2)``.

    Used by the legacy ``flow_mode='robot_only'`` path. For the full-scene
    multi-camera path, use ``tile_flow_t_shape`` instead.
    """
    w, h = target_size
    half_h = h // 2
    white_bottom = np.full((half_h, w, 3), 255, dtype=np.uint8)
    tiled = []
    for pil_img in flow_pil_list:
        head_arr = np.array(pil_img)
        tiled_arr = np.vstack([head_arr, white_bottom])
        tiled.append(Image.fromarray(tiled_arr))
    return tiled


# ---------------------------------------------------------------------------
# Full-scene multi-camera flow (default ``flow_mode='full_scene'``)
# ---------------------------------------------------------------------------

def process_camera_flow_full_scene(
    rgb_frames_native: List[np.ndarray],
    target_size: Tuple[int, int],
    codec,
    flow_method: str = "raft",
    raft_extractor=None,
    max_magnitude: Optional[float] = None,
    noise_threshold: float = 0.0,
) -> Tuple[List[Image.Image], List[float]]:
    """Compute full-scene flow PIL images for one camera's RGB sequence.

    Unlike ``process_camera_flow`` this does NOT add background texture
    nor mask flow to robot pixels — every pixel's RAFT flow (robot AND
    object motion) is encoded. Use this when the wrist-camera or whole-
    scene flow needs to stay spatially aligned with the RGB stream.

    ``noise_threshold`` (px, measured at ``target_size``): zero out flow
    vectors whose magnitude is below this value AFTER the resize step.
    Suppresses sub-pixel RAFT noise on textureless regions without
    discarding real motion. ``0.0`` disables the threshold.

    Returns ``(flow_pil_list, max_mag_list)`` of the same length as the
    input. ``flow_pil_list[0]`` is the white sentinel (zero motion at
    chunk start), matching the legacy convention.
    """
    flows_native = compute_flows(
        rgb_frames_native,
        flow_method=flow_method,
        raft_extractor=raft_extractor,
    )
    flows_resized = [resize_flow(f, target_size) for f in flows_native]
    if noise_threshold > 0:
        flows_resized = [
            threshold_flow_magnitude(f, noise_threshold) for f in flows_resized
        ]
    flow_pil_list, max_mag_list = encode_flows_to_pil(
        codec, flows_resized, max_magnitude=max_magnitude,
    )

    w, h = target_size
    zero_flow_pil = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))
    flow_pil_list.insert(0, zero_flow_pil)
    max_mag_list.insert(0, 0.0)
    return flow_pil_list, max_mag_list


def tile_flow_t_shape(
    head_flow_pil_list: List[Image.Image],
    left_flow_pil_list: List[Image.Image],
    right_flow_pil_list: List[Image.Image],
    target_size: Tuple[int, int],
) -> List[Image.Image]:
    """T-shape tile per-camera flow images: head full-size on top,
    left/right wrist halves below. Mirrors ``tshape_tile`` for RGB
    so the flow stream is spatially aligned with the tiled RGB stream
    everywhere (no white placeholder regions).

    ``target_size`` is the ``(W, H)`` of the head-camera flow image. The
    output is ``(W, H + H//2)``. Wrist flow images are bilinear-resized
    to ``(W//2, H//2)`` so the bottom row is two side-by-side halves.

    All three input lists must have the same length (typically
    ``num_frames``); the output list has the same length.
    """
    assert len(head_flow_pil_list) == len(left_flow_pil_list) == len(right_flow_pil_list), (
        f"head/left/right flow list lengths must match, got "
        f"{len(head_flow_pil_list)}, {len(left_flow_pil_list)}, "
        f"{len(right_flow_pil_list)}"
    )
    w, h = target_size
    half_h, half_w = h // 2, w // 2
    tiled = []
    for hf, lf, rf in zip(head_flow_pil_list,
                          left_flow_pil_list,
                          right_flow_pil_list):
        head_arr = np.array(hf)
        left_arr = np.array(lf)
        right_arr = np.array(rf)
        if (left_arr.shape[1], left_arr.shape[0]) != (half_w, half_h):
            left_arr = cv2.resize(left_arr, (half_w, half_h),
                                  interpolation=cv2.INTER_LINEAR)
        if (right_arr.shape[1], right_arr.shape[0]) != (half_w, half_h):
            right_arr = cv2.resize(right_arr, (half_w, half_h),
                                   interpolation=cv2.INTER_LINEAR)
        bottom = np.hstack([left_arr, right_arr])
        tiled.append(Image.fromarray(np.vstack([head_arr, bottom])))
    return tiled
