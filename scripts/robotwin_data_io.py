"""Small CPU-only readers for the supported RoboTwin Aloha-AgileX schema.

Image convention reference:
https://github.com/RoboTwin-Platform/RoboTwin/blob/main/data/decode_image_bit.py
JPEG COM ``XPL-RGB1`` marks standard RGB; unmarked RoboTwin JPEGs are legacy.
No model weights, rendered scenes or invented robot actions are used here.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any


def decode_rgb_frame(frame: Any, *, image_encoding: str = "auto") -> tuple[Any, dict]:
    """Decode one frame to a Pillow RGB image and report the applied convention.

    ``auto`` is for actual RoboTwin buffers, not arbitrary unmarked JPEGs.
    Use ``standard-rgb`` explicitly for standard externally encoded images.
    Already decoded uint8 HWC RGB arrays never undergo a channel swap.
    """
    import numpy as np
    from PIL import Image

    if image_encoding not in ("auto", "robotwin-legacy", "standard-rgb"):
        raise ValueError(f"unknown image encoding: {image_encoding}")
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        if frame.dtype != np.uint8 or frame.shape[2] != 3:
            raise ValueError("decoded frame must be uint8 [H,W,3] RGB")
        return Image.fromarray(np.ascontiguousarray(frame)), {
            "requested_encoding": image_encoding,
            "decoded_encoding": "uint8_rgb_array",
            "channel_swap": False,
        }
    if isinstance(frame, np.ndarray):
        if frame.dtype == np.uint8 and frame.ndim == 1:
            encoded = frame.tobytes()
        elif frame.dtype.kind == "S":
            encoded = frame.tobytes()
        elif frame.ndim == 0 and frame.dtype.kind == "O":
            encoded = bytes(frame.item())
        else:
            raise ValueError("encoded frame must be bytes or a one-dimensional uint8 buffer")
    elif isinstance(frame, (bytes, bytearray, memoryview, np.void)):
        encoded = bytes(frame)
    else:
        raise ValueError(f"unsupported encoded frame type: {type(frame).__name__}")
    with Image.open(BytesIO(encoded)) as source:
        marked = any(
            name == "COM" and value == b"XPL-RGB1"
            for name, value in getattr(source, "applist", ())
        )
        if image_encoding == "auto" and source.format != "JPEG":
            raise ValueError(
                "auto color detection accepts RoboTwin JPEG only; for a known "
                "standard RGB PNG use --image-encoding standard-rgb"
            )
        if marked and image_encoding == "robotwin-legacy":
            raise ValueError("standard RGB JPEG marker conflicts with robotwin-legacy")
        swap = image_encoding == "robotwin-legacy" or (
            image_encoding == "auto" and not marked
        )
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        if swap:
            rgb = rgb[..., ::-1]
        image = Image.fromarray(np.ascontiguousarray(rgb))
    return image, {
        "requested_encoding": image_encoding,
        "decoded_encoding": "robotwin_legacy" if swap else "standard_rgb",
        "standard_rgb_marker": "XPL-RGB1" if marked else None,
        "channel_swap": swap,
    }


def read_joint_vectors(handle: Any) -> tuple[Any, str]:
    """Return real [T,14] qpos; never infer EEF semantics from vector length.

    Supported order is left arm 6, left normalized gripper, right arm 6,
    right normalized gripper, matching KineWorld's RoboTwin loader.
    If both representations exist, they must agree after float32 conversion.
    """
    import numpy as np

    fields = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    paths = [f"joint_action/{name}" for name in fields]
    available = [path in handle for path in paths]
    split = None
    if any(available):
        if not all(available):
            raise ValueError("incomplete joint_action split fields; all four are required")
        arrays = []
        for path, width in zip(paths, (6, 1, 6, 1)):
            dataset = handle[path]
            if dataset.dtype.kind not in "fiu":
                raise ValueError(f"/{path} must be numeric")
            array = np.asarray(dataset[:])
            if width == 1 and array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(f"/{path} must have shape [T,{width}], got {array.shape}")
            arrays.append(array)
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError("joint_action split fields have inconsistent frame counts")
        split = np.concatenate(arrays, axis=1)
    if "joint_action/vector" in handle:
        dataset = handle["joint_action/vector"]
        if dataset.dtype.kind not in "fiu":
            raise ValueError("/joint_action/vector must be numeric")
        vectors = np.asarray(dataset[:])
        source = "/joint_action/vector"
        if split is not None and not np.array_equal(
            vectors.astype(np.float32), split.astype(np.float32)
        ):
            raise ValueError("/joint_action/vector disagrees with the split joint fields")
    elif split is not None:
        vectors = split
        source = "assembled_from_joint_action_split_fields"
    else:
        raise ValueError("missing RoboTwin joint_action vector and split joint fields")
    if vectors.ndim != 2 or vectors.shape[1] != 14 or vectors.shape[0] < 2:
        raise ValueError(f"joint actions must be [T,14] with T>=2, got {vectors.shape}")
    if not np.isfinite(vectors).all() or not np.isfinite(vectors.astype(np.float32)).all():
        raise ValueError("joint actions contain NaN/Inf or overflow float32")
    if ((vectors[:, (6, 13)] < 0) | (vectors[:, (6, 13)] > 1)).any():
        raise ValueError("joint grippers at columns 6/13 must already be normalized to [0,1]")
    return vectors, source
