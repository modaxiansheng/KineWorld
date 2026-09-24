#!/usr/bin/env python3
"""Strict action-driven flow conditioning for WorldArena2 Track-1.

This module turns the official RoboTwin 2.0 Aloha-AgileX action vector
``/joint_action/vector`` into the nine optical-flow images consumed by a
KineWorld Stage-1 chunk.  It intentionally has no zero-flow fallback:

* an action chunk uses source indices ``s, s+4, ..., s+32`` (tail-clamped),
* each 14-D vector is interpreted as ``left6, left_gripper, right6,
  right_gripper`` exactly as in RoboTwin,
* the built-in backend drives KineWorld's official ``RobotOnlyScene`` directly,
* flow images are produced by KineWorld's ``training/flow_prefix_utils.py`` and
  ``FlowCodec`` implementation, including the all-white frame-zero sentinel.

Heavy dependencies are imported only after argument parsing.  Consequently
``python action_flow_conditioning.py --help`` also works on a login machine
without SAPIEN, PyTorch, OpenCV, or h5py installed.

The integration entry point used by :mod:`infer_track1` is
``build_action_flow_conditioner(provider_spec, precomputed_root,
provider_kwargs)``.  A conditioner returns exactly one payload kind
(``frames`` here; never ``latents``) plus JSON-serializable provenance.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, runtime_checkable


ACTION_DATASET = "/joint_action/vector"
ACTION_DIM = 14
DEFAULT_CAMERA = "head_camera"
DEFAULT_VARIANT = "aloha-agilex_clean_50"
DEFAULT_KEYFRAME_COUNT = 9
DEFAULT_VISUAL_STRIDE = 4
DEFAULT_FLOW_MAX_MAGNITUDE = 25.0
SCHEMA_VERSION = 1
# Precomputed chunks have their own storage schema.  Version 1 proved the
# selected action rows but did not bind the referenced PNG bytes, so a
# same-size replacement could pass validation.  Version 2 is deliberately
# fail-closed: legacy v1 manifests must be rebuilt rather than silently read.
PRECOMPUTED_MANIFEST_SCHEMA_VERSION = 2
BUILTIN_PROVIDER_NAMES = frozenset(
    {"kineworld", "kineworld_robot_only", "kineworld-robot-only", "robot_only"}
)


class ActionFlowError(RuntimeError):
    """Base exception for a fail-closed action-flow configuration/runtime error."""


class RendererConfigurationError(ActionFlowError):
    """The requested RoboTwin embodiment/camera/renderer is not an exact match."""


class ActionSchemaError(ActionFlowError):
    """The official HDF5 action dataset is absent or has an invalid schema."""


class PrecomputedFlowError(ActionFlowError):
    """A precomputed flow chunk does not prove that it matches its actions."""


@runtime_checkable
class AlohaAgileXRendererBackend(Protocol):
    """Strict renderer contract used by :class:`ActionDrivenFlowConditioner`.

    ``joint_vectors`` is a finite float32 array with shape ``[K, 14]`` in the
    official RoboTwin qpos order.  Implementations must return exactly ``K``
    head-camera robot-only RGB or RGBA frames.  They must not synthesize a
    zero-flow fallback when rendering is unavailable.
    """

    def render_joint_vectors(
        self, joint_vectors: Any, *, camera_name: str
    ) -> Sequence[Any]:
        ...

    def describe(self) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        checked = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if checked < 1 or checked != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return checked


def chunk_source_indices(
    *,
    chunk_start: int,
    frame_count: int,
    keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
    visual_stride: int = DEFAULT_VISUAL_STRIDE,
) -> Tuple[int, ...]:
    """Return ``s, s+stride, ...`` with every tail index clamped to ``T-1``."""

    frame_count = _require_positive_int("frame_count", frame_count)
    keyframe_count = _require_positive_int("keyframe_count", keyframe_count)
    visual_stride = _require_positive_int("visual_stride", visual_stride)
    if isinstance(chunk_start, bool):
        raise ValueError("chunk_start must be an integer")
    try:
        chunk_start = int(chunk_start)
    except (TypeError, ValueError) as error:
        raise ValueError("chunk_start must be an integer") from error
    if not 0 <= chunk_start < frame_count:
        raise ValueError(
            f"chunk_start must satisfy 0 <= start < {frame_count}, got {chunk_start}"
        )
    return tuple(
        min(frame_count - 1, chunk_start + offset * visual_stride)
        for offset in range(keyframe_count)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_mapping(value: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping")
    copied = dict(value)
    try:
        json.dumps(copied, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{what} must be JSON serializable") from error
    return copied


def _load_module_from_file(module_name: str, path: Path) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = sys.modules.get(module_name)
    if cached is not None:
        cached_path = Path(getattr(cached, "__file__", "")).resolve()
        if cached_path != path:
            raise RuntimeError(
                f"module cache collision for {module_name}: {cached_path} != {path}"
            )
        return cached
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _camera_config_from_embodiment(
    embodiment_config: Mapping[str, Any],
    camera_name: str,
    effective_camera_types: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cameras = embodiment_config.get("static_camera_list")
    if not isinstance(cameras, list):
        raise RendererConfigurationError(
            "Aloha-AgileX config.yml must contain static_camera_list"
        )
    matches = [item for item in cameras if isinstance(item, Mapping) and item.get("name") == camera_name]
    if len(matches) != 1:
        raise RendererConfigurationError(
            f"expected exactly one static camera named {camera_name!r}, found {len(matches)}"
        )
    declared = dict(matches[0])
    camera_type = declared.get("type")
    if not isinstance(camera_type, str) or camera_type not in effective_camera_types:
        raise RendererConfigurationError(
            f"camera {camera_name!r} has unsupported/missing type {camera_type!r}"
        )
    position = declared.get("position")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise RendererConfigurationError(
            f"camera {camera_name!r} must declare a three-element position"
        )
    effective = dict(effective_camera_types[camera_type])
    for key in ("w", "h", "fovy"):
        if key not in effective:
            raise RendererConfigurationError(
                f"camera type {camera_type!r} is missing {key!r}"
            )
    return declared, effective


class KineWorldRobotOnlyBackend:
    """Direct adapter for KineWorld's official ``RobotOnlyScene`` API.

    ``robotwin_assets_root`` must be the official RoboTwin ``assets`` directory,
    i.e. it contains ``embodiments/aloha-agilex/config.yml`` and the URDF tree.
    Scene creation is eager so missing SAPIEN/Vulkan support fails before model
    loading rather than becoming an all-white condition later.
    """

    backend_name = "kineworld.robot_only_scene.aloha_agilex.v1"

    def __init__(
        self,
        *,
        robotwin_assets_root: Path,
        kineworld_root: Path,
        variant: str = DEFAULT_VARIANT,
        camera_name: str = DEFAULT_CAMERA,
        render_size: tuple[int, int] = (320, 240),
        camera_config: Optional[Mapping[str, Any]] = None,
        renderer_module_path: Optional[Path] = None,
    ) -> None:
        if camera_name != DEFAULT_CAMERA:
            raise RendererConfigurationError(
                f"Track-1 action flow requires {DEFAULT_CAMERA!r}; got {camera_name!r}"
            )
        if not isinstance(variant, str) or not variant.startswith("aloha-agilex_"):
            raise RendererConfigurationError(
                f"only an Aloha-AgileX variant is supported, got {variant!r}"
            )
        width = _require_positive_int("render width", render_size[0])
        height = _require_positive_int("render height", render_size[1])

        self.robotwin_assets_root = Path(robotwin_assets_root).expanduser().resolve()
        self.kineworld_root = Path(kineworld_root).expanduser().resolve()
        self.variant = variant
        self.camera_name = camera_name
        self.render_size = (width, height)
        if not self.robotwin_assets_root.is_dir():
            raise FileNotFoundError(self.robotwin_assets_root)

        module_path = (
            Path(renderer_module_path).expanduser().resolve()
            if renderer_module_path is not None
            else self.kineworld_root
            / "data_generation"
            / "envs"
            / "utils"
            / "robot_only_renderer.py"
        )
        try:
            renderer_module = _load_module_from_file(
                "_kineworld_track1_robot_only_renderer", module_path
            )
        except ImportError as error:
            raise RendererConfigurationError(
                "KineWorld robot-only renderer dependencies are unavailable; install "
                "h5py, numpy, PyYAML, and the RoboTwin-compatible SAPIEN build"
            ) from error

        required_api = ("RobotOnlyScene", "load_embodiment_config", "DEFAULT_CAMERA_CONFIG")
        missing_api = [name for name in required_api if not hasattr(renderer_module, name)]
        if missing_api:
            raise RendererConfigurationError(
                f"unsupported robot_only_renderer.py API; missing {missing_api}"
            )

        try:
            embodiment_config, robot_dir_raw = renderer_module.load_embodiment_config(
                str(self.robotwin_assets_root), "aloha-agilex"
            )
        except Exception as error:
            raise RendererConfigurationError(
                "could not load official Aloha-AgileX embodiment config under "
                f"{self.robotwin_assets_root / 'embodiments' / 'aloha-agilex'}"
            ) from error
        if not isinstance(embodiment_config, Mapping):
            raise RendererConfigurationError("Aloha-AgileX config.yml is not a mapping")
        embodiment_config = dict(embodiment_config)
        robot_dir = Path(robot_dir_raw).expanduser().resolve()
        expected_robot_dir = (
            self.robotwin_assets_root / "embodiments" / "aloha-agilex"
        ).resolve()
        if robot_dir != expected_robot_dir:
            raise RendererConfigurationError(
                f"renderer resolved unexpected embodiment directory: {robot_dir}"
            )

        arm_names = embodiment_config.get("arm_joints_name")
        if (
            not isinstance(arm_names, list)
            or len(arm_names) != 2
            or any(not isinstance(names, list) or len(names) != 6 for names in arm_names)
        ):
            raise RendererConfigurationError(
                "Aloha-AgileX config must declare two six-joint arm_joints_name lists"
            )
        gripper_names = embodiment_config.get("gripper_name")
        if not isinstance(gripper_names, list) or len(gripper_names) != 2:
            raise RendererConfigurationError(
                "Aloha-AgileX config must declare two gripper_name entries"
            )

        urdf_relative = embodiment_config.get("urdf_path")
        if not isinstance(urdf_relative, str) or not urdf_relative.strip():
            raise RendererConfigurationError("Aloha-AgileX config is missing urdf_path")
        urdf_path = (robot_dir / urdf_relative).resolve()
        if not _path_is_within(urdf_path, robot_dir):
            raise RendererConfigurationError(
                f"Aloha-AgileX urdf_path escapes its embodiment directory: {urdf_path}"
            )
        if not urdf_path.is_file():
            raise FileNotFoundError(urdf_path)

        base_camera_types = deepcopy(dict(renderer_module.DEFAULT_CAMERA_CONFIG))
        if camera_config is not None:
            if not isinstance(camera_config, Mapping):
                raise RendererConfigurationError("camera_config must be a mapping")
            for camera_type, values in camera_config.items():
                if camera_type not in base_camera_types or not isinstance(values, Mapping):
                    raise RendererConfigurationError(
                        f"camera_config contains unsupported camera type {camera_type!r}"
                    )
                base_camera_types[camera_type].update(dict(values))
        effective_camera_types = {
            name: {**dict(values), "w": width, "h": height}
            for name, values in base_camera_types.items()
        }
        declared_camera, effective_camera = _camera_config_from_embodiment(
            embodiment_config, camera_name, effective_camera_types
        )

        scene = None
        try:
            scene = renderer_module.RobotOnlyScene(
                embodiment_config, str(robot_dir), effective_camera_types
            )
            scene.setup()
        except Exception as error:
            if scene is not None:
                try:
                    scene.close()
                except Exception:
                    pass
            raise RendererConfigurationError(
                "failed to create the official KineWorld Aloha-AgileX robot-only "
                "SAPIEN scene; verify the RoboTwin SAPIEN/Vulkan runtime"
            ) from error

        cameras = getattr(scene, "cameras", None)
        if not isinstance(cameras, Mapping) or camera_name not in cameras:
            try:
                scene.close()
            finally:
                raise RendererConfigurationError(
                    f"renderer scene does not expose required camera {camera_name!r}"
                )
        left_indices = getattr(scene, "left_arm_indices", None)
        right_indices = getattr(scene, "right_arm_indices", None)
        if len(left_indices or ()) != 6 or len(right_indices or ()) != 6:
            try:
                scene.close()
            finally:
                raise RendererConfigurationError(
                    "renderer did not bind exactly six joints for each Aloha arm"
                )
        if not callable(getattr(scene, "set_pose_and_render", None)):
            try:
                scene.close()
            finally:
                raise RendererConfigurationError(
                    "RobotOnlyScene lacks set_pose_and_render(...)"
                )

        self._scene = scene
        self._closed = False
        self._description = {
            "backend": self.backend_name,
            "api": "RobotOnlyScene.setup/set_pose_and_render",
            "renderer_module": {
                "path": str(module_path),
                "sha256": _sha256_file(module_path),
            },
            "embodiment": "aloha-agilex",
            "variant": variant,
            "camera": {
                "name": camera_name,
                "declared": declared_camera,
                "effective": effective_camera,
            },
            "embodiment_config": {
                "path": str(robot_dir / "config.yml"),
                "sha256": _sha256_file(robot_dir / "config.yml"),
            },
            "urdf": {
                "path": str(urdf_path),
                "sha256": _sha256_file(urdf_path),
            },
            "render_size": [width, height],
            "action_layout": [
                "left_arm[0:6]",
                "left_gripper[6]",
                "right_arm[7:13]",
                "right_gripper[13]",
            ],
        }
        _json_mapping(self._description, what="renderer provenance")
        atexit.register(self.close)

    def describe(self) -> Mapping[str, Any]:
        return deepcopy(self._description)

    def render_joint_vectors(
        self, joint_vectors: Any, *, camera_name: str
    ) -> Sequence[Any]:
        if self._closed:
            raise ActionFlowError("robot-only renderer is closed")
        if camera_name != self.camera_name:
            raise RendererConfigurationError(
                f"backend is pinned to {self.camera_name!r}, got {camera_name!r}"
            )
        try:
            import numpy as np
        except ImportError as error:
            raise ActionFlowError("numpy is required for robot-only rendering") from error
        vectors = np.asarray(joint_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != ACTION_DIM:
            raise ActionSchemaError(
                f"renderer expected joint vectors [K,{ACTION_DIM}], got {vectors.shape}"
            )
        if vectors.shape[0] < 1 or not np.isfinite(vectors).all():
            raise ActionSchemaError("joint vectors must be non-empty and finite")

        frames = []
        for vector in vectors:
            try:
                rendered = self._scene.set_pose_and_render(
                    vector[0:6],
                    vector[7:13],
                    float(vector[6]),
                    float(vector[13]),
                    camera_name=camera_name,
                )
            except Exception as error:
                raise ActionFlowError(
                    f"Aloha-AgileX rendering failed for camera {camera_name!r}"
                ) from error
            if not isinstance(rendered, Mapping) or camera_name not in rendered:
                raise ActionFlowError(
                    f"renderer did not return requested camera {camera_name!r}"
                )
            frame = np.asarray(rendered[camera_name])
            expected_hw = (self.render_size[1], self.render_size[0])
            if frame.ndim != 3 or tuple(frame.shape[:2]) != expected_hw:
                raise ActionFlowError(
                    f"renderer camera {camera_name!r} produced shape {frame.shape}; "
                    f"expected HxW={expected_hw}"
                )
            frames.append(frame)
        if len(frames) != vectors.shape[0]:
            raise AssertionError("renderer frame count drift")
        return frames

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        scene = getattr(self, "_scene", None)
        self._scene = None
        if scene is not None:
            scene.close()


def _normalize_robot_frames(
    frames: Sequence[Any], *, expected_count: int
) -> list[Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise ActionFlowError("numpy is required to validate renderer frames") from error
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise TypeError("renderer output must be a sequence of RGB/RGBA arrays")
    if len(frames) != expected_count:
        raise ActionFlowError(
            f"renderer returned {len(frames)} frames; expected {expected_count}"
        )

    normalized = []
    common_shape = None
    for index, frame in enumerate(frames):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ActionFlowError(
                f"renderer frame {index} must be HxWx3 RGB or HxWx4 RGBA, got {array.shape}"
            )
        if array.shape[0] < 2 or array.shape[1] < 2:
            raise ActionFlowError(f"renderer frame {index} is too small: {array.shape}")
        if common_shape is None:
            common_shape = tuple(array.shape[:2])
        elif tuple(array.shape[:2]) != common_shape:
            raise ActionFlowError("renderer frames do not share one resolution")

        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
                raise ActionFlowError(
                    f"floating renderer frame {index} must be finite and in [0,1]"
                )
            array = np.rint(array * 255.0).astype(np.uint8)
        elif np.issubdtype(array.dtype, np.integer):
            if array.min() < 0 or array.max() > 255:
                raise ActionFlowError(
                    f"integer renderer frame {index} must be in [0,255]"
                )
            array = array.astype(np.uint8, copy=False)
        else:
            raise ActionFlowError(
                f"unsupported renderer frame dtype at index {index}: {array.dtype}"
            )
        # KineWorld's robot-only flow path consumes RGB.  An official SAPIEN
        # RGBA Color attachment is accepted, with alpha discarded explicitly.
        normalized.append(np.ascontiguousarray(array[..., :3]))
    return normalized


def _read_action_chunk(
    hdf5_path: Path,
    *,
    frame_count: int,
    source_indices: Sequence[int],
) -> Any:
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise ActionSchemaError(
            "h5py and numpy are required to read /joint_action/vector"
        ) from error
    hdf5_path = Path(hdf5_path).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        if ACTION_DATASET not in handle:
            raise ActionSchemaError(f"missing {ACTION_DATASET} in {hdf5_path}")
        dataset = handle[ACTION_DATASET]
        if len(dataset.shape) != 2 or tuple(dataset.shape)[1] != ACTION_DIM:
            raise ActionSchemaError(
                f"{ACTION_DATASET} must have shape [T,{ACTION_DIM}], got "
                f"{tuple(dataset.shape)} in {hdf5_path}"
            )
        if int(dataset.shape[0]) != int(frame_count):
            raise ActionSchemaError(
                f"frame_count={frame_count} disagrees with {ACTION_DATASET}.shape[0]="
                f"{dataset.shape[0]} in {hdf5_path}"
            )
        if int(dataset.shape[0]) < 2:
            raise ActionSchemaError(
                f"{ACTION_DATASET} must contain at least two frames in {hdf5_path}"
            )
        # Read individually because h5py fancy indexing rejects repeated tail
        # indices; repeats are required by Track-1's right-clamp policy.
        rows = [np.asarray(dataset[int(index)], dtype=np.float32) for index in source_indices]
    vectors = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if vectors.shape != (len(source_indices), ACTION_DIM):
        raise ActionSchemaError(
            f"selected action chunk has unexpected shape {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ActionSchemaError(
            f"{ACTION_DATASET} contains NaN/Inf at source indices {list(source_indices)}"
        )
    grippers = vectors[:, (6, 13)]
    if (grippers < 0.0).any() or (grippers > 1.0).any():
        raise ActionSchemaError(
            f"{ACTION_DATASET} grippers at columns 6/13 must be normalized to [0,1]"
        )
    return vectors


def _action_chunk_sha256(vectors: Any) -> str:
    try:
        import numpy as np
    except ImportError as error:
        raise ActionFlowError("numpy is required to hash action chunks") from error
    canonical = np.asarray(vectors, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _first_frame_size(first_frame: Any) -> list[int]:
    size = getattr(first_frame, "size", None)
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in size)
    ):
        raise TypeError("first_frame must be a PIL-like image with positive .size")
    return [int(size[0]), int(size[1])]


class ActionDrivenFlowConditioner:
    """Render actions and encode their robot-only motion as nine flow frames."""

    def __init__(
        self,
        *,
        backend: AlohaAgileXRendererBackend,
        kineworld_root: Path,
        camera_name: str = DEFAULT_CAMERA,
        flow_method: str = "raft",
        flow_device: str = "cuda",
        flow_max_magnitude: float = DEFAULT_FLOW_MAX_MAGNITUDE,
        flow_processor: Any = None,
        codec: Any = None,
        raft_extractor: Any = None,
    ) -> None:
        if not isinstance(backend, AlohaAgileXRendererBackend):
            raise TypeError(
                "backend must implement render_joint_vectors(), describe(), and close()"
            )
        if camera_name != DEFAULT_CAMERA:
            raise RendererConfigurationError(
                f"Track-1 requires camera {DEFAULT_CAMERA!r}, got {camera_name!r}"
            )
        if flow_method not in {"raft", "farneback"}:
            raise ValueError("flow_method must be 'raft' or 'farneback'")
        try:
            flow_max_magnitude = float(flow_max_magnitude)
        except (TypeError, ValueError) as error:
            raise ValueError("flow_max_magnitude must be a positive float or -1") from error
        if flow_max_magnitude != -1 and flow_max_magnitude <= 0:
            raise ValueError("flow_max_magnitude must be positive or -1")

        self.backend = backend
        self.kineworld_root = Path(kineworld_root).expanduser().resolve()
        self.camera_name = camera_name
        self.flow_method = flow_method
        self.flow_device = str(flow_device)
        self.flow_max_magnitude = flow_max_magnitude

        flow_utils_path = self.kineworld_root / "training" / "flow_prefix_utils.py"
        codec_path = self.kineworld_root / "training" / "reversible_flow_codec.py"
        self._flow_utils_path = flow_utils_path.resolve()
        self._codec_path = codec_path.resolve()

        if flow_processor is None:
            try:
                flow_utils = _load_module_from_file(
                    "_kineworld_track1_flow_prefix_utils", flow_utils_path
                )
            except ImportError as error:
                raise ActionFlowError(
                    "KineWorld flow-prefix dependencies are unavailable; install numpy, "
                    "Pillow, and OpenCV"
                ) from error
            flow_processor = getattr(flow_utils, "process_camera_flow", None)
            if not callable(flow_processor):
                raise ActionFlowError(
                    f"{flow_utils_path} lacks process_camera_flow(...)"
                )
        if codec is None:
            codec_module = _load_module_from_file(
                "_kineworld_track1_reversible_flow_codec", codec_path
            )
            codec_class = getattr(codec_module, "FlowCodec", None)
            if codec_class is None:
                raise ActionFlowError(f"{codec_path} lacks FlowCodec")
            codec = codec_class(use_16bit=False)
        if flow_method == "raft" and raft_extractor is None:
            raft_path = self.kineworld_root / "training" / "raft_flow_extractor.py"
            raft_module = _load_module_from_file(
                "_kineworld_track1_raft_flow_extractor", raft_path
            )
            extractor_class = getattr(raft_module, "RAFTFlowExtractor", None)
            if extractor_class is None:
                raise ActionFlowError(f"{raft_path} lacks RAFTFlowExtractor")
            try:
                raft_extractor = extractor_class(device=self.flow_device)
            except Exception as error:
                raise ActionFlowError(
                    "could not initialize KineWorld's torchvision RAFT-large extractor; "
                    "the action-flow path will not fall back to white flow"
                ) from error

        self._process_camera_flow = flow_processor
        self._codec = codec
        self._raft_extractor = raft_extractor
        self._backend_description = _json_mapping(
            backend.describe(), what="renderer backend description"
        )
        camera = self._backend_description.get("camera")
        if not isinstance(camera, Mapping) or camera.get("name") != camera_name:
            raise RendererConfigurationError(
                "renderer provenance does not prove a matching head_camera"
            )
        if self._backend_description.get("embodiment") != "aloha-agilex":
            raise RendererConfigurationError(
                "renderer provenance does not prove the aloha-agilex embodiment"
            )
        urdf = self._backend_description.get("urdf")
        if not isinstance(urdf, Mapping) or not urdf.get("path") or not urdf.get("sha256"):
            raise RendererConfigurationError(
                "renderer provenance must include the resolved URDF path and SHA-256"
            )

    def describe(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "conditioner": "action_driven_robot_only_flow.v1",
            "payload": "frames",
            "action_dataset": ACTION_DATASET,
            "action_dim": ACTION_DIM,
            "camera": self.camera_name,
            "flow": {
                "method": self.flow_method,
                "device": self.flow_device,
                "codec": "FlowCodec(use_16bit=False)",
                "max_magnitude": self.flow_max_magnitude,
                "first_frame": "white_zero_flow_sentinel",
                "flow_prefix_utils": {
                    "path": str(self._flow_utils_path),
                    "sha256": _sha256_file(self._flow_utils_path),
                },
                "flow_codec": {
                    "path": str(self._codec_path),
                    "sha256": _sha256_file(self._codec_path),
                },
            },
            "renderer": deepcopy(self._backend_description),
        }

    def get_chunk_flow(
        self,
        *,
        episode_id: int,
        hdf5_path: Path,
        first_frame: Any,
        frame_count: int,
        chunk_start: int,
        keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
        visual_stride: int = DEFAULT_VISUAL_STRIDE,
        target_size: tuple[int, int],
    ) -> Mapping[str, Any]:
        episode_id = _require_positive_int("episode_id", episode_id)
        target_width = _require_positive_int("target width", target_size[0])
        target_height = _require_positive_int("target height", target_size[1])
        first_size = _first_frame_size(first_frame)
        indices = chunk_source_indices(
            chunk_start=chunk_start,
            frame_count=frame_count,
            keyframe_count=keyframe_count,
            visual_stride=visual_stride,
        )
        hdf5_path = Path(hdf5_path).expanduser().resolve()
        vectors = _read_action_chunk(
            hdf5_path,
            frame_count=frame_count,
            source_indices=indices,
        )
        action_digest = _action_chunk_sha256(vectors)

        rendered = self.backend.render_joint_vectors(
            vectors, camera_name=self.camera_name
        )
        robot_rgb = _normalize_robot_frames(
            rendered, expected_count=keyframe_count
        )
        try:
            flow_frames, max_magnitudes = self._process_camera_flow(
                robot_rgb,
                target_size=(target_width, target_height),
                codec=self._codec,
                flow_method=self.flow_method,
                raft_extractor=self._raft_extractor,
                max_magnitude=self.flow_max_magnitude,
            )
        except Exception as error:
            raise ActionFlowError(
                "KineWorld robot-only flow extraction failed; refusing zero-flow fallback"
            ) from error
        if isinstance(flow_frames, (str, bytes)) or not isinstance(flow_frames, Sequence):
            raise ActionFlowError("process_camera_flow did not return a frame sequence")
        if len(flow_frames) != keyframe_count:
            raise ActionFlowError(
                f"flow extractor returned {len(flow_frames)} frames; expected {keyframe_count}"
            )
        if len(max_magnitudes) != keyframe_count:
            raise ActionFlowError("flow magnitude metadata length mismatch")

        checked_frames = []
        for index, frame in enumerate(flow_frames):
            convert = getattr(frame, "convert", None)
            if not callable(convert):
                raise TypeError(f"flow frame {index} is not a PIL-like image")
            rgb = convert("RGB")
            if tuple(rgb.size) != (target_width, target_height):
                raise ActionFlowError(
                    f"flow frame {index} has size {rgb.size}; expected "
                    f"{(target_width, target_height)}"
                )
            checked_frames.append(rgb)
        # Verify the shared helper retained KineWorld's mandatory frame-zero
        # sentinel.  This is a semantic check, not a fallback construction.
        if checked_frames[0].getextrema() != ((255, 255), (255, 255), (255, 255)):
            raise ActionFlowError("flow frame zero is not KineWorld's white sentinel")
        if float(max_magnitudes[0]) != 0.0:
            raise ActionFlowError("flow frame-zero magnitude metadata must be 0.0")

        stat = hdf5_path.stat()
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "mode": "action_flow",
            "control_type": "official_14d_joint_action",
            "episode_id": episode_id,
            "source": {
                "hdf5_path": str(hdf5_path),
                "dataset": ACTION_DATASET,
                "shape": [int(frame_count), ACTION_DIM],
                "file_size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "selected_action_sha256_float32_le": action_digest,
            },
            "chunk": {
                "start": int(chunk_start),
                "keyframe_count": int(keyframe_count),
                "visual_stride": int(visual_stride),
                "source_indices": list(indices),
                "tail_clamped": len(set(indices)) != len(indices),
            },
            "first_frame_size": first_size,
            "target_size": [target_width, target_height],
            "flow": {
                "method": self.flow_method,
                "codec": "FlowCodec(use_16bit=False)",
                "configured_max_magnitude": self.flow_max_magnitude,
                "actual_max_magnitudes": [float(value) for value in max_magnitudes],
                "frame_zero": "white_zero_flow_sentinel",
                "robot_frame_channels": [int(frame.shape[2]) for frame in robot_rgb],
            },
            "renderer": deepcopy(self._backend_description),
        }
        _json_mapping(provenance, what="chunk flow provenance")
        # Exactly one payload kind: frames.  Do not add a ``latents`` key with
        # None, because the Track-1 adapter deliberately treats that as schema
        # ambiguity.
        return {"frames": tuple(checked_frames), "provenance": provenance}

    def release_episode(self, *, episode_id: int) -> None:
        # The SAPIEN scene and RAFT weights are deliberately reused across
        # episodes.  This hook exists for the infer_track1 provider contract.
        _require_positive_int("episode_id", episode_id)

    def close(self) -> None:
        self.backend.close()


def _validate_precomputed_renderer(provenance: Mapping[str, Any]) -> None:
    renderer = provenance.get("renderer")
    if not isinstance(renderer, Mapping):
        raise PrecomputedFlowError("manifest provenance is missing renderer metadata")
    if renderer.get("embodiment") != "aloha-agilex":
        raise PrecomputedFlowError("precomputed renderer embodiment is not aloha-agilex")
    camera = renderer.get("camera")
    if not isinstance(camera, Mapping) or camera.get("name") != DEFAULT_CAMERA:
        raise PrecomputedFlowError("precomputed renderer camera is not head_camera")
    urdf = renderer.get("urdf")
    if not isinstance(urdf, Mapping) or not urdf.get("path") or not urdf.get("sha256"):
        raise PrecomputedFlowError("precomputed renderer provenance lacks URDF identity")


class PrecomputedActionFlowConditioner:
    """Read strictly manifested action-derived flow chunks from disk.

    Layout::

        ROOT/episode1/chunk_000000/manifest.json
        ROOT/episode1/chunk_000000/flow_00.png ... flow_08.png

    The manifest is generated by this module's CLI.  Schema v2 includes a
    SHA-256 of the exact selected float32 action rows and of every encoded PNG,
    preventing accidental reuse for a different trajectory or silent frame
    replacement. Schema v1 is rejected and must be rebuilt.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def describe(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_schema_version": PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
            "conditioner": "precomputed_action_flow.v1",
            "payload": "frames",
            "root": str(self.root),
            "layout": "episode<ID>/chunk_<START:06d>/manifest.json",
            "validation": "selected_float32_action_sha256+png_file_sha256",
        }

    def get_chunk_flow(
        self,
        *,
        episode_id: int,
        hdf5_path: Path,
        first_frame: Any,
        frame_count: int,
        chunk_start: int,
        keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
        visual_stride: int = DEFAULT_VISUAL_STRIDE,
        target_size: tuple[int, int],
    ) -> Mapping[str, Any]:
        episode_id = _require_positive_int("episode_id", episode_id)
        target_width = _require_positive_int("target width", target_size[0])
        target_height = _require_positive_int("target height", target_size[1])
        _first_frame_size(first_frame)
        indices = chunk_source_indices(
            chunk_start=chunk_start,
            frame_count=frame_count,
            keyframe_count=keyframe_count,
            visual_stride=visual_stride,
        )
        vectors = _read_action_chunk(
            Path(hdf5_path), frame_count=frame_count, source_indices=indices
        )
        expected_digest = _action_chunk_sha256(vectors)
        chunk_dir = self.root / f"episode{episode_id}" / f"chunk_{int(chunk_start):06d}"
        manifest_path = chunk_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PrecomputedFlowError(f"invalid manifest {manifest_path}") from error
        manifest = _json_mapping(manifest, what="precomputed manifest")
        manifest_schema_version = manifest.get("schema_version")
        if manifest_schema_version != PRECOMPUTED_MANIFEST_SCHEMA_VERSION:
            if manifest_schema_version == 1:
                raise PrecomputedFlowError(
                    "legacy precomputed manifest schema v1 does not bind flow PNG "
                    f"bytes; rebuild this chunk with schema v"
                    f"{PRECOMPUTED_MANIFEST_SCHEMA_VERSION}: {manifest_path}"
                )
            raise PrecomputedFlowError(
                f"unsupported precomputed manifest schema "
                f"{manifest_schema_version!r}; expected "
                f"{PRECOMPUTED_MANIFEST_SCHEMA_VERSION}: {manifest_path}"
            )
        expected_fields = {
            "episode_id": episode_id,
            "chunk_start": int(chunk_start),
            "frame_count": int(frame_count),
            "keyframe_count": int(keyframe_count),
            "visual_stride": int(visual_stride),
            "source_indices": list(indices),
            "target_size": [target_width, target_height],
            "action_sha256_float32_le": expected_digest,
        }
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in expected_fields.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise PrecomputedFlowError(
                f"precomputed chunk manifest mismatch at {manifest_path}: {mismatches}"
            )
        frame_names = manifest.get("frames")
        if (
            isinstance(frame_names, (str, bytes))
            or not isinstance(frame_names, list)
            or len(frame_names) != keyframe_count
            or len(set(frame_names)) != keyframe_count
        ):
            raise PrecomputedFlowError(
                "manifest frames must list every flow PNG exactly once"
            )
        frame_sha256 = manifest.get("frame_sha256")
        if not isinstance(frame_sha256, Mapping) or set(frame_sha256) != set(frame_names):
            raise PrecomputedFlowError(
                "schema-v2 manifest frame_sha256 must bind every listed flow PNG"
            )
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            raise PrecomputedFlowError("manifest lacks chunk provenance")
        _validate_precomputed_renderer(provenance)

        try:
            from PIL import Image
        except ImportError as error:
            raise PrecomputedFlowError("Pillow is required for precomputed flow") from error
        frames = []
        for index, name in enumerate(frame_names):
            if not isinstance(name, str) or Path(name).name != name:
                raise PrecomputedFlowError(
                    f"manifest frame {index} must be a local basename"
                )
            path = chunk_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_frame_sha256 = frame_sha256.get(name)
            if (
                not isinstance(expected_frame_sha256, str)
                or len(expected_frame_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_frame_sha256
                )
            ):
                raise PrecomputedFlowError(
                    f"manifest frame SHA-256 is invalid for {name!r}"
                )
            actual_frame_sha256 = _sha256_file(path)
            if actual_frame_sha256 != expected_frame_sha256:
                raise PrecomputedFlowError(
                    f"precomputed frame SHA-256 mismatch for {path}: "
                    f"{actual_frame_sha256} != {expected_frame_sha256}"
                )
            with Image.open(path) as image:
                frame = image.convert("RGB").copy()
            if frame.size != (target_width, target_height):
                raise PrecomputedFlowError(
                    f"precomputed frame {path} has size {frame.size}, expected "
                    f"{(target_width, target_height)}"
                )
            frames.append(frame)
        if frames[0].getextrema() != ((255, 255), (255, 255), (255, 255)):
            raise PrecomputedFlowError("precomputed frame zero is not white zero flow")
        loaded_provenance = deepcopy(dict(provenance))
        loaded_provenance["precomputed"] = {
            "root": str(self.root),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        }
        _json_mapping(loaded_provenance, what="precomputed provenance")
        return {"frames": tuple(frames), "provenance": loaded_provenance}

    def release_episode(self, *, episode_id: int) -> None:
        _require_positive_int("episode_id", episode_id)


def _pop_alias(
    values: dict[str, Any], canonical: str, aliases: Sequence[str], *, required: bool
) -> Any:
    present = [name for name in (canonical, *aliases) if name in values]
    if len(present) > 1:
        raise ValueError(f"provide only one of {present}")
    if not present:
        if required:
            raise ValueError(f"missing provider kwarg {canonical!r}")
        return None
    return values.pop(present[0])


def _build_builtin_conditioner(provider_kwargs: Mapping[str, Any]) -> ActionDrivenFlowConditioner:
    kwargs = dict(provider_kwargs)
    assets_root = _pop_alias(
        kwargs,
        "robotwin_assets_root",
        ("assets_root", "dataset_root"),
        required=True,
    )
    default_kineworld_root = Path(__file__).resolve().parents[1]
    kineworld_root = Path(kwargs.pop("kineworld_root", default_kineworld_root))
    variant = kwargs.pop("variant", DEFAULT_VARIANT)
    camera_name = kwargs.pop("camera_name", DEFAULT_CAMERA)
    render_size_raw = kwargs.pop("render_size", (320, 240))
    if (
        isinstance(render_size_raw, (str, bytes))
        or not isinstance(render_size_raw, Sequence)
        or len(render_size_raw) != 2
    ):
        raise ValueError("render_size must be [width, height]")
    render_size = (int(render_size_raw[0]), int(render_size_raw[1]))
    camera_config = kwargs.pop("camera_config", None)
    renderer_module_path = kwargs.pop("renderer_module_path", None)
    flow_method = kwargs.pop("flow_method", "raft")
    flow_device = kwargs.pop("flow_device", "cuda")
    flow_max_magnitude = kwargs.pop(
        "flow_max_magnitude", DEFAULT_FLOW_MAX_MAGNITUDE
    )
    if kwargs:
        raise ValueError(f"unknown kineworld_robot_only provider kwargs: {sorted(kwargs)}")

    backend = KineWorldRobotOnlyBackend(
        robotwin_assets_root=Path(assets_root),
        kineworld_root=kineworld_root,
        variant=variant,
        camera_name=camera_name,
        render_size=render_size,
        camera_config=camera_config,
        renderer_module_path=(
            Path(renderer_module_path) if renderer_module_path is not None else None
        ),
    )
    try:
        return ActionDrivenFlowConditioner(
            backend=backend,
            kineworld_root=kineworld_root,
            camera_name=camera_name,
            flow_method=flow_method,
            flow_device=flow_device,
            flow_max_magnitude=flow_max_magnitude,
        )
    except Exception:
        backend.close()
        raise


def _build_imported_provider(
    provider_spec: str,
    precomputed_root: Optional[Path],
    provider_kwargs: Mapping[str, Any],
) -> Any:
    if provider_spec.count(":") != 1:
        raise ValueError(
            "custom provider spec must be 'python.module:factory_or_object'"
        )
    module_name, attribute_name = provider_spec.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError(
            "custom provider spec must be 'python.module:factory_or_object'"
        )
    module = importlib.import_module(module_name)
    provider = getattr(module, attribute_name)
    kwargs = dict(provider_kwargs)
    if precomputed_root is not None:
        if "precomputed_root" in kwargs:
            raise ValueError("precomputed_root was supplied twice")
        kwargs["precomputed_root"] = precomputed_root
    if inspect.isclass(provider) or inspect.isfunction(provider) or inspect.ismethod(provider):
        provider = provider(**kwargs)
    elif not callable(getattr(provider, "get_chunk_flow", None)):
        if not callable(provider):
            raise TypeError(f"custom provider {provider_spec!r} is not constructible")
        provider = provider(**kwargs)
    elif kwargs:
        raise ValueError("an already-created provider object does not accept provider_kwargs")
    if not callable(getattr(provider, "get_chunk_flow", None)):
        raise TypeError(f"custom provider {provider_spec!r} lacks get_chunk_flow(...)")
    describe = getattr(provider, "describe", None)
    if callable(describe):
        _json_mapping(describe(), what="custom provider description")
    return provider


def build_action_flow_conditioner(
    provider_spec: Optional[str],
    precomputed_root: Optional[Path],
    provider_kwargs: dict[str, Any],
) -> Any:
    """Build the provider contract consumed by ``infer_track1.py``.

    Supported built-ins:

    * ``kineworld_robot_only``: render official 14-D actions with the checked
      KineWorld/RoboTwin Aloha-AgileX SAPIEN backend.  Requires provider kwarg
      ``robotwin_assets_root``.
    * no provider (or ``precomputed``) plus ``precomputed_root``: load the
      strictly manifested output produced by this module's CLI.
    * ``python.module:factory``: an explicitly requested external provider.

    Supplying neither a renderer nor a precomputed root is always an error.
    """

    if not isinstance(provider_kwargs, dict):
        raise TypeError("provider_kwargs must be a dict")
    normalized = provider_spec.strip() if isinstance(provider_spec, str) else None
    if normalized == "":
        normalized = None
    root = Path(precomputed_root).expanduser().resolve() if precomputed_root is not None else None

    if normalized in (None, "precomputed"):
        if root is None:
            raise ValueError(
                "action flow requires provider_spec='kineworld_robot_only' or "
                "a precomputed_root; zero flow is not a fallback"
            )
        if provider_kwargs:
            raise ValueError("precomputed provider does not accept provider_kwargs")
        return PrecomputedActionFlowConditioner(root)
    if normalized in BUILTIN_PROVIDER_NAMES:
        if root is not None:
            raise ValueError(
                "choose either the live KineWorld renderer or precomputed_root, not both"
            )
        return _build_builtin_conditioner(provider_kwargs)
    return _build_imported_provider(normalized, root, provider_kwargs)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_precomputed_chunk(
    *,
    output_root: Path,
    episode_id: int,
    chunk_start: int,
    frame_count: int,
    keyframe_count: int,
    visual_stride: int,
    target_size: tuple[int, int],
    result: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    frames = result.get("frames")
    if frames is None or result.get("latents") is not None:
        raise ValueError("CLI provider must return frames and no latents")
    provenance = _json_mapping(result.get("provenance"), what="chunk provenance")
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("chunk provenance lacks source actions")
    action_digest = source.get("selected_action_sha256_float32_le")
    if not isinstance(action_digest, str):
        raise ValueError("chunk provenance lacks selected action SHA-256")
    source_indices = chunk_source_indices(
        chunk_start=chunk_start,
        frame_count=frame_count,
        keyframe_count=keyframe_count,
        visual_stride=visual_stride,
    )
    chunk_dir = output_root / f"episode{episode_id}" / f"chunk_{chunk_start:06d}"
    manifest_path = chunk_dir / "manifest.json"
    if chunk_dir.exists() and not overwrite:
        raise FileExistsError(
            f"precomputed chunk already exists: {chunk_dir}; pass --overwrite"
        )
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frame_names = []
    frame_sha256 = {}
    for index, frame in enumerate(frames):
        name = f"flow_{index:02d}.png"
        path = chunk_dir / name
        frame.convert("RGB").save(path, format="PNG")
        frame_names.append(name)
        frame_sha256[name] = _sha256_file(path)
    manifest = {
        "schema_version": PRECOMPUTED_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "chunk_start": chunk_start,
        "frame_count": frame_count,
        "keyframe_count": keyframe_count,
        "visual_stride": visual_stride,
        "source_indices": list(source_indices),
        "target_size": [target_size[0], target_size[1]],
        "action_sha256_float32_le": action_digest,
        "frames": frame_names,
        "frame_sha256": frame_sha256,
        "provenance": provenance,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render official RoboTwin Aloha-AgileX /joint_action/vector chunks "
            "and encode strict KineWorld robot-only flow conditions"
        )
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="official episode HDF5")
    parser.add_argument("--first-frame", type=Path, required=True, help="official episode PNG")
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--chunk-start",
        type=int,
        action="append",
        help="source start; repeat for multiple chunks (default: 0)",
    )
    parser.add_argument(
        "--robotwin-assets-root",
        type=Path,
        required=True,
        help="RoboTwin assets/ containing embodiments/aloha-agilex/config.yml",
    )
    parser.add_argument(
        "--kineworld-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--camera", choices=(DEFAULT_CAMERA,), default=DEFAULT_CAMERA)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--target-height", type=int, default=240)
    parser.add_argument("--flow-method", choices=("raft", "farneback"), default="raft")
    parser.add_argument("--flow-device", default="cuda")
    parser.add_argument(
        "--flow-max-magnitude", type=float, default=DEFAULT_FLOW_MAX_MAGNITUDE
    )
    parser.add_argument(
        "--renderer-module-path",
        type=Path,
        help="explicit official KineWorld robot_only_renderer.py override",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        import h5py
        from PIL import Image
    except ImportError as error:
        raise ActionFlowError(
            "CLI execution requires h5py and Pillow (but --help does not)"
        ) from error
    hdf5_path = args.hdf5.expanduser().resolve()
    with h5py.File(hdf5_path, "r") as handle:
        if ACTION_DATASET not in handle:
            raise ActionSchemaError(f"missing {ACTION_DATASET} in {hdf5_path}")
        shape = tuple(handle[ACTION_DATASET].shape)
    if len(shape) != 2 or shape[1] != ACTION_DIM or shape[0] < 2:
        raise ActionSchemaError(
            f"{ACTION_DATASET} must be [T,{ACTION_DIM}], got {shape}"
        )
    frame_count = int(shape[0])
    first_path = args.first_frame.expanduser().resolve()
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    with Image.open(first_path) as image:
        first_frame = image.convert("RGB").copy()

    provider_kwargs = {
        "robotwin_assets_root": str(args.robotwin_assets_root),
        "kineworld_root": str(args.kineworld_root),
        "variant": args.variant,
        "camera_name": args.camera,
        "render_size": [args.render_width, args.render_height],
        "flow_method": args.flow_method,
        "flow_device": args.flow_device,
        "flow_max_magnitude": args.flow_max_magnitude,
    }
    if args.renderer_module_path is not None:
        provider_kwargs["renderer_module_path"] = str(args.renderer_module_path)
    conditioner = build_action_flow_conditioner(
        provider_spec="kineworld_robot_only",
        precomputed_root=None,
        provider_kwargs=provider_kwargs,
    )
    output_root = args.output_root.expanduser().resolve()
    starts = args.chunk_start or [0]
    try:
        print(json.dumps(conditioner.describe(), ensure_ascii=False, sort_keys=True))
        for start in starts:
            result = conditioner.get_chunk_flow(
                episode_id=args.episode_id,
                hdf5_path=hdf5_path,
                first_frame=first_frame,
                frame_count=frame_count,
                chunk_start=start,
                keyframe_count=DEFAULT_KEYFRAME_COUNT,
                visual_stride=DEFAULT_VISUAL_STRIDE,
                target_size=(args.target_width, args.target_height),
            )
            manifest = _write_precomputed_chunk(
                output_root=output_root,
                episode_id=args.episode_id,
                chunk_start=start,
                frame_count=frame_count,
                keyframe_count=DEFAULT_KEYFRAME_COUNT,
                visual_stride=DEFAULT_VISUAL_STRIDE,
                target_size=(args.target_width, args.target_height),
                result=result,
                overwrite=args.overwrite,
            )
            print(str(manifest), flush=True)
            conditioner.release_episode(episode_id=args.episode_id)
    finally:
        close = getattr(conditioner, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
