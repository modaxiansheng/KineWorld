"""Pure-logic and fake-renderer tests for action_flow_conditioning.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("action_flow_conditioning.py")
SPEC = importlib.util.spec_from_file_location("action_flow_conditioning_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


try:
    import h5py
    import numpy as np
    from PIL import Image

    HAS_ARRAY_STACK = True
except ImportError:
    HAS_ARRAY_STACK = False


class ChunkIndexTests(unittest.TestCase):
    def test_track1_stride_four_indices(self) -> None:
        self.assertEqual(
            MODULE.chunk_source_indices(
                chunk_start=7,
                frame_count=100,
                keyframe_count=9,
                visual_stride=4,
            ),
            (7, 11, 15, 19, 23, 27, 31, 35, 39),
        )

    def test_tail_is_right_clamped(self) -> None:
        self.assertEqual(
            MODULE.chunk_source_indices(
                chunk_start=8,
                frame_count=12,
                keyframe_count=9,
                visual_stride=4,
            ),
            (8, 11, 11, 11, 11, 11, 11, 11, 11),
        )

    def test_invalid_start_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.chunk_source_indices(chunk_start=12, frame_count=12)

    def test_builder_has_no_implicit_zero_flow(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero flow is not a fallback"):
            MODULE.build_action_flow_conditioner(None, None, {})


@unittest.skipUnless(HAS_ARRAY_STACK, "requires numpy, h5py, and Pillow")
class FakeRendererIntegrationTests(unittest.TestCase):
    class FakeRenderer:
        def __init__(self) -> None:
            self.received = None
            self.closed = False

        def describe(self):
            return {
                "backend": "test.fake_renderer",
                "embodiment": "aloha-agilex",
                "camera": {
                    "name": "head_camera",
                    "declared": {"name": "head_camera", "type": "D435"},
                    "effective": {"w": 32, "h": 24, "fovy": 37},
                },
                "urdf": {"path": "/test/aloha.urdf", "sha256": "0" * 64},
            }

        def render_joint_vectors(self, joint_vectors, *, camera_name):
            assert camera_name == "head_camera"
            self.received = np.asarray(joint_vectors).copy()
            frames = []
            for index in range(len(joint_vectors)):
                frame = np.full((24, 32, 4), 255, dtype=np.uint8)
                frame[8:16, 2 + index : 6 + index, :3] = 0
                frames.append(frame)
            return frames

        def close(self):
            self.closed = True

    @staticmethod
    def fake_flow_processor(
        frames,
        *,
        target_size,
        codec,
        flow_method,
        raft_extractor,
        max_magnitude,
    ):
        assert len(frames) == 9
        assert all(frame.shape == (24, 32, 3) for frame in frames)
        assert flow_method == "farneback"
        width, height = target_size
        output = [Image.new("RGB", (width, height), "white")]
        output.extend(
            Image.new("RGB", (width, height), (index, 0, 0))
            for index in range(1, 9)
        )
        return output, [0.0] + [float(max_magnitude)] * 8

    def test_hdf5_actions_drive_fake_renderer_and_return_only_frames(self) -> None:
        backend = self.FakeRenderer()
        kineworld_root = MODULE_PATH.parents[1]
        conditioner = MODULE.ActionDrivenFlowConditioner(
            backend=backend,
            kineworld_root=kineworld_root,
            flow_method="farneback",
            flow_max_magnitude=25.0,
            flow_processor=self.fake_flow_processor,
            codec=object(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            hdf5_path = Path(temporary) / "episode1.hdf5"
            vectors = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)
            vectors[:, 6] = np.linspace(0.0, 1.0, 12)
            vectors[:, 13] = np.linspace(1.0, 0.0, 12)
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset("joint_action/vector", data=vectors)
            result = conditioner.get_chunk_flow(
                episode_id=1,
                hdf5_path=hdf5_path,
                first_frame=Image.new("RGB", (640, 480), "black"),
                frame_count=12,
                chunk_start=8,
                keyframe_count=9,
                visual_stride=4,
                target_size=(32, 24),
            )

        self.assertEqual(set(result), {"frames", "provenance"})
        self.assertEqual(len(result["frames"]), 9)
        self.assertEqual(
            result["provenance"]["chunk"]["source_indices"],
            [8, 11, 11, 11, 11, 11, 11, 11, 11],
        )
        np.testing.assert_array_equal(
            backend.received,
            vectors[[8, 11, 11, 11, 11, 11, 11, 11, 11]],
        )
        self.assertEqual(
            result["frames"][0].getextrema(),
            ((255, 255), (255, 255), (255, 255)),
        )

    def test_wrong_action_width_fails_before_rendering(self) -> None:
        backend = self.FakeRenderer()
        conditioner = MODULE.ActionDrivenFlowConditioner(
            backend=backend,
            kineworld_root=MODULE_PATH.parents[1],
            flow_method="farneback",
            flow_processor=self.fake_flow_processor,
            codec=object(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            hdf5_path = Path(temporary) / "bad.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset(
                    "joint_action/vector", data=np.zeros((9, 13), dtype=np.float32)
                )
            with self.assertRaises(MODULE.ActionSchemaError):
                conditioner.get_chunk_flow(
                    episode_id=1,
                    hdf5_path=hdf5_path,
                    first_frame=Image.new("RGB", (640, 480)),
                    frame_count=9,
                    chunk_start=0,
                    target_size=(32, 24),
                )
        self.assertIsNone(backend.received)


if __name__ == "__main__":
    unittest.main()
