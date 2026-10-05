"""CPU-only synthetic schema fixtures, not demonstration data or model results."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

import prepare_inference_episode as prepare
from robotwin_data_io import decode_rgb_frame, read_joint_vectors


def jpeg_bytes(*, marked: bool = False) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), (190, 20, 45)).save(stream, format="JPEG", quality=95)
    value = stream.getvalue()
    if marked:
        # Synthetic fixture header uses the documented upstream COM payload.
        marker = b"XPL-RGB1"
        value = value[:2] + b"\xff\xfe" + (len(marker) + 2).to_bytes(2, "big") + marker + value[2:]
    return value


class DecodeTests(unittest.TestCase):
    def test_auto_legacy_swaps_and_marked_standard_does_not(self):
        legacy, metadata = decode_rgb_frame(jpeg_bytes())
        standard, marked = decode_rgb_frame(jpeg_bytes(marked=True))
        self.assertTrue(metadata["channel_swap"])
        self.assertFalse(marked["channel_swap"])
        self.assertEqual(marked["standard_rgb_marker"], "XPL-RGB1")
        np.testing.assert_array_equal(np.asarray(legacy), np.asarray(standard)[..., ::-1])

    def test_padded_bytes_and_vlen_uint8_buffers(self):
        data = jpeg_bytes(marked=True)
        expected, _ = decode_rgb_frame(data)
        for value in (np.bytes_(data + b"\0" * 64), np.frombuffer(data, dtype=np.uint8), np.void(data)):
            actual, _ = decode_rgb_frame(value)
            np.testing.assert_array_equal(actual, expected)

    def test_explicit_standard_and_marker_conflict(self):
        image, metadata = decode_rgb_frame(jpeg_bytes(), image_encoding="standard-rgb")
        self.assertFalse(metadata["channel_swap"])
        self.assertGreater(image.getpixel((0, 0))[0], image.getpixel((0, 0))[2])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            decode_rgb_frame(jpeg_bytes(marked=True), image_encoding="robotwin-legacy")

    def test_decoded_rgb_arrays_are_not_swapped(self):
        values = np.full((4, 6, 3), [10, 20, 240], dtype=np.uint8)
        image, _ = decode_rgb_frame(values)
        np.testing.assert_array_equal(image, values)

    def test_unknown_container_is_not_guessed(self):
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(stream, format="PNG")
        with self.assertRaisesRegex(ValueError, "JPEG only"):
            decode_rgb_frame(stream.getvalue())
        image, _ = decode_rgb_frame(stream.getvalue(), image_encoding="standard-rgb")
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw.hdf5"
        self.instruction = self.root / "instruction.json"
        self.output = self.root / "prepared"
        self.actions = np.arange(70, dtype=np.float32).reshape(5, 14) / 100
        self.actions[:, 6] = 0.25
        self.actions[:, 13] = 0.75
        with h5py.File(self.source, "w") as handle:
            handle.attrs["fixture_only"] = True
            group = handle.create_group("joint_action")
            for name, values in (
                ("left_arm", self.actions[:, :6]), ("left_gripper", self.actions[:, 6]),
                ("right_arm", self.actions[:, 7:13]), ("right_gripper", self.actions[:, 13:14]),
            ):
                group.create_dataset(name, data=values)
            encoded = jpeg_bytes()
            handle.create_dataset("observation/head_camera/rgb", data=np.asarray([encoded] * 5, dtype=f"S{len(encoded)}"))
            handle.create_dataset("metadata/calibration", data=np.eye(3))
        self.instruction.write_text(json.dumps({"seen": ["First source text.", "第二条源指令。"], "unseen": ["Held out."]}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *extra):
        return prepare.build_parser().parse_args([
            "--hdf5", str(self.source), "--instruction-json", str(self.instruction),
            "--output-root", str(self.output), *extra,
        ])

    def test_full_trajectory_metadata_and_selected_instruction_round_trip(self):
        before = prepare.sha256_file(self.source)
        record = prepare.prepare_episode(self.args("--instruction-index", "1"))
        exported = self.output / record["output_paths"]["hdf5"]
        with h5py.File(exported, "r") as handle:
            np.testing.assert_array_equal(handle["joint_action/vector"][:], self.actions)
            np.testing.assert_array_equal(handle["joint_action/left_arm"][:], self.actions[:, :6])
            np.testing.assert_array_equal(handle["metadata/calibration"][:], np.eye(3))
            self.assertTrue(handle.attrs["fixture_only"])
            self.assertEqual(len(handle["observation/head_camera/rgb"]), 5)
        self.assertEqual(prepare.sha256_file(self.source), before)
        self.assertEqual(record["trajectory_frame_count"], 5)
        self.assertFalse(record["temporal_resampling"])
        self.assertEqual(record["instruction_selection"]["text"], "第二条源指令。")
        text = json.loads((self.output / record["output_paths"]["instruction"]).read_text(encoding="utf-8"))
        self.assertEqual(text["instruction"], "第二条源指令。")
        for kind, digest in record["output_sha256"].items():
            self.assertEqual(prepare.sha256_file(self.output / record["output_paths"][kind]), digest)

    def test_existing_vector_source_is_copied_byte_for_byte(self):
        with h5py.File(self.source, "r+") as handle:
            handle.create_dataset("joint_action/vector", data=self.actions)
        record = prepare.prepare_episode(self.args())
        self.assertEqual(record["source_hdf5_sha256"], record["output_sha256"]["hdf5"])

    def test_refuses_overwrite_without_altering_first_export(self):
        first = prepare.prepare_episode(self.args())
        with self.assertRaises(FileExistsError):
            prepare.prepare_episode(self.args())
        for kind, digest in first["output_sha256"].items():
            self.assertEqual(prepare.sha256_file(self.output / first["output_paths"][kind]), digest)

    def test_second_id_can_be_appended(self):
        prepare.prepare_episode(self.args())
        second = prepare.prepare_episode(self.args("--episode-id", "2"))
        self.assertTrue((self.output / second["output_paths"]["hdf5"]).is_file())

    def test_rgb_action_length_mismatch_fails_before_output(self):
        with h5py.File(self.source, "r+") as handle:
            del handle["observation/head_camera/rgb"]
            handle.create_dataset("observation/head_camera/rgb", data=np.asarray([jpeg_bytes()], dtype="S1024"))
        with self.assertRaisesRegex(ValueError, "frame counts"):
            prepare.prepare_episode(self.args())
        self.assertFalse(self.output.exists())

    def test_split_gripper_width_not_silently_truncated(self):
        with h5py.File(self.source, "r+") as handle:
            del handle["joint_action/left_gripper"]
            handle.create_dataset("joint_action/left_gripper", data=np.zeros((5, 2)))
        with self.assertRaisesRegex(ValueError, "shape"):
            prepare.prepare_episode(self.args())

    def test_existing_vector_must_agree_with_split_fields(self):
        with h5py.File(self.source, "r+") as handle:
            handle.create_dataset("joint_action/vector", data=np.zeros((5, 14)))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare.prepare_episode(self.args())

    def test_nonfinite_and_out_of_range_gripper_fail(self):
        for value in (float("nan"), 2.0):
            with h5py.File(self.source, "r+") as handle:
                handle["joint_action/left_gripper"][2] = value
            with self.assertRaises(ValueError):
                prepare.prepare_episode(self.args())
        self.assertFalse(self.output.exists())

    def test_instruction_is_not_fabricated_when_missing(self):
        self.instruction.write_text('{"unrelated":"no prompt"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source instruction"):
            prepare.prepare_episode(self.args())

    def test_vector_only_and_incomplete_split(self):
        with h5py.File(self.source, "r+") as handle:
            del handle["joint_action"]
            handle.create_dataset("joint_action/vector", data=self.actions)
            actual, source = read_joint_vectors(handle)
            np.testing.assert_array_equal(actual, self.actions)
            self.assertEqual(source, "/joint_action/vector")
            handle.create_dataset("joint_action/left_arm", data=self.actions[:, :6])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_joint_vectors(handle)


if __name__ == "__main__":
    unittest.main()
