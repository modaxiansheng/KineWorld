#!/usr/bin/env python3
"""CPU-only unit tests for strict Track-1 action provenance validation."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from validate_track1_outputs import (
    ACTION_DIM,
    CHUNK_HORIZON,
    KEYFRAMES_PER_CHUNK,
    VISUAL_STRIDE,
    compute_action_chunk_sha256,
    canonical_signature,
    expected_action_indices,
    expected_chunk_starts,
    invalid_sha256_fields,
    sha256_file,
    validate_action_chunk_provenance,
    validate_run_manifest,
)


def write_rgb8_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    width, height = size
    scanline = b"\x00" + bytes(color) * width
    raw = scanline * height

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class FakeActionDataset:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.shape = (len(rows), ACTION_DIM)

    def __getitem__(self, index: int) -> list[float]:
        return self.rows[index]


class ActionDigestValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.hdf5_path = self.root / "episode1.hdf5"
        self.hdf5_path.write_bytes(b"test-hdf5-identity")
        self.urdf_path = self.root / "aloha.urdf"
        self.config_path = self.root / "config.yml"
        self.renderer_path = self.root / "robot_only_renderer.py"
        self.urdf_path.write_bytes(b"<robot name='aloha-agilex'/>")
        self.config_path.write_bytes(b"urdf_path: aloha.urdf\n")
        self.renderer_path.write_bytes(b"# strict renderer\n")
        self.renderer = {
            "embodiment": "aloha-agilex",
            "camera": {"name": "head_camera"},
            "action_layout": [
                "left_arm[0:6]",
                "left_gripper[6]",
                "right_arm[7:13]",
                "right_gripper[13]",
            ],
            "urdf": {
                "path": str(self.urdf_path),
                "sha256": sha256_file(self.urdf_path),
            },
            "embodiment_config": {
                "path": str(self.config_path),
                "sha256": sha256_file(self.config_path),
            },
            "renderer_module": {
                "path": str(self.renderer_path),
                "sha256": sha256_file(self.renderer_path),
            },
        }
        self.frame_count = 75
        rows = []
        for frame_index in range(self.frame_count):
            row = [frame_index * 0.01 + column * 0.001 for column in range(ACTION_DIM)]
            row[6] = (frame_index % 10) / 10.0
            row[13] = ((frame_index + 3) % 10) / 10.0
            rows.append(row)
        self.dataset = FakeActionDataset(rows)
        self.chunks = self._valid_chunks()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _valid_chunks(self) -> list[dict]:
        chunks = []
        for chunk_index, chunk_start in enumerate(
            expected_chunk_starts(self.frame_count)
        ):
            indices = expected_action_indices(chunk_start, self.frame_count)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "keyframe_count": KEYFRAMES_PER_CHUNK,
                    "generated_frame_range": [
                        chunk_start,
                        chunk_start + CHUNK_HORIZON,
                    ],
                    "consumed_frame_range": [
                        chunk_start,
                        min(chunk_start + CHUNK_HORIZON, self.frame_count - 1),
                    ],
                    "flow_conditioning": {
                        "mode": "action_flow",
                        "control_type": "official_14d_joint_action",
                        "episode_id": 1,
                        "source": {
                            "hdf5_path": str(self.hdf5_path),
                            "dataset": "/joint_action/vector",
                            "shape": [self.frame_count, ACTION_DIM],
                            "selected_action_sha256_float32_le": (
                                compute_action_chunk_sha256(self.dataset, indices)
                            ),
                        },
                        "chunk": {
                            "start": chunk_start,
                            "keyframe_count": KEYFRAMES_PER_CHUNK,
                            "visual_stride": VISUAL_STRIDE,
                            "source_indices": indices,
                            "tail_clamped": len(set(indices)) != len(indices),
                        },
                        "first_frame_size": [640, 480],
                        "target_size": [320, 240],
                        "flow": {"frame_zero": "white_zero_flow_sentinel"},
                        "renderer": copy.deepcopy(self.renderer),
                    },
                    "flow_model_adapter": {
                        "source_size": [320, 240],
                        "target_size": [320, 240],
                        "resize_policy": "identity",
                    },
                }
            )
        return chunks

    def _validate(
        self,
        chunks: list[dict],
        expected_conditioner: dict | None = None,
    ) -> tuple[dict[str, bool], list[str]]:
        checks, errors, renderer_identity = validate_action_chunk_provenance(
            chunks=chunks,
            action_dataset=self.dataset,
            episode_id=1,
            frame_count=self.frame_count,
            hdf5_path=self.hdf5_path,
            native_size=(320, 240),
            first_frame_size=(640, 480),
            expected_conditioner=expected_conditioner,
        )
        self.assertIsNotNone(renderer_identity)
        return checks, errors

    def _precomputed_chunks(self) -> tuple[list[dict], dict]:
        chunks = copy.deepcopy(self.chunks)
        precomputed_root = self.root / "precomputed"
        conditioner = {
            "schema_version": 1,
            "manifest_schema_version": 2,
            "conditioner": "precomputed_action_flow.v1",
            "payload": "frames",
            "root": str(precomputed_root),
            "layout": "episode<ID>/chunk_<START:06d>/manifest.json",
            "validation": "selected_float32_action_sha256+png_file_sha256",
        }
        for chunk in chunks:
            flow = chunk["flow_conditioning"]
            chunk_start = flow["chunk"]["start"]
            chunk_dir = (
                precomputed_root / "episode1" / f"chunk_{chunk_start:06d}"
            )
            chunk_dir.mkdir(parents=True)
            frame_names = [
                f"flow_{frame_index:02d}.png"
                for frame_index in range(KEYFRAMES_PER_CHUNK)
            ]
            for frame_index, frame_name in enumerate(frame_names):
                color = (255, 255, 255) if frame_index == 0 else (
                    frame_index,
                    20,
                    30,
                )
                write_rgb8_png(chunk_dir / frame_name, (320, 240), color)
            frame_sha256 = {
                frame_name: sha256_file(chunk_dir / frame_name)
                for frame_name in frame_names
            }
            manifest = {
                "schema_version": 2,
                "episode_id": 1,
                "chunk_start": chunk_start,
                "frame_count": self.frame_count,
                "keyframe_count": KEYFRAMES_PER_CHUNK,
                "visual_stride": VISUAL_STRIDE,
                "source_indices": flow["chunk"]["source_indices"],
                "target_size": [320, 240],
                "action_sha256_float32_le": flow["source"][
                    "selected_action_sha256_float32_le"
                ],
                "frames": frame_names,
                "frame_sha256": frame_sha256,
                "provenance": copy.deepcopy(flow),
            }
            manifest_path = chunk_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            flow["precomputed"] = {
                "root": str(precomputed_root),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        return chunks, conditioner

    @staticmethod
    def _rewrite_manifest(chunks: list[dict], chunk_index: int, manifest: dict) -> None:
        precomputed = chunks[chunk_index]["flow_conditioning"]["precomputed"]
        manifest_path = Path(precomputed["manifest"])
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        precomputed["manifest_sha256"] = sha256_file(manifest_path)

    def test_real_float32_little_endian_digest_passes(self) -> None:
        indices = expected_action_indices(64, self.frame_count)
        independent_bytes = b"".join(
            struct.pack("<14f", *self.dataset[index]) for index in indices
        )
        self.assertEqual(
            compute_action_chunk_sha256(self.dataset, indices),
            hashlib.sha256(independent_bytes).hexdigest(),
        )
        checks, errors = self._validate(self.chunks)
        self.assertTrue(all(checks.values()), errors)
        self.assertEqual(errors, [])

    def test_training_parity_resize_from_320x240_to_320x256_passes(self) -> None:
        resized = copy.deepcopy(self.chunks)
        for chunk in resized:
            chunk["flow_model_adapter"] = {
                "source_size": [320, 240],
                "target_size": [320, 256],
                "resize_policy": (
                    "pil_rgb_default_resize_matching_training_online_v1"
                ),
            }
        checks, errors, renderer_identity = validate_action_chunk_provenance(
            chunks=resized,
            action_dataset=self.dataset,
            episode_id=1,
            frame_count=self.frame_count,
            hdf5_path=self.hdf5_path,
            native_size=(320, 256),
            flow_source_size=(320, 240),
            first_frame_size=(640, 480),
        )
        self.assertIsNotNone(renderer_identity)
        self.assertTrue(all(checks.values()), errors)
        self.assertEqual(errors, [])

    def test_non_hex_forged_digest_fails_closed(self) -> None:
        forged = copy.deepcopy(self.chunks)
        forged[0]["flow_conditioning"]["source"][
            "selected_action_sha256_float32_le"
        ] = "not-a-hash"
        checks, errors = self._validate(forged)
        self.assertFalse(checks["provenance_action_digests_recomputed"])
        self.assertTrue(any("not 64-digit hex" in error for error in errors))

    def test_well_formed_but_wrong_digest_fails_closed(self) -> None:
        forged = copy.deepcopy(self.chunks)
        forged[1]["flow_conditioning"]["source"][
            "selected_action_sha256_float32_le"
        ] = "0" * 64
        checks, errors = self._validate(forged)
        self.assertFalse(checks["provenance_action_digests_recomputed"])
        self.assertTrue(any("digest mismatch" in error for error in errors))

    def test_digest_is_recomputed_from_declared_source_indices(self) -> None:
        forged = copy.deepcopy(self.chunks)
        forged[0]["flow_conditioning"]["chunk"]["source_indices"][0] = 1
        checks, errors = self._validate(forged)
        self.assertFalse(checks["provenance_action_chunk_structure"])
        self.assertFalse(checks["provenance_action_digests_recomputed"])
        self.assertTrue(any("digest mismatch" in error for error in errors))

    def test_precomputed_v2_dynamic_tail_passes(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        checks, errors = self._validate(chunks, conditioner)
        self.assertTrue(all(checks.values()), errors)
        self.assertEqual(errors, [])
        tail = chunks[-1]["flow_conditioning"]["chunk"]
        self.assertEqual(tail["start"], 64)
        self.assertEqual(
            tail["source_indices"],
            expected_action_indices(64, self.frame_count),
        )
        self.assertTrue(tail["tail_clamped"])

    def test_precomputed_missing_chunk_block_fails_closed(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        chunks[-1]["flow_conditioning"].pop("precomputed")
        checks, errors = self._validate(chunks, conditioner)
        self.assertFalse(checks["provenance_precomputed_manifests_verified"])
        self.assertTrue(
            any("lacks required precomputed provenance" in error for error in errors)
        )

    def test_precomputed_white_png_replacement_fails_closed(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        manifest_path = Path(
            chunks[0]["flow_conditioning"]["precomputed"]["manifest"]
        )
        write_rgb8_png(manifest_path.parent / "flow_01.png", (320, 240), (255, 255, 255))
        checks, errors = self._validate(chunks, conditioner)
        self.assertFalse(checks["provenance_precomputed_manifests_verified"])
        self.assertTrue(any("frame SHA-256 mismatch" in error for error in errors))

    def test_precomputed_legacy_v1_manifest_fails_closed(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        manifest_path = Path(
            chunks[0]["flow_conditioning"]["precomputed"]["manifest"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 1
        manifest.pop("frame_sha256")
        self._rewrite_manifest(chunks, 0, manifest)
        checks, errors = self._validate(chunks, conditioner)
        self.assertFalse(checks["provenance_precomputed_manifests_verified"])
        self.assertTrue(any("schema_version=1" in error for error in errors))

    def test_precomputed_v2_requires_all_nine_frame_hashes(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        manifest_path = Path(
            chunks[0]["flow_conditioning"]["precomputed"]["manifest"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["frame_sha256"].pop(manifest["frames"][-1])
        self._rewrite_manifest(chunks, 0, manifest)
        checks, errors = self._validate(chunks, conditioner)
        self.assertFalse(checks["provenance_precomputed_manifests_verified"])
        self.assertTrue(any("does not cover all nine" in error for error in errors))

    def test_precomputed_first_frame_must_be_white_even_when_rehashed(self) -> None:
        chunks, conditioner = self._precomputed_chunks()
        manifest_path = Path(
            chunks[0]["flow_conditioning"]["precomputed"]["manifest"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_name = manifest["frames"][0]
        first_path = manifest_path.parent / first_name
        write_rgb8_png(first_path, (320, 240), (1, 2, 3))
        manifest["frame_sha256"][first_name] = sha256_file(first_path)
        self._rewrite_manifest(chunks, 0, manifest)
        checks, errors = self._validate(chunks, conditioner)
        self.assertFalse(checks["provenance_precomputed_manifests_verified"])
        self.assertTrue(any("not a white sentinel" in error for error in errors))

    def _run_manifest(self) -> dict:
        conditioner = {
            "schema_version": 1,
            "conditioner": "action_driven_robot_only_flow.v1",
            "payload": "frames",
            "renderer": copy.deepcopy(self.renderer),
        }
        signature_inputs = {
            "checkpoint_sha256": "1" * 64,
            "conditioning_mode": "action_flow",
            "control_type": "action_driven",
            "action_flow_conditioner": conditioner,
            "action_flow_provider_kwargs_sha256": "2" * 64,
            "keyframes_per_chunk": 9,
            "visual_stride": 4,
            "frame_count_source": "/joint_action/vector",
            "output_width": 640,
            "output_height": 480,
            "fps": 24,
            "video_codec": "h264/libx264/yuv420p",
        }
        return {
            "schema_version": 1,
            "run_signature": canonical_signature(signature_inputs),
            "signature_inputs": signature_inputs,
            "dataset_root": str(self.root),
            "checkpoint": {"sha256": "1" * 64},
            "conditioning": {
                "mode": "action_flow",
                "control_type": "action_driven",
                "conditioner": conditioner,
                "zero_flow_is_explicit_baseline": False,
            },
        }

    def test_run_manifest_signature_and_renderer_files_pass(self) -> None:
        manifest = self._run_manifest()
        checks, errors, summary = validate_run_manifest(
            run_manifest=manifest,
            run_manifest_path=self.root / "run_config.json",
            dataset_root=self.root,
            required_control_type="action_driven",
            expected_fps=24.0,
            expected_width=640,
            expected_height=480,
        )
        self.assertTrue(all(checks.values()), errors)
        self.assertEqual(errors, [])
        self.assertIsNotNone(summary["renderer_identity"])

    def test_run_precomputed_conditioner_requires_manifest_v2(self) -> None:
        _chunks, conditioner = self._precomputed_chunks()
        conditioner["manifest_schema_version"] = 1
        manifest = self._run_manifest()
        manifest["conditioning"]["conditioner"] = conditioner
        manifest["signature_inputs"]["action_flow_conditioner"] = conditioner
        manifest["run_signature"] = canonical_signature(manifest["signature_inputs"])
        checks, errors, _summary = validate_run_manifest(
            run_manifest=manifest,
            run_manifest_path=self.root / "run_config.json",
            dataset_root=self.root,
            required_control_type="action_driven",
            expected_fps=24.0,
            expected_width=640,
            expected_height=480,
        )
        self.assertFalse(checks["run_action_provider_trusted"])
        self.assertTrue(any("manifest schema v2" in error for error in errors))

    def test_forged_urdf_sha_fails_even_with_recomputed_run_signature(self) -> None:
        manifest = self._run_manifest()
        conditioner = manifest["conditioning"]["conditioner"]
        conditioner["renderer"]["urdf"]["sha256"] = "0" * 64
        manifest["signature_inputs"]["action_flow_conditioner"] = conditioner
        manifest["run_signature"] = canonical_signature(manifest["signature_inputs"])
        checks, errors, _summary = validate_run_manifest(
            run_manifest=manifest,
            run_manifest_path=self.root / "run_config.json",
            dataset_root=self.root,
            required_control_type="action_driven",
            expected_fps=24.0,
            expected_width=640,
            expected_height=480,
        )
        self.assertFalse(checks["run_action_provider_trusted"])
        self.assertTrue(any("URDF SHA-256 mismatch" in error for error in errors))

    def test_non_hex_sha_field_is_reported_recursively(self) -> None:
        invalid = invalid_sha256_fields(
            {"source": {"selected_action_sha256_float32_le": "not-a-hash"}}
        )
        self.assertEqual(
            invalid,
            ["$.source.selected_action_sha256_float32_le"],
        )


if __name__ == "__main__":
    unittest.main()
