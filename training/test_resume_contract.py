"""Pure-CPU tests for the fail-closed exact-resume workload contract."""

import copy
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from resume_contract import (
    RESUME_BINDING_FIELDS,
    ResumeBindingError,
    build_resume_binding,
    build_trainer_state,
    resolve_vram_probe_optimizer_step,
    validate_resume_binding,
    validate_trainer_state,
)


def _binding():
    return build_resume_binding(
        world_size=16,
        per_device_batch=3,
        gradient_accumulation_steps=2,
        dataset_num_workers=0,
        data_shuffle_seed=42,
        training_manifest_sha256="a" * 64,
        video_objective="track1_conditional_rgb",
        flow_mode="robot_only",
        size=[320, 240],
        num_frames=33,
        num_video_frames=9,
        visual_stride=4,
    )


class ResumeContractTests(unittest.TestCase):
    def test_vram_probe_step_is_fresh_absolute_or_resumed_fixed_delay(self):
        self.assertEqual(
            resolve_vram_probe_optimizer_step(configured_probe_step=3),
            3,
        )
        self.assertEqual(
            resolve_vram_probe_optimizer_step(
                configured_probe_step=3,
                restored_optimizer_step=64,
            ),
            67,
        )
        self.assertEqual(
            resolve_vram_probe_optimizer_step(
                configured_probe_step=3,
                restored_optimizer_step=0,
            ),
            3,
        )

    def test_vram_probe_schedule_rejects_invalid_steps(self):
        for invalid_check_step in (0, -1, True, "3"):
            with self.subTest(configured_probe_step=invalid_check_step):
                with self.assertRaises(ResumeBindingError):
                    resolve_vram_probe_optimizer_step(
                        configured_probe_step=invalid_check_step
                    )
        for invalid_restored_step in (-1, True, "64"):
            with self.subTest(restored_optimizer_step=invalid_restored_step):
                with self.assertRaises(ResumeBindingError):
                    resolve_vram_probe_optimizer_step(
                        configured_probe_step=3,
                        restored_optimizer_step=invalid_restored_step,
                    )

    def test_trainer_state_round_trip(self):
        binding = _binding()
        state = build_trainer_state(
            global_step=127,
            optimizer_step=64,
            resume_binding=binding,
        )
        self.assertEqual(
            validate_trainer_state(state, binding),
            (127, 64),
        )
        self.assertEqual(set(state["resume_binding"]), set(RESUME_BINDING_FIELDS))

    def test_every_bound_workload_change_fails_closed(self):
        current = _binding()
        replacements = {
            "schema_version": 2,
            "world_size": 8,
            "per_device_batch": 2,
            "gradient_accumulation_steps": 1,
            "dataset_num_workers": 1,
            "data_shuffle_seed": 43,
            "training_manifest_sha256": "b" * 64,
            "video_objective": "joint_dual_stream",
            "flow_mode": "full_scene",
            "size_w": 384,
            "size_h": 256,
            "num_frames": 49,
            "num_video_frames": 8,
            "visual_stride": 3,
        }
        self.assertEqual(set(replacements), set(RESUME_BINDING_FIELDS))
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                stored = copy.deepcopy(current)
                stored[field] = replacement
                with self.assertRaisesRegex(ResumeBindingError, field):
                    validate_resume_binding(stored, current)

    def test_missing_unknown_and_wrong_type_fields_fail_closed(self):
        current = _binding()

        missing = copy.deepcopy(current)
        del missing["visual_stride"]
        with self.assertRaisesRegex(ResumeBindingError, "missing fields"):
            validate_resume_binding(missing, current)

        unknown = copy.deepcopy(current)
        unknown["future_field"] = "unsafe-to-ignore"
        with self.assertRaisesRegex(ResumeBindingError, "unknown fields"):
            validate_resume_binding(unknown, current)

        wrong_type = copy.deepcopy(current)
        wrong_type["world_size"] = "16"
        with self.assertRaisesRegex(ResumeBindingError, "world_size"):
            validate_resume_binding(wrong_type, current)

    def test_legacy_or_incomplete_trainer_state_is_not_resumable(self):
        binding = _binding()
        for missing_field in ("global_step", "optimizer_step", "resume_binding"):
            with self.subTest(missing_field=missing_field):
                state = build_trainer_state(
                    global_step=12,
                    optimizer_step=6,
                    resume_binding=binding,
                )
                del state[missing_field]
                with self.assertRaisesRegex(
                    ResumeBindingError, missing_field
                ):
                    validate_trainer_state(state, binding)

    def test_invalid_binding_values_are_rejected_when_built(self):
        with self.assertRaisesRegex(ResumeBindingError, "size"):
            build_resume_binding(
                world_size=16,
                per_device_batch=3,
                gradient_accumulation_steps=2,
                dataset_num_workers=0,
                data_shuffle_seed=42,
                training_manifest_sha256="a" * 64,
                video_objective="track1_conditional_rgb",
                flow_mode="robot_only",
                size=[320],
                num_frames=33,
                num_video_frames=9,
                visual_stride=4,
            )


if __name__ == "__main__":
    unittest.main()
