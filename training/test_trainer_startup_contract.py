import os
import tempfile
import unittest

from trainer_startup_contract import TrainerStartupError, publish_trainer_ready


class _FakeAccelerator:
    def __init__(self, local_main=True, barrier_hook=None):
        self.is_local_main_process = local_main
        self.barriers = 0
        self.barrier_hook = barrier_hook

    def wait_for_everyone(self):
        self.barriers += 1
        if self.barrier_hook is not None:
            self.barrier_hook(self.barriers)


class TrainerStartupContractTests(unittest.TestCase):
    def test_local_main_atomically_publishes_after_two_barriers(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "startup", "node_0.ready")
            observations = []

            def observe(barrier_number):
                observations.append((barrier_number, os.path.exists(path)))

            accelerator = _FakeAccelerator(barrier_hook=observe)
            self.assertTrue(publish_trainer_ready(accelerator, path, "token-0"))
            self.assertEqual(accelerator.barriers, 2)
            self.assertEqual(observations, [(1, False), (2, False)])
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "token-0\n")

    def test_non_local_main_only_participates_in_barriers(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "node_1.ready")
            accelerator = _FakeAccelerator(local_main=False)
            self.assertFalse(publish_trainer_ready(accelerator, path, "token-1"))
            self.assertEqual(accelerator.barriers, 2)
            self.assertFalse(os.path.exists(path))

    def test_no_configuration_is_a_noop(self):
        accelerator = _FakeAccelerator()
        self.assertFalse(publish_trainer_ready(accelerator, "", ""))
        self.assertEqual(accelerator.barriers, 0)

    def test_partial_or_unsafe_configuration_fails_closed(self):
        accelerator = _FakeAccelerator()
        with self.assertRaises(TrainerStartupError):
            publish_trainer_ready(accelerator, "/tmp/ready", "")
        with self.assertRaises(TrainerStartupError):
            publish_trainer_ready(accelerator, "relative.ready", "token")
        with self.assertRaises(TrainerStartupError):
            publish_trainer_ready(accelerator, "/tmp/ready", "bad\ntoken")
        self.assertEqual(accelerator.barriers, 0)

    def test_preexisting_sentinel_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ready")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("stale\n")
            accelerator = _FakeAccelerator()
            with self.assertRaises(TrainerStartupError):
                publish_trainer_ready(accelerator, path, "fresh")
            self.assertEqual(accelerator.barriers, 1)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "stale\n")


if __name__ == "__main__":
    unittest.main()
