"""
RoboTwin-compatible model wrapper for the KineWorld flow-action policy.

Bridges the WebSocket client with the RoboTwin evaluation loop. Runs in the
RoboTwin conda environment.

Single-frame I2V conditioning (no historical frames): the server consumes only
the current per-camera frame plus the current qpos. action[0] in the returned
chunk is GT-anchored to the current qpos (mirrors training); the client skips it
via ``pixel_prefix=1``.

Chunk layout (must match the server's num_frames and deploy_policy.yml): the
server returns ``num_frames`` actions; the client trims to
``action_chunk_size = 1 (anchor) + EXECUTE_WINDOW`` and deploy_policy executes
the future actions. ``overlap`` cross-fade is supported and defaults to 0.
"""

import numpy as np
from typing import Optional

from flow_action_client import FlowActionClient


class KineWorldPolicy:
    """Wraps the remote inference server as a RoboTwin-compatible model."""

    # action[0] is always the GT anchor and the only token deploy_policy skips.
    PIXEL_PREFIX = 1

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        action_chunk_size: int = 49,
        overlap: int = 0,
        task_name: Optional[str] = None,
    ):
        self.client = FlowActionClient(host=host, port=port)
        self.action_chunk_size = action_chunk_size
        self.overlap = max(0, int(overlap))
        self.task_name = task_name

        self.instruction: Optional[str] = None
        self.observation_window = None

        metadata = self.client.server_metadata
        self.cameras = metadata.get(
            "cameras", ["head_camera", "left_camera", "right_camera"]
        )
        self.num_frames = int(metadata.get("num_frames", action_chunk_size))
        self.pixel_prefix = self.PIXEL_PREFIX

        # Cached most-recent observation (single frame + qpos); the server only
        # consumes the latest observation, so no sliding history is kept.
        self._latest_frame: Optional[dict] = None
        self._latest_qpos: Optional[np.ndarray] = None

        # Previous chunk's last OVERLAP actions, kept for the cross-fade at the
        # next chunk boundary. With overlap=0 (default) it stays None.
        self._prev_overlap: Optional[np.ndarray] = None

        # The server returns `num_frames` actions; the client uses
        # [0:action_chunk_size] for execution and
        # [action_chunk_size : action_chunk_size + overlap] as the cross-fade
        # tail for the next chunk. When overlap > 0 both slices must fit inside
        # `num_frames`, else the cross-fade silently no-ops at each boundary.
        if self.action_chunk_size + self.overlap > self.num_frames:
            raise ValueError(
                f"Invalid chunk layout: action_chunk_size={self.action_chunk_size} "
                f"+ overlap={self.overlap} = {self.action_chunk_size + self.overlap} "
                f"> num_frames={self.num_frames}. Reduce action_chunk_size to <= "
                f"{self.num_frames - self.overlap} or set overlap=0."
            )
        if self.action_chunk_size <= self.pixel_prefix:
            raise ValueError(
                f"Invalid chunk layout: action_chunk_size={self.action_chunk_size} "
                f"<= pixel_prefix={self.pixel_prefix}. No future actions would "
                f"be executed each chunk."
            )

        print(f"[KineWorld] Connected to server — metadata: {metadata}")
        print(
            f"[KineWorld] action_chunk_size={self.action_chunk_size}, "
            f"pixel_prefix={self.pixel_prefix}, overlap={self.overlap} "
            f"(execute={self.action_chunk_size - self.pixel_prefix}/chunk, "
            f"cross_fade={'on' if self.overlap > 0 else 'off'})"
        )

    # ---- Interface expected by deploy_policy.py ----

    def set_language(self, instruction: str):
        self.instruction = instruction
        print(f"[KineWorld] Instruction: {instruction}")

    def has_history(self) -> bool:
        """True iff at least one observation has been pushed.

        ``deploy_policy.eval`` uses this to seed the cache with the current
        observation only on cold-start (right after ``reset()``); on later
        chunks the cache already holds the observation pushed at the end of the
        previous chunk's execution loop.
        """
        return self._latest_frame is not None

    def should_request_observation(self) -> bool:
        """Hint to the outer eval loop: True iff the next ``get_action`` needs a
        fresh observation.

        With single-frame I2V conditioning only ``_latest_frame`` is consumed,
        and it is already filled at the END of the previous chunk's execution
        loop; a fresh outer-loop render is only needed at cold-start.
        """
        return self._latest_frame is None

    def update_observation_window(
        self, img_arr: list, state: np.ndarray
    ):
        """Cache the latest observation.

        Args:
            img_arr: list of 3 images [head, right, left] as (H,W,3) uint8,
                     following RoboTwin's encode_obs convention.
            state:   (14,) float32 joint state vector (absolute qpos).
        """
        # encode_obs() returns true RGB ([head, right, left]); pass it straight
        # through with no channel flip (train == infer == RGB).
        frame_dict = {
            "head_camera": img_arr[0].copy(),
            "left_camera": img_arr[2].copy(),
            "right_camera": img_arr[1].copy(),
        }
        self._latest_frame = frame_dict
        self._latest_qpos = np.asarray(state, dtype=np.float32).copy()

        self.observation_window = {
            "images": frame_dict,
            "instruction": self.instruction or "",
            "qpos": self._latest_qpos,
        }

    def get_action(self) -> np.ndarray:
        """Send the latest observation to the server, return absolute qpos actions.

        Returns ``(action_chunk_size, action_dim)`` float32. ``action[0]`` is
        the GT anchor (= current qpos); the remaining
        ``action_chunk_size - pixel_prefix`` are future predictions to execute.
        When ``overlap > 0`` the first ``overlap`` predictions are cross-faded
        with the previous chunk's stored tail.
        """
        assert self.observation_window is not None, (
            "Call update_observation_window first!"
        )

        result = self.client.infer(self.observation_window)
        actions = np.asarray(result["actions"], dtype=np.float32)

        # ---- Chunk overlap cross-fade ----
        head_start = self.pixel_prefix
        head_end = head_start + self.overlap
        if (
            self.overlap > 0
            and self._prev_overlap is not None
            and head_end <= len(actions)
        ):
            head = actions[head_start:head_end].copy()
            # Linear cross-fade: weights[0] mostly previous, weights[-1] mostly
            # new; endpoints 0 and 1 excluded to avoid degenerate blends.
            weights = np.linspace(0.0, 1.0, self.overlap + 2, dtype=np.float32)[1:-1]
            weights = weights.reshape(-1, 1)
            actions[head_start:head_end] = (
                (1.0 - weights) * self._prev_overlap + weights * head
            )

        # Save this chunk's predicted-but-not-executed tail for the next chunk's
        # cross-fade (the OVERLAP frames just after what deploy_policy executes).
        store_start = self.action_chunk_size
        store_end = store_start + self.overlap
        if self.overlap > 0 and store_end <= len(actions):
            self._prev_overlap = actions[store_start:store_end].copy()
        else:
            self._prev_overlap = None

        return actions[: self.action_chunk_size]

    def reset(self):
        """Reset episode state on the server and locally."""
        self.client.reset(task_name=self.task_name)
        self.instruction = None
        self.observation_window = None
        self._latest_frame = None
        self._latest_qpos = None
        self._prev_overlap = None
        print("[KineWorld] Episode reset")
