"""
RAFT-based dense optical flow extractor (training + inference).

Extracted from ``examples/wanvideo/flow_train/video_flow_codec_pipeline.py``
so the action-expert pipeline (training dataset + inference server) does
NOT depend on files outside ``flow_action_train/robotwin``.

Public API:
    * ``RAFTFlowExtractor`` — torchvision RAFT-large wrapper with batched
      consecutive-pair inference (``__call__`` for single pair,
      ``batch_call`` for an entire frame list).
    * ``compute_flow_farneback`` — classical (CPU) fallback used by the
      ``flow_method='farneback'`` code path.

The two flow back-ends are kept bit-identical to the original
``video_flow_codec_pipeline.py`` definitions; do not edit one without
mirroring the other if/when the upstream copy still exists.
"""
from __future__ import annotations
import os

import cv2
import numpy as np


class RAFTFlowExtractor:
    """Dense optical flow via the torchvision RAFT-large pretrained model."""

    def __init__(self, device: str = "cuda"):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = (
            raft_large(weights=Raft_Large_Weights.DEFAULT)
            .to(self.device)
            .eval()
        )

    def _preprocess(self, frame: np.ndarray):
        """``(H, W, 3) uint8`` → ``(1, 3, H', W')`` normalized + 8x-padded."""
        import torch
        import torch.nn.functional as F

        img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        img = img.unsqueeze(0)

        _, _, h, w = img.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")

        return img.to(self.device), h, w

    def __call__(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute dense flow ``frame1 -> frame2``.

        Returns ``(H, W, 2) float32`` with ``flow[..., 0]=dx``,
        ``flow[..., 1]=dy`` in pixels.
        """
        import torch

        img1, orig_h, orig_w = self._preprocess(frame1)
        img2, _, _ = self._preprocess(frame2)

        # cuDNN's native grid_sampler hits CUDNN_STATUS_NOT_SUPPORTED on
        # Hopper (H20/H100) for the correlation-pyramid sampling shapes
        # used by torchvision's RAFT. Disable cuDNN for the forward pass
        # so PyTorch falls back to the native CUDA kernel.
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda"),
            torch.backends.cudnn.flags(enabled=False),
        ):
            flow_preds = self.model(img1, img2)

        flow = (
            flow_preds[-1]
            .squeeze(0)
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )
        return flow[:orig_h, :orig_w].astype(np.float32)

    def batch_call(
        self,
        frames: list,
        max_batch_size: int = 120,
        use_autocast: bool = True,
    ) -> list:
        """Pairwise flow over ``frames``; returns ``len(frames)-1`` arrays."""
        import torch

        if len(frames) < 2:
            return []

        max_batch_size = int(
            os.environ.get("KINEWORLD_RAFT_MAX_BATCH_SIZE", max_batch_size)
        )
        max_batch_size = max(1, max_batch_size)

        tensors = []
        orig_h, orig_w = None, None
        for frame in frames:
            t, h, w = self._preprocess(frame)
            tensors.append(t)
            if orig_h is None:
                orig_h, orig_w = h, w

        img1_all = torch.cat(tensors[:-1], dim=0)
        img2_all = torch.cat(tensors[1:], dim=0)
        n_pairs = img1_all.shape[0]

        flow_results = []
        for start in range(0, n_pairs, max_batch_size):
            end = min(start + max_batch_size, n_pairs)
            # See note in __call__: disable cuDNN for the RAFT forward to
            # dodge the Hopper/cuDNN grid_sampler bug.
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", enabled=use_autocast),
                torch.backends.cudnn.flags(enabled=False),
            ):
                preds = self.model(
                    img1_all[start:end].contiguous(),
                    img2_all[start:end].contiguous(),
                )
            batch_flow = (
                preds[-1].float().permute(0, 2, 3, 1).cpu().numpy()
            )
            for i in range(batch_flow.shape[0]):
                flow_results.append(
                    batch_flow[i, :orig_h, :orig_w].astype(np.float32)
                )

        return flow_results


def compute_flow_farneback(
    frame1: np.ndarray, frame2: np.ndarray
) -> np.ndarray:
    """Farneback dense optical flow (classical, no GPU required)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    return flow.astype(np.float32)
