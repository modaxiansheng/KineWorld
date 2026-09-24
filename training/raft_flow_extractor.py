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
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


RAFT_LARGE_DEFAULT_FILENAME = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
RAFT_LARGE_DEFAULT_BYTES = 21_106_607
RAFT_LARGE_DEFAULT_SHA256 = (
    "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_raft_weights_path() -> Path:
    """Resolve the RAFT weights locally without initiating a download."""
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights

    explicit = os.environ.get("KINEWORLD_RAFT_WEIGHTS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    filename = Path(urlparse(Raft_Large_Weights.DEFAULT.url).path).name
    if filename != RAFT_LARGE_DEFAULT_FILENAME:
        raise RuntimeError(
            f"torchvision RAFT DEFAULT changed to {filename!r}; this run is "
            f"locked to {RAFT_LARGE_DEFAULT_FILENAME!r}"
        )
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def verify_raft_weights_file(path: Path | None = None) -> tuple[Path, str]:
    """Require the exact official torchvision RAFT-large file and full SHA."""
    resolved = path or resolve_raft_weights_path()
    resolved = Path(resolved).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"offline RAFT weights are missing: {resolved}; stage the official "
            f"{RAFT_LARGE_DEFAULT_FILENAME} or set KINEWORLD_RAFT_WEIGHTS_PATH"
        )
    size = resolved.stat().st_size
    if size != RAFT_LARGE_DEFAULT_BYTES:
        raise RuntimeError(
            f"RAFT weight byte-size mismatch: {size} != "
            f"{RAFT_LARGE_DEFAULT_BYTES} ({resolved})"
        )
    digest = _sha256_file(resolved)
    if digest != RAFT_LARGE_DEFAULT_SHA256:
        raise RuntimeError(
            f"RAFT weight SHA-256 mismatch: {digest} != "
            f"{RAFT_LARGE_DEFAULT_SHA256} ({resolved})"
        )
    return resolved, digest


def load_raft_large_offline(device: str = "cuda"):
    """Strictly load RAFT-large from the verified local file only."""
    import torch
    from torchvision.models.optical_flow import raft_large

    weights_path, digest = verify_raft_weights_file()
    try:
        state_dict = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # older HCU PyTorch builds
        state_dict = torch.load(weights_path, map_location="cpu")
    model = raft_large(weights=None, progress=False)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict RAFT state load unexpectedly reported missing/unexpected "
            f"keys: {incompatible.missing_keys}/{incompatible.unexpected_keys}"
        )
    return model.to(torch.device(device)).eval(), weights_path, digest


class RAFTFlowExtractor:
    """Dense optical flow via the torchvision RAFT-large pretrained model."""

    def __init__(self, device: str = "cuda"):
        import torch

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model, self.weights_path, self.weights_sha256 = (
            load_raft_large_offline(str(self.device))
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
            torch.amp.autocast("cuda", enabled=self.device.type == "cuda"),
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
                torch.amp.autocast(
                    "cuda", enabled=use_autocast and self.device.type == "cuda"
                ),
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
