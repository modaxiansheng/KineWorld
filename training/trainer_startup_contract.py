"""Fail-closed trainer startup handshake for multi-node launchers.

The host launcher must not treat a container as ready merely because its
entrypoint is running.  A trainer publishes this sentinel only after every
distributed rank has completed model/data preparation and exact-resume load.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


class TrainerStartupError(RuntimeError):
    """The trainer readiness contract is incomplete or unsafe."""


def _fsync_directory(path: str) -> None:
    """Persist a directory entry where the platform supports directory fsync."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not permit opening a directory with os.open.  The HCU
        # production path is Linux, where failure must remain fail-closed.
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_trainer_ready(
    accelerator,
    ready_file: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Atomically publish a per-node readiness token after a global barrier.

    ``accelerator.is_local_main_process`` selects exactly one writer per node,
    while both barriers include every distributed rank.  With neither setting
    present the helper is a no-op for ordinary, non-orchestrated training.
    Supplying only one setting is always an error.

    Returns ``True`` only on the local process that wrote the sentinel.
    """

    if ready_file is None:
        ready_file = os.environ.get("TRAINER_READY_FILE")
    if token is None:
        token = os.environ.get("TRAINER_READY_TOKEN")

    if not ready_file and not token:
        return False
    if not ready_file or not token:
        raise TrainerStartupError(
            "TRAINER_READY_FILE and TRAINER_READY_TOKEN must be set together"
        )
    if not os.path.isabs(ready_file):
        raise TrainerStartupError(
            f"TRAINER_READY_FILE must be absolute: {ready_file!r}"
        )
    if "\n" in token or "\r" in token:
        raise TrainerStartupError("TRAINER_READY_TOKEN must be one line")

    parent = os.path.dirname(ready_file)
    temporary = None
    wrote = False
    try:
        # Phase 1: every rank has reached this point after prepare/resume.
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            os.makedirs(parent, mode=0o750, exist_ok=True)
            if os.path.lexists(ready_file):
                raise TrainerStartupError(
                    "refusing pre-existing trainer readiness sentinel: "
                    f"{ready_file}"
                )

            descriptor, temporary = tempfile.mkstemp(
                dir=parent,
                prefix=f".{os.path.basename(ready_file)}.",
                suffix=".pending",
                text=True,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(token)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Persist only a non-ready temporary entry before the second
            # global phase.  A rank or node that cannot prepare its token
            # therefore prevents every final ready marker from appearing.
            _fsync_directory(parent)

        # Phase 2: all per-node writers prepared and persisted their token.
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # This rename is deliberately the final fallible operation.  Once
            # the host-visible marker exists there is no later barrier/fsync
            # that can fail while the launcher has already released its lock.
            os.replace(temporary, ready_file)
            temporary = None
            wrote = True
        return wrote
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
