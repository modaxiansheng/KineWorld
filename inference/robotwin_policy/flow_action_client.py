"""
Minimal WebSocket client for the flow-action inference server.

Runs in the RoboTwin conda environment.
Dependencies: websockets, msgpack, numpy (no torch / diffsynth required).

Usage:
    client = FlowActionClient(host="localhost", port=8000)
    result = client.infer(obs_dict)   # -> {"actions": ndarray(N, action_dim)}
    client.reset()                    # clear episode state on server
"""

import logging
import time
from typing import Dict, Tuple

import numpy as np
import websockets.sync.client

import msgpack_numpy

log = logging.getLogger("flow_action_client")


class FlowActionClient:
    """WebSocket client following the standard websocket-policy interface."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        log.info(f"Waiting for server at {self._uri} ...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    ping_interval=None, ping_timeout=None,
                    close_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                log.info(f"Connected — server metadata: {metadata}")
                return conn, metadata
            except ConnectionRefusedError:
                log.info("Still waiting for server ...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        """Send an observation, receive the action dict.

        Args:
            obs: {
                "images": {cam_name: (H, W, 3) uint8 ndarray, ...},
                "qpos": (action_dim,) float32 ndarray (absolute qpos),
                "instruction": str,
            }

        Returns:
            {"actions": (N, action_dim) float32 ndarray}  — absolute qpos.
        """
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, task_name: str = None) -> None:
        """Tell the server to clear per-episode buffers."""
        msg = {"__reset__": True}
        if task_name is not None:
            msg["task_name"] = task_name
        data = self._packer.pack(msg)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error during reset:\n{response}")
        result = msgpack_numpy.unpackb(response)
        log.info(f"Reset: {result}")

    def close(self) -> None:
        self._ws.close()
