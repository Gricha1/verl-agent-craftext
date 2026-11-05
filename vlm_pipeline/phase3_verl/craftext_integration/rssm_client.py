"""The agent loop's side of phase4_world_model/rssm_service.py: frames go out, h_t, z_t come back."""
from __future__ import annotations

from multiprocessing.connection import Client
from typing import Any

import numpy as np


class RssmClient:
    """One vector rollout, one connection. Blocking: the agent loop calls it through ``asyncio.to_thread``."""

    def __init__(self, socket: str, *, learn: bool) -> None:
        self.connection = Client(socket, family="AF_UNIX", authkey=b"rssm")
        self._call("begin", learn)

    def _call(self, *message: Any) -> Any:
        self.connection.send(message)
        reply = self.connection.recv()
        if isinstance(reply, Exception):
            raise RuntimeError("rssm_service failed") from reply
        return reply

    def first(self, frames: list[np.ndarray]) -> np.ndarray:
        """h_0 = 0, z_0 = posterior mode of the first frames. Returns [lanes, h + z]."""
        return self._call("first", np.stack(frames))

    def step(self, lanes: list[int], actions: list[str], rewards: list[float], ends: list[bool],
             frames: list[np.ndarray]) -> np.ndarray:
        """h_{t+1} of the lanes that acted, z_{t+1} from their new frames. Returns the whole [lanes, h + z]."""
        return self._call("step", lanes, actions, rewards, ends, np.stack(frames))

    def close(self) -> None:
        self.connection.close()
