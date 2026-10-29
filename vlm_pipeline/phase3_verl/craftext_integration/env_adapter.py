"""Thin adapter around the existing JAX-vmapped CrafText implementation.

This module deliberately does not create Ray actors.  A single
``CagedCraftextOptimisticVecEnv`` owns all requested lanes and performs reset
and step with JAX ``vmap``.  The legacy package contains the environment
implementation itself; keeping it outside the modern veRL checkout avoids
forking Craftax/CrafText while the PPO plumbing is migrated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .actions import ParsedAction, parse_reasoning_action


def _legacy_root() -> Path:
    """Return the checkout containing ``agent_system``.

    Deployment supplies ``CRAFTEXT_LEGACY_ROOT`` explicitly.  The fallback is
    convenient for the monorepo development layout and is intentionally
    validated rather than silently importing an unrelated installed package.
    """

    configured = os.environ.get("CRAFTEXT_LEGACY_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).resolve().parents[2] / "wm_actoropt_envval")
    for candidate in candidates:
        if (candidate / "agent_system" / "environments" / "env_package").is_dir():
            return candidate
    raise RuntimeError(
        "CrafText environment source not found. Set CRAFTEXT_LEGACY_ROOT to the "
        "checkout that contains agent_system/environments/env_package."
    )


class CraftextJaxVectorEnv:
    """Text-facing, batched CrafText adapter for the modern agent loop.

    ``num_envs`` may be 1 for a correctness smoke test and 64 for the normal
    vectorised path.  No environment lane has a dedicated Ray process.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        max_steps: int,
        reset_ratio: int | None = None,
        config_name: str = "debug_square_8x8",
        use_pixel_obs: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        # The legacy DistilBERT encoder hard-codes ``cache_dir='.'``.  In a
        # modern veRL run the checkout is deliberately read-only, so make the
        # worker's current directory an explicitly writable, per-run cache.
        # This is process-local in the vector agent-loop Ray worker.
        runtime_cache = os.environ.get("CRAFTEXT_RUNTIME_CACHE_DIR")
        if runtime_cache:
            Path(runtime_cache).mkdir(parents=True, exist_ok=True)
            os.chdir(runtime_cache)
        root = _legacy_root()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        from agent_system.environments.env_package.caged_craftext import envs as legacy_envs  # noqa: PLC0415
        from agent_system.environments.env_package.caged_craftext.envs import (  # noqa: PLC0415
            CagedCraftextOptimisticVecEnv,
        )

        if use_pixel_obs:
            # The legacy renderer draws 64-px blocks (576x576) for GIFs.  The
            # visual SLM was trained on Craftax-Classic-Pixels observations,
            # i.e. the same renderer at the 7-px agent block size (63x63 with
            # the inventory band), so render frames at that resolution.
            from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_AGENT  # noqa: PLC0415

            def _render_agent_frame(state):
                state = legacy_envs._strip_all_mobs_for_state(state)
                return np.asarray(legacy_envs.render_classic(state, block_pixel_size=BLOCK_PIXEL_SIZE_AGENT))

            legacy_envs._render_craftax_frame = _render_agent_frame

        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.use_pixel_obs = bool(use_pixel_obs)
        self._env = CagedCraftextOptimisticVecEnv(
            seed=int(seed),
            env_num=self.num_envs,
            reset_ratio=reset_ratio,
            is_train=True,
            env_kwargs={
                "config_name": str(config_name),
                "observation_type": "ascii",
                "use_pixel_obs": self.use_pixel_obs,
                # Explicitly prohibit the old one-Ray-actor-per-text-render mode.
                "use_ray_text_render_workers": False,
            },
        )

    @staticmethod
    def _observations(infos: list[dict[str, Any]]) -> list[str]:
        return [str(info.get("text_render", "The world is empty.")) for info in infos]

    @staticmethod
    def _frames(observations: list[Any]) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for observation in observations:
            frame = np.asarray(observation)
            if frame.shape != (63, 63, 3):
                raise RuntimeError(f"CrafText pixel renderer returned {frame.shape}, expected (63, 63, 3)")
            if np.issubdtype(frame.dtype, np.floating):
                # The JAX renderer returns float 0..255; round rather than truncate.
                frame = np.clip(np.rint(frame), 0, 255)
            frames.append(frame.astype(np.uint8, copy=False))
        return frames

    def reset(self):
        observations, infos = self._env.reset()
        text = self._observations(infos)
        if self.use_pixel_obs:
            return text, self._frames(observations), infos
        return text, infos

    def step(
        self, actions: list[ParsedAction]
    ):
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        observations, rewards, dones, infos = self._env.step([action.action_id for action in actions])
        text = self._observations(infos)
        if self.use_pixel_obs:
            return text, self._frames(observations), rewards, dones, infos
        return text, rewards, dones, infos

    def parse_and_step(
        self, responses: list[str]
    ):
        if len(responses) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} responses, got {len(responses)}")
        actions = [parse_reasoning_action(response) for response in responses]
        stepped = self.step(actions)
        if self.use_pixel_obs:
            observations, frames, rewards, dones, infos = stepped
            return actions, observations, frames, rewards, dones, infos
        observations, rewards, dones, infos = stepped
        return actions, observations, rewards, dones, infos

    def close(self) -> None:
        self._env.close()
