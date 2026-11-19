"""Dreamer 4 (4ku/dreamer4 3489e13) dataset adapter for our Crafter episodes; copied into the dreamer4 checkout as
dreamer4/data/wood.py (it registers there as WoodEpisodeDataset).

Data dir: frames.npy uint8 [N, 63 or 64, 63 or 64, 3] (64 x 64 Crafter frames are cut to the 63 x 63 view), actions.npy
(action taken in frame t), rewards.npy (reward of the t -> t+1 transition), terminals.npy, wood_manifest.json with starts,
lengths, n_actions and optionally:
  tasks          per-episode task index -> episode_meta()["task"], the task indicator of the phase-2 agent tokens
  bc             per-episode 0/1 -> bc_weight(), which episodes behaviour cloning may imitate (default all)
  terminal_rule  "abs_reward": terminal iff |arriving reward| >= 0.5 (reward v2: +1 success, -1 death);
                 "reward": a frame is terminal iff its arriving reward is >= 0.5 (the wood tasks: the only +1 is the task
                 success, which ends the episode); otherwise no frame is terminal from the reward (Qwen Crafter data,
                 where deaths pay nothing distinct)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dreamer4.data.base import EpisodeVideoDataset


class WoodEpisodeDataset(EpisodeVideoDataset):

    @classmethod
    def recognizes(cls, root: Path) -> bool:
        return (Path(root) / "wood_manifest.json").exists()

    @classmethod
    def from_path(cls, root: Path, *, proprio: str = "none", actions: bool = False, **kwargs) -> "WoodEpisodeDataset":
        return cls(root, actions)

    def __init__(self, root, actions=False):
        self.root = Path(root)
        self.with_actions = actions
        self.manifest = json.loads((self.root / "wood_manifest.json").read_text())
        self.starts, self.lengths, self.n_actions = self.manifest["starts"], self.manifest["lengths"], self.manifest["n_actions"]
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.rewards = np.load(self.root / "rewards.npy", mmap_mode="r")
        self.terminals = np.load(self.root / "terminals.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.lengths)

    def episode_frames(self, i: int) -> int:
        return self.lengths[i]

    @property
    def action_dim(self):
        return self.n_actions if self.with_actions else None

    def episode_meta(self, i: int):
        tasks = self.manifest.get("tasks")
        return {"task": int(tasks[i])} if tasks is not None else {}

    def bc_weight(self, i: int) -> float:
        bc = self.manifest.get("bc")
        return float(bc[i]) if bc is not None else 1.0

    def continues_from_reward(self, reward_pred):
        if self.manifest.get("terminal_rule") == "reward":
            return (reward_pred < 0.5).float()
        if self.manifest.get("terminal_rule") == "abs_reward":   # reward v2: +1 success, -1 death, both end the episode
            return (torch.as_tensor(reward_pred).abs() < 0.5).float()
        return torch.ones_like(torch.as_tensor(reward_pred, dtype=torch.float32))

    def _load_clip(self, i: int, start: int, length: int) -> dict:
        first = self.starts[i] + start
        clip = {"video": np.array(self.frames[first:first + length, :63, :63])}   # 64 x 64 Crafter frames: the 63 x 63 view
        if self.with_actions:
            moves = np.array(self.actions[first:first + length - 1])
            clip["actions"] = np.eye(self.n_actions, dtype=np.float32)[moves]
            clip["rewards"] = np.array(self.rewards[first:first + length - 1], dtype=np.float32)
            clip["terminals"] = np.array(self.terminals[first:first + length - 1], dtype=bool)
        return clip
