"""Portable recordings of real Crafter steps and counterfactual plans."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class EpisodeRecording:
    def __init__(self, directory, seed, horizon, limit):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.info = dict(schema=1, seed=seed, horizon=horizon, limit=limit, frame_size=63, steps=0)
        write_json(self.path / "progress.json", self.info)

    def capture(self, record, start, dream, reality, rewards, actions, death_steps):
        self.record = record
        self.size = self.info["frame_size"]
        candidates, horizon = dream.shape[:2]
        self.sheet = Image.new("RGB", ((horizon + 1) * self.size, (candidates * 2 + 1) * self.size))
        self.paste(start, 0, 0)
        scores = np.asarray(record["dream"])
        record["advantage"] = (scores - scores.mean()).tolist()
        record["dream_step_reward"] = rewards[0].tolist()
        record["real_step_reward"] = rewards[1].tolist()
        record["death_step"] = [int(step) + 1 if step >= 0 else None for step in death_steps]
        record["full_actions"] = actions
        record["real_length"] = [len(frames) for frames in reality]
        for plan in range(candidates):
            for k, frame in enumerate(dream[plan, :record["plan_len"][plan]]):
                self.paste(frame, k + 1, 2 * plan + 1)
            for k, frame in enumerate(reality[plan]):
                self.paste(frame, k + 1, 2 * plan + 2)

    def paste(self, frame, column, row):
        self.sheet.paste(Image.fromarray(frame), (column * self.size, row * self.size))

    def finish_step(self, chosen, action, next_frame, reward, achievements, vitals, end):
        record = self.record
        record.update(chosen=chosen, action=action, reward=reward, achievements=sorted(achievements),
                      vitals=list(map(float, vitals)), episode_end=end)
        self.paste(next_frame, 1, 0)
        name = f"{record['t']:04d}"
        self.sheet.save(self.path / f"{name}.png", compress_level=3)
        write_json(self.path / f"{name}.json", record)
        self.info.update(steps=record["t"] + 1, end=end)
        write_json(self.path / "progress.json", self.info)
        if end:
            write_json(self.path / "manifest.json", self.info)
