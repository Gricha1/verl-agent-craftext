# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from verl.utils.reward_score.gsm8k import compute_score


class Gsm8kVecEnv:
    """
    Batched single-turn GSM8K environment.
    Each slot scores one model response against ground_truth from dataset env_kwargs.
    """

    def __init__(
        self,
        env_num: int = 1,
        group_n: int = 1,
        *,
        score_method: str = "strict",
        format_score: float = 0.0,
        correct_score: float = 1.0,
    ) -> None:
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.batch_size = self.env_num * self.group_n
        self.score_method = str(score_method)
        self.format_score = float(format_score)
        self.correct_score = float(correct_score)
        self._states: List[Dict[str, Any]] = [{} for _ in range(self.batch_size)]

    def reset(self, kwargs: Optional[List[Dict[str, Any]]] = None):
        if kwargs is None:
            raise ValueError("Gsm8kVecEnv.reset requires env_kwargs from the GSM8K parquet batch")
        if len(kwargs) > self.batch_size:
            raise ValueError(
                f"Got {len(kwargs)} env_kwargs, but env batch_size={self.batch_size}"
            )

        pad_n = self.batch_size - len(kwargs)
        padded_kwargs = list(kwargs) + [
            {"question": "", "ground_truth": "", "data_source": "openai/gsm8k"}
        ] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []
        for i, (kw, keep) in enumerate(zip(padded_kwargs, valid_mask)):
            if not keep:
                self._states[i] = {}
                obs_list.append("")
                info_list.append({"data_source": "openai/gsm8k", "active": False})
                continue

            question = str(kw.get("question") or kw.get("instruction") or "").strip()
            ground_truth = str(kw.get("ground_truth") or "").strip()
            data_source = str(kw.get("data_source") or "openai/gsm8k")
            self._states[i] = {
                "question": question,
                "ground_truth": ground_truth,
                "data_source": data_source,
            }
            obs_list.append(question)
            info_list.append(
                {
                    "instruction": question,
                    "text_render": "(awaiting solution)",
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "active": True,
                }
            )

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [inf for inf, keep in zip(info_list, valid_mask) if keep]
        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(
                f"Got {len(actions)} actions, but env batch_size={self.batch_size}"
            )

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        obs_list: List[str] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for i, (action, keep) in enumerate(zip(padded_actions, valid_mask)):
            if not keep:
                obs_list.append("")
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({"data_source": "openai/gsm8k", "active": False})
                continue

            state = self._states[i]
            ground_truth = state.get("ground_truth", "")
            data_source = state.get("data_source", "openai/gsm8k")
            score = float(
                compute_score(
                    solution_str=action,
                    ground_truth=ground_truth,
                    method=self.score_method,
                    format_score=self.format_score,
                    score=self.correct_score,
                )
            )
            obs_list.append("")
            reward_list.append(score)
            done_list.append(True)
            info_list.append(
                {
                    "instruction": state.get("question", ""),
                    "text_render": "(answered)",
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "won": bool(score >= self.correct_score),
                    "active": True,
                }
            )

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [inf for inf, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self) -> None:
        return


def build_gsm8k_envs(
    env_num: int = 1,
    group_n: int = 1,
    *,
    score_method: str = "strict",
    format_score: float = 0.0,
    correct_score: float = 1.0,
):
    return Gsm8kVecEnv(
        env_num=env_num,
        group_n=group_n,
        score_method=score_method,
        format_score=format_score,
        correct_score=correct_score,
    )
