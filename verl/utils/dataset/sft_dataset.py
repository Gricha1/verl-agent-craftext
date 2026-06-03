# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length
        self.data_config = config

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls.iloc[0] if isinstance(ls, pandas.core.series.Series) else ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                def extract_key(x):
                    item = series_to_item(x)
                    # If item is a dict, extract the key; otherwise return as-is
                    if isinstance(item, dict):
                        return item[key]
                    else:
                        # If prompt_dict_keys is set but data is not a dict, return the item itself
                        # This handles cases where the data structure doesn't match the config
                        return item
                self.prompts = self.prompts.apply(extract_key, axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                def extract_key(x):
                    item = series_to_item(x)
                    # If item is a dict, extract the key; otherwise return as-is
                    if isinstance(item, dict):
                        return item[key]
                    else:
                        # If response_dict_keys is set but data is not a dict, return the item itself
                        # This handles cases where the data structure doesn't match the config
                        return item
                self.responses = self.responses.apply(extract_key, axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

        self.horizons = None
        if "horizon" in self.dataframe.columns:
            self.horizons = [int(x) for x in self.dataframe["horizon"].tolist()]
        else:
            self.horizons = [self._infer_horizon_from_response(r) for r in self.responses]

        self.states = self._load_optional_column("state")
        self.states_after = self._load_optional_column("state_after")
        self.action_tokens = self._load_optional_column("action_token")
        self.has_inverse_columns = (
            self.states is not None
            and self.states_after is not None
            and self.action_tokens is not None
            and any(str(s).strip() for s in self.states_after)
        )

        self.future_rewards_raw = self._load_optional_column("future_rewards")
        self.future_actions_raw = self._load_optional_column("future_actions")
        self.instructions = self._load_optional_column("instruction")
        self._planning_wm_enable = self._config_bool(
            getattr(self.data_config, "planning_wm_enable", False)
        )
        self._planning_wm_horizon = max(
            1, int(getattr(self.data_config, "planning_wm_horizon", 1) or 1)
        )
        self.has_planning_columns = (
            self._planning_wm_enable
            and self.states is not None
            and self.future_rewards_raw is not None
            and self.future_actions_raw is not None
            and self.horizons is not None
        )

    @staticmethod
    def _config_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _load_optional_column(self, name: str):
        if name not in self.dataframe.columns:
            return None
        series = self.dataframe[name]
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        return [str(x or "") for x in series.tolist()]

    @staticmethod
    def _infer_horizon_from_response(response: str) -> int:
        """Reward targets are concatenated i/j/k/l chars without spaces."""
        try:
            from agent_system.environments.env_package.caged_craftext.reward_tokens import (
                is_reward_token_response,
            )

            raw = str(response or "").strip()
            if is_reward_token_response(raw):
                return max(1, len(raw))
        except Exception:
            pass
        reward_chars = frozenset("ijkl")
        n = 0
        for ch in str(response or "").strip():
            if ch in reward_chars:
                n += 1
            else:
                break
        return max(1, n)

    def _tokenize_response(self, tokenizer, response: str, *, planning_horizon: int | None = None):
        """Tokenize SFT response; reward WM and planning use per-letter ids."""
        if planning_horizon is not None:
            try:
                from agent_system.environments.env_package.caged_craftext.action_tokens import (
                    is_action_token_sequence,
                    tokenize_action_response_ids,
                )

                if is_action_token_sequence(response, horizon=int(planning_horizon)):
                    ids = tokenize_action_response_ids(
                        tokenizer,
                        response,
                        add_eos=True,
                        expected_len=int(planning_horizon),
                    )
                    response_ids = torch.tensor(ids, dtype=torch.long)
                    response_attention_mask = torch.ones_like(response_ids)
                    return response_ids, response_attention_mask
            except Exception:
                pass

        try:
            from agent_system.environments.env_package.caged_craftext.reward_tokens import (
                is_reward_token_response,
                tokenize_reward_response_ids,
            )

            if is_reward_token_response(response):
                ids = tokenize_reward_response_ids(tokenizer, response, add_eos=True)
                response_ids = torch.tensor(ids, dtype=torch.long)
                response_attention_mask = torch.ones_like(response_ids)
                return response_ids, response_attention_mask
        except Exception:
            pass

        response_chat_str = str(response) + tokenizer.eos_token
        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        return response_ids_output["input_ids"][0], response_ids_output["attention_mask"][0]

    def _build_sft_tensors(self, prompt: str, response: str, *, planning_horizon: int | None = None):
        """Tokenize one (prompt, response) pair with prompt masked in loss."""
        tokenizer = self.tokenizer
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids, response_attention_mask = self._tokenize_response(
            tokenizer, response, planning_horizon=planning_horizon
        )

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(
                size=(self.max_length - sequence_length,), dtype=input_ids.dtype
            ) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,), dtype=attention_mask.dtype
            )
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _inverse_sample_for_index(self, item: int):
        """Build (prompt, response) for inverse-action WM from reward row metadata."""
        if not self.has_inverse_columns:
            return None, None, False
        try:
            from agent_system.environments.prompts.world_model_inverse_action import (
                format_inverse_action_prompt,
                is_unchanged_transition,
            )
        except Exception:
            return None, None, False

        state_before = str(self.states[item] or "").strip()
        state_after = str(self.states_after[item] or "").strip()
        action = str(self.action_tokens[item] or "").strip().split()[0]
        if not state_before or not state_after or not action:
            return None, None, False
        if is_unchanged_transition(state_before, state_after):
            return None, None, False
        return format_inverse_action_prompt(state_before, state_after), action, True

    def _planning_sample_for_index(self, item: int):
        """Return-conditioned planning: (s_t, R̂) -> a_t..a_{t+H-1} for horizon=H rows only."""
        if not self.has_planning_columns:
            return None, None, False, 0
        try:
            from agent_system.environments.prompts.world_model_planning import (
                cumulative_return_from_rewards,
                format_planning_prompt,
                format_planning_target,
                parse_future_json_list,
                planning_row_valid,
            )
        except Exception:
            return None, None, False, 0

        h_plan = int(self._planning_wm_horizon)
        horizon = int(self.horizons[item])
        state = str(self.states[item] or "").strip()
        if not state:
            return None, None, False, 0

        future_acts = parse_future_json_list(self.future_actions_raw[item])
        future_rews = parse_future_json_list(self.future_rewards_raw[item])
        if not planning_row_valid(
            horizon=horizon,
            planning_horizon=h_plan,
            future_actions=future_acts,
            future_rewards=future_rews,
        ):
            return None, None, False, 0

        target_return = cumulative_return_from_rewards(future_rews)
        task = ""
        if self.instructions is not None:
            task = str(self.instructions[item] or "").strip()
        prompt = format_planning_prompt(
            state,
            target_return=target_return,
            task=task,
            horizon=h_plan,
        )
        response = format_planning_target(future_acts)
        return prompt, response, True, int(target_return)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response = self.responses[item]

        out = self._build_sft_tensors(prompt, response)

        if self.horizons is not None:
            horizon = int(self.horizons[item])
        else:
            horizon = self._infer_horizon_from_response(response)
        out["horizon"] = torch.tensor(horizon, dtype=torch.long)

        inv_prompt, inv_response, inv_valid = self._inverse_sample_for_index(item)
        if inv_valid and inv_prompt is not None and inv_response is not None:
            inv = self._build_sft_tensors(inv_prompt, inv_response)
        else:
            inv = {
                "input_ids": out["input_ids"].clone(),
                "attention_mask": out["attention_mask"].clone(),
                "position_ids": out["position_ids"].clone(),
                "loss_mask": torch.zeros_like(out["loss_mask"]),
            }

        out["inverse_input_ids"] = inv["input_ids"]
        out["inverse_attention_mask"] = inv["attention_mask"]
        out["inverse_position_ids"] = inv["position_ids"]
        out["inverse_loss_mask"] = inv["loss_mask"]
        out["inverse_valid"] = torch.tensor(1 if inv_valid else 0, dtype=torch.long)

        plan_prompt, plan_response, plan_valid, plan_return = self._planning_sample_for_index(item)
        if plan_valid and plan_prompt is not None and plan_response is not None:
            plan = self._build_sft_tensors(
                plan_prompt,
                plan_response,
                planning_horizon=int(self._planning_wm_horizon),
            )
        else:
            plan = {
                "input_ids": out["input_ids"].clone(),
                "attention_mask": out["attention_mask"].clone(),
                "position_ids": out["position_ids"].clone(),
                "loss_mask": torch.zeros_like(out["loss_mask"]),
            }
        out["planning_input_ids"] = plan["input_ids"]
        out["planning_attention_mask"] = plan["attention_mask"]
        out["planning_position_ids"] = plan["position_ids"]
        out["planning_loss_mask"] = plan["loss_mask"]
        out["planning_valid"] = torch.tensor(1 if plan_valid else 0, dtype=torch.long)
        out["planning_target_return"] = torch.tensor(int(plan_return), dtype=torch.long)
        return out
