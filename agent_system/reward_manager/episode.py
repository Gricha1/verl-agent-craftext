# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
import torch
import numpy as np

from verl.utils.actor_value_token import compute_remaining_return_scalars


class EpisodeRewardManager:
    """The reward manager.

    Token score modes (mutually resolved as below):

    1. ``use_episode_return_as_token_reward=True``:
       full-episode return ``episode_rewards`` on every step's last response token.

    2. ``use_remaining_return_as_token_reward=True`` (and episode-return False):
       discounted return-to-go ``G_t^gamma`` on last response token of step t,
       with ``remaining_return_gamma`` (default 1.0 = undiscounted suffix sum).

    3. both False:
       per-step env reward ``r_t`` (FINAL dual / step-TD style).

    Important: do NOT set algorithm.gamma alone and expect token G_t to change —
    token remaining returns use ``remaining_return_gamma`` (or the value passed here).
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length=False,
        use_episode_return_as_token_reward=True,
        use_remaining_return_as_token_reward=False,
        remaining_return_gamma=1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        self.use_episode_return_as_token_reward = bool(use_episode_return_as_token_reward)
        self.use_remaining_return_as_token_reward = bool(use_remaining_return_as_token_reward)
        self.remaining_return_gamma = float(remaining_return_gamma)
        if self.use_episode_return_as_token_reward and self.use_remaining_return_as_token_reward:
            raise ValueError(
                "Set only one of use_episode_return_as_token_reward / "
                "use_remaining_return_as_token_reward"
            )

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        n = len(data)
        scores = np.zeros(n, dtype=np.float64)

        if self.use_remaining_return_as_token_reward:
            step_rewards_raw = []
            traj_uids = []
            step_idxs = []
            for i in range(n):
                item = data[i]
                sr = item.non_tensor_batch.get("rewards")
                if sr is None:
                    raise ValueError(
                        "use_remaining_return_as_token_reward requires non_tensor_batch['rewards']"
                    )
                step_rewards_raw.append(float(np.asarray(sr, dtype=np.float64).reshape(())))
                traj_uids.append(str(item.non_tensor_batch["traj_uid"]))
                esi = item.non_tensor_batch.get("episode_step_idx")
                step_idxs.append(None if esi is None else int(np.asarray(esi).reshape(())))
            if any(s is None for s in step_idxs):
                episode_step_idx = None
            else:
                episode_step_idx = step_idxs
            scores = compute_remaining_return_scalars(
                traj_uids,
                step_rewards_raw,
                episode_step_idx=episode_step_idx,
                gamma=self.remaining_return_gamma,
            )
        else:
            for i in range(n):
                item = data[i]
                episode_rewards = item.non_tensor_batch["episode_rewards"]
                episode_lengths = item.non_tensor_batch["episode_lengths"]
                if self.use_episode_return_as_token_reward:
                    if self.normalize_by_length:
                        scores[i] = float(episode_rewards) / float(max(episode_lengths, 1))
                    else:
                        scores[i] = float(np.asarray(episode_rewards, dtype=np.float64).reshape(()))
                else:
                    step_reward = item.non_tensor_batch.get("rewards")
                    if step_reward is not None:
                        scores[i] = float(np.asarray(step_reward, dtype=np.float64).reshape(()))
                    elif self.normalize_by_length:
                        scores[i] = float(episode_rewards) / float(max(episode_lengths, 1))
                    else:
                        scores[i] = float(np.asarray(episode_rewards, dtype=np.float64).reshape(()))

        already_print_data_sources = {}
        for i in range(n):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            data_source = data_item.non_tensor_batch["data_source"]
            score = float(scores[i])
            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
