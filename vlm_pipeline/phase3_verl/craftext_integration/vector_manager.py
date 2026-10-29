"""TransferQueue manager that rolls a whole PPO batch through one JAX env.

Modern veRL's generic V1 worker creates an agent-loop object per prompt.  That
is excellent for web tools, but wrong for CrafText: it would initialise one
environment/encoder per prompt.  This manager owns one vectorised CrafText
environment in one Ray worker and advances all lanes together.  It publishes
one PPO item per environment action, exactly like the legacy rollout loop:
the next CrafText observation is a new prompt, never part of the old response.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopWorker, DictConfigWrap
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from .actions import ParsedAction
from .agent_loop import CraftextAgentLoopOutput
from .env_adapter import CraftextJaxVectorEnv
from .slm_vision.prompt import VISUAL_SLM_ACTION_HISTORY, parse_slm_action, vision_image_only_prompt


class _TokenCodec(AgentLoopBase):
    """Uses veRL's continuous-token implementation without owning an env."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):  # pragma: no cover - never dispatched
        raise NotImplementedError("_TokenCodec is used only by CraftextVectorAgentLoopWorker")


def _value_at(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, NonTensorStack):
        return value[index].data
    if isinstance(value, NonTensorData):
        return value.data
    raise TypeError(f"Unsupported V1 prompt field type: {type(value)}")


class CraftextVectorAgentLoopWorker(AgentLoopWorker):
    """One Ray actor, one JAX-vmapped env, N logical CrafText trajectories."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks: set[asyncio.Task[Any]] = set()
        cfg = self.config.get("craftext", {})
        self.num_envs = int(cfg.get("num_envs", 4))
        self.max_env_steps = int(cfg.get("max_env_steps", 3))
        self.per_turn_max_tokens = int(cfg.get("per_turn_max_tokens", 96))
        self.seed = int(cfg.get("seed", 0))
        self.validation_seed = int(cfg.get("validation_seed", self.seed + 1))
        self.env_config_name = str(cfg.get("env_config_name", "debug_square_8x8"))
        self.reset_ratio = int(cfg.get("reset_ratio", 1))
        self.action_history_length = int(cfg.get("action_history_length", 50))
        self.reasoning_history_length = int(cfg.get("reasoning_history_length", 3))
        # Must equal the CE prompt protocol of the checkpoint: 8 for
        # ``vision_image_actions8_v1``, 0 for ``vision_image_only_v1`` (no
        # history line at all).
        self.visual_prompt_history = int(cfg.get("visual_prompt_history", VISUAL_SLM_ACTION_HISTORY))
        if self.visual_prompt_history not in (0, VISUAL_SLM_ACTION_HISTORY):
            raise ValueError(f"visual_prompt_history must be 0 or {VISUAL_SLM_ACTION_HISTORY}")
        self.reward_mode = str(cfg.get("reward_mode", "immediate"))
        self.remaining_return_gamma = float(cfg.get("remaining_return_gamma", 1.0))
        # Mirrors legacy veRL's actor invalid-action penalty.  It is applied
        # to this action's final-token reward after any return-to-go shaping,
        # so it does not leak backwards into earlier actions in the episode.
        self.use_invalid_action_penalty = bool(cfg.get("use_invalid_action_penalty", False))
        self.invalid_action_penalty_coef = float(cfg.get("invalid_action_penalty_coef", 0.0))
        self.validation_trajectory_dir = cfg.get("validation_trajectory_dir")
        self.validation_trajectory_limit = int(cfg.get("validation_trajectory_limit", 10))
        self.codec = _TokenCodec(
            trainer_config=DictConfigWrap(self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            hf_model_type=self.hf_model_type,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        self.visual_marker_ids = [
            int(self.tokenizer.convert_tokens_to_ids(f"<|crafter_visual_{index}|>")) for index in range(4)
        ]
        if any(token_id < 0 or token_id == self.tokenizer.unk_token_id for token_id in self.visual_marker_ids):
            raise RuntimeError("Visual SLM tokenizer is missing the four reserved CrafText marker tokens")

    async def generate_sequences(self, batch: TensorDict) -> None:
        validate = bool(batch["validate"]) if "validate" in batch else False
        batch.pop("validate", None)
        expected_batch_size = int(self.config.data.val_batch_size if validate else self.config.data.train_batch_size)
        if len(batch) != expected_batch_size:
            raise ValueError(
                f"CrafText vector worker expects {expected_batch_size} prompts, got {len(batch)}"
            )
        task = asyncio.create_task(self._run_vector_batch(batch, validate=validate))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def _sampling_params(self, *, validate: bool) -> dict[str, Any]:
        rollout = self.config.actor_rollout_ref.rollout
        params = {
            "temperature": rollout.temperature,
            "top_p": rollout.top_p,
            "top_k": rollout.top_k,
            "repetition_penalty": 1.0,
            "logprobs": rollout.calculate_log_probs,
        }
        if validate:
            params.update(
                temperature=rollout.val_kwargs.temperature,
                top_p=rollout.val_kwargs.top_p,
                top_k=rollout.val_kwargs.top_k,
            )
        return params

    async def _run_vector_batch(self, batch: TensorDict, *, validate: bool) -> None:
        num_envs = len(batch)
        prompts = [{key: _value_at(value, lane) for key, value in batch.items()} for lane in range(len(batch))]
        uids = [str(prompt["uid"]) for prompt in prompts]
        partition = "val" if validate else "train"
        for uid in uids:
            await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "running"})

        env = CraftextJaxVectorEnv(
            num_envs=num_envs,
            # A new vector environment is created for every validation call;
            # keeping this seed fixed makes all lanes deterministic across PPO
            # iterations and distinct from the training vector environment.
            seed=self.validation_seed if validate else self.seed,
            max_steps=self.max_env_steps,
            reset_ratio=self.reset_ratio,
            config_name=self.env_config_name,
            use_pixel_obs=True,
        )
        try:
            observations, frames, infos = env.reset()
            tasks = [str(info.get("instruction", "Complete the CrafText task safely.")) for info in infos]
            action_histories: list[list[str]] = [[] for _ in range(num_envs)]
            reasoning_histories: list[list[str]] = [[] for _ in range(num_envs)]
            transitions: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
            outputs_by_lane: list[list[CraftextAgentLoopOutput]] = [[] for _ in range(num_envs)]
            active = [True] * num_envs
            sampling_params = self._sampling_params(validate=validate)

            for turn in range(self.max_env_steps):
                generated: dict[int, Any] = {}
                generation_tasks = []
                lanes = []
                prompt_ids: dict[int, list[int]] = {}
                for lane, is_active in enumerate(active):
                    if not is_active:
                        active[lane] = False
                        continue
                    # The visual SLM has no chat template: its CE prompt is plain
                    # text, tokenised as-is, after the four visual marker ids.
                    prompt_text = vision_image_only_prompt(
                        tasks[lane],
                        action_histories[lane][-self.visual_prompt_history :] if self.visual_prompt_history else None,
                    )
                    prompt_ids[lane] = self.visual_marker_ids + self.tokenizer.encode(
                        prompt_text, add_special_tokens=False
                    )
                    if len(prompt_ids[lane]) > self.codec.rollout_config.prompt_length:
                        raise RuntimeError(
                            f"Visual SLM prompt has {len(prompt_ids[lane])} tokens, above rollout.prompt_length"
                        )
                    params = deepcopy(sampling_params)
                    # ``response_length`` is a per-action bound, as in r8.
                    params["max_tokens"] = min(self.per_turn_max_tokens, self.codec.rollout_config.response_length)
                    lanes.append(lane)
                    generation_tasks.append(
                        self.llm_client.generate(
                            request_id=f"craftext-{uuid4().hex}",
                            prompt_ids=prompt_ids[lane],
                            sampling_params=params,
                            image_data=[frames[lane]],
                            priority=lane,
                        )
                    )
                if not lanes:
                    break
                for lane, result in zip(lanes, await asyncio.gather(*generation_tasks), strict=True):
                    generated[lane] = result

                actions: list[ParsedAction] = []
                decoded: list[str] = [""] * num_envs
                turn_outputs: dict[int, tuple[list[int], list[int], list[float] | None]] = {}
                for lane in range(num_envs):
                    if lane not in generated:
                        actions.append(ParsedAction(action_id=0, action_name="NOOP", reasoning="", valid=False))
                        continue
                    result = generated[lane]
                    # One template-free assistant turn: every generated token
                    # (including a closing EOS) is trainable, as in CE.
                    response_ids = list(result.token_ids)
                    mask = [1] * len(response_ids)
                    logprobs = list(result.log_probs) if result.log_probs else None
                    if len(mask) > self.codec.rollout_config.response_length:
                        raise RuntimeError("CrafText action response exceeded rollout.response_length")
                    turn_outputs[lane] = (response_ids, mask, logprobs)
                    decoded[lane] = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
                    actions.append(parse_slm_action(decoded[lane]))

                next_observations, next_frames, rewards, dones, next_infos = env.step(actions)
                for lane in lanes:
                    reward, done = float(rewards[lane]), bool(dones[lane])
                    transition = {
                        "turn": turn,
                        "task": tasks[lane],
                        "observation": observations[lane],
                        "response": decoded[lane],
                        "executed_action": actions[lane].action_name,
                        "valid_action": actions[lane].valid,
                        "reward": reward,
                        "done": done,
                        "success": bool(next_infos[lane].get("instruction_done", False) or next_infos[lane].get("won", False)),
                        "next_observation": next_observations[lane],
                    }
                    transitions[lane].append(transition)
                    response_ids, response_mask, response_logprobs = turn_outputs[lane]
                    trainable = [index for index, value in enumerate(response_mask) if value]
                    if not trainable:
                        raise RuntimeError("CrafText action produced no trainable assistant tokens")
                    outputs_by_lane[lane].append(
                        CraftextAgentLoopOutput(
                            prompt_ids=prompt_ids[lane],
                            response_ids=response_ids,
                            response_mask=response_mask,
                            response_logprobs=response_logprobs,
                            reward_score=reward,
                            num_turns=1,
                            metrics=AgentLoopMetrics(),
                            extra_fields={
                                "craftext_turn_end_indices": [trainable[-1]],
                                "craftext_turn_rewards": [reward],
                                "craftext_transitions": [transition],
                                "craftext_trajectory_id": uids[lane],
                                "craftext_episode_step_idx": turn,
                                "craftext_generated_tokens": int(sum(response_mask)),
                                "craftext_action_valid": float(actions[lane].valid),
                                "craftext_action_invalid": float(not actions[lane].valid),
                                "craftext_invalid_action_penalty": float(
                                    self.invalid_action_penalty_coef
                                    if self.use_invalid_action_penalty and not actions[lane].valid
                                    else 0.0
                                ),
                                "craftext_action_noop": float(actions[lane].action_name == "NOOP"),
                                "craftext_action_repeated": float(
                                    bool(action_histories[lane])
                                    and actions[lane].action_name == action_histories[lane][-1]
                                ),
                                # Stored separately from text telemetry and moved to
                                # ``multi_modal_inputs`` at the TQ boundary below.
                                "craftext_pixel_values": torch.from_numpy(
                                    frames[lane].transpose(2, 0, 1).copy()
                                ).unsqueeze(0).to(dtype=torch.float32).div_(255.0),
                            },
                        )
                    )
                    action_histories[lane].append(actions[lane].action_name)
                    if self.action_history_length > 0:
                        action_histories[lane] = action_histories[lane][-self.action_history_length :]
                    if actions[lane].reasoning:
                        reasoning_histories[lane].append(actions[lane].reasoning[:200])
                        if self.reasoning_history_length > 0:
                            reasoning_histories[lane] = reasoning_histories[lane][-
                                self.reasoning_history_length :
                            ]
                    if done or turn + 1 >= self.max_env_steps:
                        active[lane] = False
                observations, frames, infos = next_observations, next_frames, next_infos

            for lane in range(num_envs):
                if self.reward_mode == "remaining_return":
                    remaining = 0.0
                    for output in reversed(outputs_by_lane[lane]):
                        reward = float(output.extra_fields["craftext_turn_rewards"][0])
                        remaining = reward + self.remaining_return_gamma * remaining
                        output.extra_fields["craftext_turn_rewards"] = [remaining]
                elif self.reward_mode != "immediate":
                    raise ValueError(f"Unsupported CrafText reward_mode: {self.reward_mode}")
                # Legacy's apply_invalid_action_penalty subtracts this amount
                # from the final response token / step reward after reward
                # construction.  Do it after return-to-go shaping so it does
                # not leak backwards into earlier actions in the episode.
                for output in outputs_by_lane[lane]:
                    penalty = float(output.extra_fields["craftext_invalid_action_penalty"])
                    output.extra_fields["craftext_turn_rewards"][0] -= penalty
                episode_return = float(sum(item["reward"] for item in transitions[lane]))
                episode_steps = len(transitions[lane])
                episode_success = float(any(item["success"] for item in transitions[lane]))
                action_count = max(episode_steps, 1)
                parse_rate = sum(float(item["valid_action"]) for item in transitions[lane]) / action_count
                noop_rate = sum(float(item["executed_action"] == "NOOP") for item in transitions[lane]) / action_count
                repeated_rate = sum(
                    float(index > 0 and item["executed_action"] == transitions[lane][index - 1]["executed_action"])
                    for index, item in enumerate(transitions[lane])
                ) / action_count
                if not outputs_by_lane[lane]:
                    continue
                # Only the final PPO item carries whole-episode figures.  This
                # prevents a multi-step episode from receiving a larger weight
                # in aggregate train/validation metrics.
                final_output = outputs_by_lane[lane][-1]
                final_telemetry = {
                    "craftext_episode_final": True,
                    "craftext_episode_return": episode_return,
                    "craftext_episode_steps": episode_steps,
                    "craftext_episode_success": episode_success,
                }
                final_output.extra_fields.update(final_telemetry)
                if validate:
                    final_output.extra_fields["reward_extra_info"] = {
                        "craftext_episode_success": episode_success,
                        "craftext_episode_return": episode_return,
                        "craftext_episode_steps": episode_steps,
                        "craftext_episode_parse_rate": parse_rate,
                        "craftext_episode_invalid_rate": 1.0 - parse_rate,
                        "craftext_episode_noop_rate": noop_rate,
                        "craftext_episode_repeated_rate": repeated_rate,
                    }

            if validate and self.validation_trajectory_dir:
                output_dir = Path(str(self.validation_trajectory_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                global_step = int(prompts[0].get("global_steps", 0)) if prompts else 0
                path = output_dir / f"step_{global_step:06d}.jsonl"
                with path.open("w", encoding="utf-8") as file:
                    for lane, lane_transitions in enumerate(transitions[: self.validation_trajectory_limit]):
                        file.write(
                            json.dumps(
                                {
                                    "lane": lane,
                                    "base_seed": self.validation_seed,
                                    "transitions": lane_transitions,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            await self._publish(outputs_by_lane, prompts, partition=partition)
            for uid in uids:
                await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "finished"})
        except Exception:
            for uid in uids:
                await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "failure"})
            raise
        finally:
            env.close()

    async def _publish(
        self, outputs_by_lane: list[list[CraftextAgentLoopOutput]], prompts: list[dict[str, Any]], *, partition: str
    ) -> None:
        keys, fields, tags = [], [], []
        for lane, (lane_outputs, prompt) in enumerate(zip(outputs_by_lane, prompts, strict=True)):
            if not lane_outputs:
                raise RuntimeError(f"CrafText lane {lane} ended without an action output")
            uid = str(prompt["uid"])
            for step_idx, output in enumerate(lane_outputs):
                prompt_ids = torch.tensor(output.prompt_ids, dtype=torch.int64)
                response_ids = torch.tensor(output.response_ids, dtype=torch.int64)
                input_ids = torch.cat([prompt_ids, response_ids], dim=0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                field = output.as_dict()
                field.update(prompt)
                field.pop("multi_modal_data", None)
                # ``as_dict`` nests ``extra_fields``; the frame must not stay there.
                pixel_values = field.get("extra_fields", {}).pop("craftext_pixel_values", None)
                if pixel_values is None:
                    raise RuntimeError("Visual CrafText PPO item is missing the frame used for its action")
                field["loss_mask"] = field["response_mask"]
                field["input_ids"] = input_ids
                field["position_ids"] = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
                field["multi_modal_inputs"] = {"pixel_values": pixel_values}
                keys.append(f"{uid}_0_{step_idx}")
                fields.append(field)
                tags.append(
                    {
                        "status": "success",
                        "prompt_len": prompt_ids.size(0),
                        "response_len": response_ids.size(0),
                        "seq_len": input_ids.size(0),
                        "global_steps": prompt["global_steps"],
                        # V1 uses these tags both for off-policy accounting and
                        # for metric construction, where it casts them to int.
                        # This synchronous step was generated from exactly the
                        # weights represented by the source prompt.
                        "min_global_steps": int(prompt["global_steps"]),
                        "max_global_steps": int(prompt["global_steps"]),
                    }
                )
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id=partition,
        )


class CraftextVectorAgentLoopManager(AgentLoopManagerTQ):
    """V1 manager which schedules one vector worker rather than env actors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(CraftextVectorAgentLoopWorker)
