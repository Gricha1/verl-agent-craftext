"""CrafText multi-turn loop for modern veRL's asynchronous rollout API.

The stock ``AgentLoopOutput.as_dict`` has one trajectory-level reward and puts
it on the final token.  CrafText instead emits an environment reward after
every action.  This module preserves the turn boundaries and the companion
worker materialises them as dense PPO ``rm_scores``.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    ToolListWrap,
    register,
)

from .actions import parse_reasoning_action
from .env_adapter import CraftextJaxVectorEnv
from .rewards import route_turn_rewards
from .trajectory import TurnTrajectory

logger = logging.getLogger(__name__)


def _craftext_prompt(
    *,
    task: str,
    observation: str,
    step_count: int = 0,
    action_history: list[str] | None = None,
    reasoning_history: list[str] | None = None,
) -> str:
    """Build the per-action policy prompt used by the legacy CrafText run.

    A response is deliberately *not* fed back verbatim.  The old rollout
    stores the canonical executed action and a bounded reasoning excerpt in
    prompt-side history, while each PPO item remains one model response.
    Keeping that boundary here is important: an observation is context for the
    next action, not a zero-masked token span in the preceding response.
    """
    action_history_text = "\n".join(
        f"{index}. {action}" for index, action in enumerate(action_history or [], start=1)
    ) or "No actions taken yet."
    reasoning_history_text = "\n".join(
        f"{index}. {reasoning}" for index, reasoning in enumerate(reasoning_history or [], start=1) if reasoning
    )
    reasoning_section = (
        "\nRecent reasoning from prior steps (oldest → newest):\n"
        f"{reasoning_history_text}\n"
        if reasoning_history_text
        else ""
    )
    return (
        "Your goal is to complete the following task:\n"
        f"**TASK:** {task}\n\n"
        f"You have taken {step_count} actions so far. Here is the history of your recent actions:\n"
        f"{action_history_text}\n\n"
        f"You are now at step {step_count}.\n"
        f"{reasoning_section}\n"
        "This is what you currently see:\n"
        f"{observation}\n\n"
        "First, think about what to do next. Then, choose one of the available actions and write it in the "
        "<action> tag.\n"
        "Your available actions are:\n"
        "NOOP, LEFT, RIGHT, UP, DOWN, DO, SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, "
        "MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, "
        "MAKE_IRON_SWORD\n"
        "Write EXACTLY one of the action names above (UPPERCASE, underscores preserved) inside the <action> tag."
    )


class CraftextAgentLoopOutput(AgentLoopOutput):
    """Agent output which serialises dense turn rewards for PPO.

    V1 publishes ``output.as_dict()`` directly to TransferQueue.  Overriding
    that boundary means the generic V1 worker never gets a chance to collapse
    the episode reward onto the final response token.
    """

    def as_dict(self) -> dict[str, Any]:
        output = super().as_dict()
        ends = self.extra_fields["craftext_turn_end_indices"]
        rewards = self.extra_fields["craftext_turn_rewards"]
        output["rm_scores"] = torch.tensor(
            route_turn_rewards(output["response_mask"].tolist(), ends, rewards), dtype=torch.float32
        )
        return output


@register("craftext_reasoning")
class CraftextReasoningAgentLoop(AgentLoopBase):
    """One complete CrafText episode, retaining all assistant/env turn tokens."""

    def __init__(self, *args: Any, tools: ToolListWrap | None = None, **kwargs: Any) -> None:
        # CrafText has no function tools; accepting ``tools`` keeps Hydra's
        # generic AgentLoop construction contract intact.
        del tools
        super().__init__(*args, **kwargs)
        craftext_config = self.config.get("craftext", {})
        self.max_env_steps = int(craftext_config.get("max_env_steps", 3))
        self.per_turn_max_tokens = int(craftext_config.get("per_turn_max_tokens", 96))
        self.seed = int(craftext_config.get("seed", 0))

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any) -> AgentLoopOutput:
        start_time = time.perf_counter()
        priority = int(priority)
        env = CraftextJaxVectorEnv(
            num_envs=1,
            seed=self.seed + priority,
            max_steps=self.max_env_steps,
            reset_ratio=1,
        )
        try:
            observations, infos = env.reset()
            initial_info = infos[0]
            task = str(initial_info.get("instruction", "Complete the CrafText task safely."))
            messages: list[dict[str, str]] = [
                {"role": "system", "content": "You are a careful embodied CrafText agent."},
                {"role": "user", "content": _craftext_prompt(task=task, observation=observations[0])},
            ]
            runtime_ids = await self.ct_build_initial_tokens(messages)
            initial_prompt_ids = runtime_ids.copy()
            response_mask: list[int] = []
            response_logprobs: list[float] | None = []
            trajectory = TurnTrajectory()
            transitions: list[dict[str, Any]] = []
            metrics: dict[str, float | int] = {
                "generate_sequences": 0.0,
                "tool_calls": 0.0,
                "compute_score": 0.0,
                "num_preempted": 0,
            }
            request_id = f"craftext-{priority}-{uuid4().hex}"

            for turn in range(self.max_env_steps):
                remaining = self.rollout_config.response_length - len(response_mask)
                if remaining <= 0:
                    break
                turn_sampling = deepcopy(sampling_params)
                turn_sampling["max_tokens"] = min(self.per_turn_max_tokens, remaining)
                generation_start = time.perf_counter()
                model_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_ids,
                    sampling_params=turn_sampling,
                    priority=priority,
                )
                metrics["generate_sequences"] += time.perf_counter() - generation_start
                metrics["num_preempted"] += int(model_output.num_preempted or 0)

                merge_result, response_mask, aligned_logprobs = await self.ct_merge_assistant_token(
                    runtime_ids,
                    model_output.token_ids,
                    response_mask,
                    response_logprobs if (response_logprobs or model_output.log_probs) else None,
                    assistant_logprobs=model_output.log_probs if model_output.log_probs else None,
                )
                runtime_ids = merge_result.token_ids
                response_logprobs = aligned_logprobs
                decoded = self.tokenizer.decode(model_output.token_ids, skip_special_tokens=True)
                parsed, next_observations, rewards, dones, next_infos = env.parse_and_step([decoded])
                action = parsed[0]
                reward = float(rewards[0])
                done = bool(dones[0])
                trajectory.append(response_mask=response_mask, reward=reward)
                transitions.append(
                    {
                        "turn": turn,
                        "task": task,
                        "observation": observations[0],
                        "response": decoded,
                        "executed_action": action.action_name,
                        "valid_action": action.valid,
                        "reward": reward,
                        "done": done,
                        "success": bool(next_infos[0].get("instruction_done", False) or next_infos[0].get("won", False)),
                        "next_observation": next_observations[0],
                    }
                )
                if done or turn + 1 >= self.max_env_steps:
                    break

                previous_messages = list(messages)
                messages.extend(
                    [
                        {"role": "assistant", "content": decoded},
                        {
                            "role": "user",
                            "content": _craftext_prompt(
                                task=task, observation=next_observations[0], last_reward=reward
                            ),
                        },
                    ]
                )
                merge_result, response_mask, aligned_logprobs = await self.ct_merge_context_msg(
                    previous_messages,
                    messages,
                    runtime_ids,
                    response_mask,
                    response_logprobs,
                )
                runtime_ids = merge_result.token_ids
                response_logprobs = aligned_logprobs
                observations, infos = next_observations, next_infos

            # ``response_mask`` spans exactly the response suffix of runtime_ids.
            response_ids = runtime_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = runtime_ids[: len(runtime_ids) - len(response_mask)] if response_mask else initial_prompt_ids
            if len(response_ids) > self.rollout_config.response_length:
                raise RuntimeError("CrafText response exceeded rollout.response_length")
            metrics["generate_sequences"] = float(metrics["generate_sequences"])
            metrics["wall_time"] = time.perf_counter() - start_time
            extra_fields = trajectory.as_extra_fields()
            extra_fields["craftext_transitions"] = transitions
            extra_fields["craftext_episode_return"] = float(sum(trajectory.turn_rewards))
            extra_fields["craftext_env_steps"] = len(trajectory.turn_rewards)
            extra_fields["craftext_success"] = float(any(item["success"] for item in transitions))
            extra_fields["craftext_generated_tokens"] = int(sum(response_mask))
            logger.info(
                "craftext_rollout env_steps=%d return=%.4g success=%.0f generated_tokens=%d",
                extra_fields["craftext_env_steps"],
                extra_fields["craftext_episode_return"],
                extra_fields["craftext_success"],
                extra_fields["craftext_generated_tokens"],
            )
            return CraftextAgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                reward_score=float(sum(trajectory.turn_rewards)),
                num_turns=len(trajectory.turn_rewards) * 2 + 1,
                metrics=AgentLoopMetrics(**metrics),
                extra_fields=extra_fields,
            )
        finally:
            env.close()


