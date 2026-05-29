# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self._canonical_action_cache_key = None
        self._canonical_action_padded = None
        self._canonical_action_lengths = None
        self.device_name = get_device_name()

    def _get_canonical_action_tokens(self, device: torch.device):
        tokenizer_path = self.config.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(
                "entropy_over_valid_actions requires actor.tokenizer_path (set in fsdp_workers init)"
            )
        trust_remote_code = bool(self.config.get("trust_remote_code", False))
        cache_key = (tokenizer_path, trust_remote_code)
        if self._canonical_action_cache_key != cache_key:
            from verl.utils.action_set_entropy import _load_canonical_action_token_cache

            padded, lengths, _ = _load_canonical_action_token_cache(
                tokenizer_path, trust_remote_code
            )
            self._canonical_action_cache_key = cache_key
            self._canonical_action_padded = padded
            self._canonical_action_lengths = lengths
        return (
            self._canonical_action_padded.to(device),
            self._canonical_action_lengths.to(device),
        )

    def _compute_action_set_log_scores(
        self,
        micro_batch: dict,
        temperature: float,
    ) -> torch.Tensor:
        """Log-probability scores for each canonical <action>X</action> string. Shape (B, num_actions)."""
        from verl.utils.action_set_entropy import (
            action_log_scores_one_action_batch,
            build_prompt_action_batch_for_one_action,
            build_prompt_action_sequences,
        )

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        batch_size, seqlen = input_ids.shape
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seqlen - response_length

        canonical_padded, canonical_lengths = self._get_canonical_action_tokens(input_ids.device)
        num_actions = canonical_lengths.shape[0]
        pad_token_id = int(self.config.get("pad_token_id", 0))
        temp = max(float(temperature), 1e-8)
        length_normalize = bool(self.config.get("entropy_action_length_normalize", True))
        batched_forward = bool(self.config.get("entropy_action_batched_forward", False))

        prompt_ids = input_ids[:, :prompt_length]
        prompt_mask = attention_mask[:, :prompt_length]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        mrope = position_ids.dim() == 3

        def _run_actor_forward(cand_ids, cand_mask, cand_pos_ids, mm_inputs):
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                pos = cand_pos_ids
                if mrope:
                    pos = pos.unsqueeze(0).expand(3, -1, -1)
                output = self.actor_module(
                    input_ids=cand_ids,
                    attention_mask=cand_mask,
                    position_ids=pos,
                    **mm_inputs,
                    use_cache=False,
                )
                if self.use_fused_kernels and hasattr(output, "logits") and output.logits is None:
                    raise NotImplementedError(
                        "entropy_over_valid_actions requires non-fused logits path; "
                        "set actor_rollout_ref.model.use_fused_kernels=False"
                    )
                return output.logits / temp

        if batched_forward:
            cand_ids, cand_mask, cand_pos_ids, _ = build_prompt_action_sequences(
                prompt_ids,
                prompt_mask,
                canonical_padded,
                canonical_lengths,
                pad_token_id,
            )
            flat_batch = cand_ids.shape[0]
            chunk_size = int(self.config.get("entropy_action_batched_chunk_size", 16))
            chunk_size = max(1, min(chunk_size, flat_batch))

            mm_full = multi_modal_inputs
            if mm_full:
                mm_full = {
                    k: v.repeat_interleave(num_actions, dim=0) for k, v in mm_full.items()
                }

            log_scores = torch.zeros(
                batch_size, num_actions, device=input_ids.device, dtype=torch.float32
            )
            for start in range(0, flat_batch, chunk_size):
                end = min(start + chunk_size, flat_batch)
                chunk_mm = (
                    {k: v[start:end] for k, v in mm_full.items()} if mm_full else {}
                )
                logits = _run_actor_forward(
                    cand_ids[start:end],
                    cand_mask[start:end],
                    cand_pos_ids[start:end],
                    chunk_mm,
                )
                for local_row, global_row in enumerate(range(start, end)):
                    action_idx = global_row % num_actions
                    action_len = int(canonical_lengths[action_idx].item())
                    log_scores[global_row // num_actions, action_idx] = (
                        action_log_scores_one_action_batch(
                            logits[local_row : local_row + 1],
                            cand_ids[global_row : global_row + 1],
                            prompt_length,
                            action_len,
                            length_normalize=length_normalize,
                        )
                        .squeeze(0)
                        .float()
                    )
                del logits
            get_torch_device().empty_cache()
            return log_scores

        log_scores = torch.zeros(
            batch_size, num_actions, device=input_ids.device, dtype=torch.float32
        )
        # One forward per action (B sequences), not B*17 — avoids huge logits OOM.
        for j in range(num_actions):
            action_len = int(canonical_lengths[j].item())
            cand_ids, cand_mask, cand_pos_ids = build_prompt_action_batch_for_one_action(
                prompt_ids,
                prompt_mask,
                canonical_padded[j],
                action_len,
                pad_token_id,
            )
            logits = _run_actor_forward(cand_ids, cand_mask, cand_pos_ids, multi_modal_inputs)
            log_scores[:, j] = action_log_scores_one_action_batch(
                logits,
                cand_ids,
                prompt_length,
                action_len,
                length_normalize=length_normalize,
            ).float()
            del logits
            get_torch_device().empty_cache()

        return log_scores

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        if calculate_entropy and self.config.get("entropy_over_valid_actions", False):
            calculate_entropy = False
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        # Применяем маску action токенов, если включено log_prob_action_only
        log_prob_action_only = self.config.get("log_prob_action_only", False)
        if log_prob_action_only:
            # Получаем tokenizer из config или создаем заново
            tokenizer = None
            if hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer
            elif hasattr(self.config, 'tokenizer_path') or 'tokenizer_path' in self.config:
                from verl.utils import hf_tokenizer
                tokenizer_path = self.config.get('tokenizer_path', None)
                if tokenizer_path:
                    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=self.config.get("trust_remote_code", False))
            
            if tokenizer is not None:
                # Декодируем response токены в текст
                responses = batch["responses"]
                response_texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
                
                # Извлекаем маску action токенов
                from verl.utils.action_token_extraction import extract_action_token_positions_simple
                action_mask = extract_action_token_positions_simple(
                    response_texts=response_texts,
                    response_token_ids=responses,
                    tokenizer=tokenizer
                )
                
                # Применяем маску к log_probs: устанавливаем 0 для не-action токенов
                # Это эквивалентно игнорированию их при суммировании
                log_probs = log_probs * action_mask.float()
                
                if calculate_entropy:
                    # Также применяем маску к entropy
                    entropys = entropys * action_mask.float()

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        effective_entropy_coeff = float(
            data.meta_info.get("entropy_coeff", self.config.entropy_coeff)
        )
        metrics = {"actor/entropy_coeff": effective_entropy_coeff}

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = effective_entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    entropy_over_valid_actions = bool(
                        self.config.get("entropy_over_valid_actions", False)
                    )
                    if self.config.use_dynamic_bsz:
                        loss_scale = len(data) / self.config.ppo_mini_batch_size
                    else:
                        loss_scale = 1.0 / self.gradient_accumulation

                    # Action-set entropy: separate backward so 17 candidate forwards are not
                    # on the same autograd graph as PPO (avoids OOM during backward / vLLM wake_up).
                    if entropy_coeff != 0 and entropy_over_valid_actions:
                        from verl.utils.action_set_entropy import entropy_loss_over_action_scores

                        log_scores = self._compute_action_set_log_scores(data, temperature)
                        sample_mask = None
                        if self.config.get("entropy_action_only_valid_rollouts", False):
                            if "is_action_valid" in data:
                                sample_mask = torch.tensor(
                                    data["is_action_valid"],
                                    device=log_scores.device,
                                    dtype=torch.bool,
                                )
                        entropy_loss, entropy_per_sample = entropy_loss_over_action_scores(
                            log_scores,
                            temperature=temperature,
                            sample_mask=sample_mask,
                        )
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        metrics["actor/action_set_entropy"] = entropy_per_sample.detach().mean().item()
                        (-entropy_coeff * entropy_loss * loss_scale).backward()
                        del log_scores, entropy_loss, entropy_per_sample
                        get_torch_device().empty_cache()

                    calculate_entropy = entropy_coeff != 0 and not entropy_over_valid_actions
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    policy_loss = pg_loss
                    if entropy_coeff != 0 and not entropy_over_valid_actions:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    (policy_loss * loss_scale).backward()
                    get_torch_device().empty_cache()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        get_torch_device().empty_cache()
        self.actor_optimizer.zero_grad()
        return metrics
