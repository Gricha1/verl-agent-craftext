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
from typing import Optional, Tuple

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
        self._action_vocab_ids = None
        self._entropy_band_coef = None
        self.device_name = get_device_name()

    def _get_canonical_action_tokens(self, device: torch.device):
        tokenizer_path = self.config.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(
                "entropy_over_valid_actions requires actor.tokenizer_path (set in fsdp_workers init)"
            )
        trust_remote_code = bool(self.config.get("trust_remote_code", False))
        single_token_actions = bool(self.config.get("single_token_actions", False))
        cache_key = (tokenizer_path, trust_remote_code, single_token_actions)
        if self._canonical_action_cache_key != cache_key:
            from verl.utils.action_set_entropy import _load_canonical_action_token_cache

            padded, lengths, _ = _load_canonical_action_token_cache(
                tokenizer_path, trust_remote_code, single_token_actions=single_token_actions
            )
            self._canonical_action_cache_key = cache_key
            self._canonical_action_padded = padded
            self._canonical_action_lengths = lengths
            self._action_vocab_ids = None
        return (
            self._canonical_action_padded.to(device),
            self._canonical_action_lengths.to(device),
        )

    def _get_action_vocab_ids(self, device: torch.device) -> torch.Tensor:
        canonical_padded, canonical_lengths = self._get_canonical_action_tokens(device)
        if self._action_vocab_ids is None:
            from verl.utils.action_set_entropy import action_vocab_ids_from_canonical

            self._action_vocab_ids = action_vocab_ids_from_canonical(
                canonical_padded.cpu(), canonical_lengths.cpu()
            )
        return self._action_vocab_ids.to(device)

    def _compute_action_set_log_scores(
        self,
        micro_batch: dict,
        temperature: float,
    ) -> torch.Tensor:
        """Log-probability scores for each canonical <action>X</action> string. Shape (B, num_actions)."""
        from verl.utils.action_set_entropy import (
            action_log_scores_from_next_token_logits,
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

        use_single_token_fastpath = (
            bool(self.config.get("single_token_actions", False))
            and bool(self.config.get("entropy_action_single_token_fastpath", True))
            and int(canonical_lengths.max().item()) == 1
        )
        if use_single_token_fastpath:
            prompt_pos_ids = (prompt_mask.cumsum(dim=1) - 1).clamp(min=0) * prompt_mask
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                pos = prompt_pos_ids
                if mrope:
                    pos = pos.unsqueeze(0).expand(3, -1, -1)
                try:
                    output = self.actor_module(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        position_ids=pos,
                        **multi_modal_inputs,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                except TypeError:
                    output = self.actor_module(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        position_ids=pos,
                        **multi_modal_inputs,
                        use_cache=False,
                    )
                if self.use_fused_kernels and hasattr(output, "logits") and output.logits is None:
                    raise NotImplementedError(
                        "entropy_over_valid_actions requires non-fused logits path; "
                        "set actor_rollout_ref.model.use_fused_kernels=False"
                    )
                logits = output.logits / temp
            # (B, 1, V) or (B, V): keep only 17 action-token logits to avoid full-vocab softmax backward.
            if logits.dim() == 3:
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = logits
            del logits, output
            action_vocab_ids = self._get_action_vocab_ids(input_ids.device)
            action_logits = next_token_logits.index_select(-1, action_vocab_ids)
            del next_token_logits
            get_torch_device().empty_cache()
            return action_logits.float()

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
            if chunk_size < 0:
                # -1: one forward over the full B×num_actions flat batch (no chunking).
                chunk_size = flat_batch
            else:
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

    @staticmethod
    def _greedy_response_token_accuracy(logits, labels, response_mask):
        """Argmax token accuracy on teacher-forced response positions."""
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask = response_mask.bool()
            correct = (preds == labels) & mask
            return correct.sum(), mask.sum()

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        compute_accuracy=False,
        response_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            acc_stats: optional (correct, total) token counts for WM accuracy
        """
        response_length = micro_batch["responses"].size(-1)
        acc_stats = None
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
                    if calculate_entropy or compute_accuracy:
                        inplace_backward = False
                    if compute_accuracy and response_mask is not None:
                        logits_for_acc = logits_rmpad
                        if self.use_ulysses_sp:
                            logits_for_acc = gather_outpus_and_unpad(
                                logits_for_acc,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        full_logits = pad_input(
                            hidden_states=logits_for_acc,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        logits_resp = full_logits[:, -response_length - 1 : -1, :]
                        acc_stats = self._greedy_response_token_accuracy(
                            logits_resp, micro_batch["responses"], response_mask
                        )
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
                    if self.config.get("single_token_actions", False):
                        from verl.utils.actor_value_token import action_token_id_tensor, mask_logits_to_allowed

                        tokenizer = self._get_tokenizer()
                        action_ids = action_token_id_tensor(tokenizer, device=logits.device)
                        logits[:, 0, :] = mask_logits_to_allowed(logits[:, 0, :], action_ids)
                    if compute_accuracy and response_mask is not None:
                        acc_stats = self._greedy_response_token_accuracy(
                            logits, micro_batch["responses"], response_mask
                        )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, acc_stats

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
                entropy, log_probs, _ = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
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

    def _get_tokenizer(self):
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            return self.tokenizer
        tokenizer_path = self.config.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("actor_value_token requires actor.tokenizer_path")
        from verl.utils import hf_tokenizer

        self.tokenizer = hf_tokenizer(
            tokenizer_path, trust_remote_code=self.config.get("trust_remote_code", False)
        )
        return self.tokenizer

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_actor_token_values(self, data: DataProto) -> torch.Tensor:
        """Per-token V(s) from critic prompt + teacher-forced response (same layout as separate critic)."""
        from verl.utils.actor_value_token import per_token_values_from_critic_prompt

        self.actor_module.eval()
        batch = data.select(
            batch_keys=[
                "value_input_ids",
                "value_attention_mask",
                "value_position_ids",
                "responses",
                "attention_mask",
            ]
        ).batch
        tokenizer = self._get_tokenizer()
        response_length = batch["responses"].size(1)
        response_mask = batch["attention_mask"][:, -response_length:]
        from verl.utils.actor_value_token import return_bin_spec_from_actor_cfg

        spec = return_bin_spec_from_actor_cfg(self.config)
        micro_batch_size = int(
            data.meta_info.get(
                "micro_batch_size",
                self.config.get(
                    "actor_value_micro_batch_size_per_gpu",
                    self.config.get(
                        "ppo_micro_batch_size_per_gpu",
                        16,
                    ),
                ),
            )
        )
        micro_batch_size = max(micro_batch_size, 1)
        n = batch["value_input_ids"].size(0)
        value_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n, micro_batch_size):
                end = min(start + micro_batch_size, n)
                values = per_token_values_from_critic_prompt(
                    self.actor_module,
                    value_input_ids=batch["value_input_ids"][start:end],
                    value_attention_mask=batch["value_attention_mask"][start:end],
                    value_position_ids=batch["value_position_ids"][start:end],
                    responses=batch["responses"][start:end],
                    response_mask=response_mask[start:end],
                    tokenizer=tokenizer,
                    spec=spec,
                )
                value_chunks.append(values)
            values = torch.cat(value_chunks, dim=0)
        return values

    def _forward_value_bin_logits(
        self,
        micro_batch: dict,
        temperature: float,
    ) -> torch.Tensor:
        from verl.utils.actor_value_token import (
            forward_value_bin_logits_per_response_token,
            return_bin_spec_from_actor_cfg,
        )

        responses = micro_batch["responses"]
        response_length = responses.size(1)
        if "response_mask" in micro_batch:
            response_mask = micro_batch["response_mask"]
        else:
            response_mask = micro_batch["attention_mask"][:, -response_length:]
        spec = return_bin_spec_from_actor_cfg(self.config)
        return forward_value_bin_logits_per_response_token(
            self.actor_module,
            value_input_ids=micro_batch["value_input_ids"],
            value_attention_mask=micro_batch["value_attention_mask"],
            value_position_ids=micro_batch["value_position_ids"],
            responses=responses,
            response_mask=response_mask,
            tokenizer=self._get_tokenizer(),
            temperature=temperature,
            spec=spec,
        )

    def _compute_return_token_ce_loss(
        self,
        micro_batch: dict,
        temperature: float,
    ) -> tuple[torch.Tensor, float, float]:
        from verl.utils.actor_value_token import (
            return_bin_distribution_entropy,
            return_bin_spec_from_actor_cfg,
            return_token_value_loss,
        )

        bin_logits = self._forward_value_bin_logits(micro_batch, temperature)
        response_length = micro_batch["responses"].size(1)
        if "response_mask" in micro_batch:
            response_mask = micro_batch["response_mask"]
        else:
            response_mask = micro_batch["attention_mask"][:, -response_length:]

        spec = return_bin_spec_from_actor_cfg(self.config)
        if bool(self.config.get("actor_value_target_from_returns", False)):
            target_returns = micro_batch["returns"].detach()
        else:
            target_returns = micro_batch["actor_value_target_returns"]

        target_encoding = str(self.config.get("actor_value_target_encoding", "one_hot"))
        ce_loss, _hard_bins = return_token_value_loss(
            bin_logits,
            target_returns,
            target_encoding=target_encoding,
            response_mask=response_mask,
            spec=spec,
        )
        entropy_coef = float(self.config.get("actor_value_entropy_coef", 0.0))
        if entropy_coef != 0.0:
            if bin_logits.dim() == 3:
                valid = response_mask.bool()
                entropy_term = return_bin_distribution_entropy(bin_logits[valid]).mean()
            else:
                entropy_term = return_bin_distribution_entropy(bin_logits).mean()
            loss = ce_loss - entropy_coef * entropy_term
        else:
            loss = ce_loss
        with torch.no_grad():
            target_bins = (
                target_returns.clamp(min=spec.vmin, max=spec.vmax) - spec.vmin
            ) / spec.step
            target_bins = target_bins.round().to(dtype=torch.long)
            if bin_logits.dim() == 3:
                pred_bins = bin_logits.argmax(dim=-1)
                if target_bins.dim() == 1:
                    target_bins = target_bins.unsqueeze(1).expand_as(pred_bins)
                valid = response_mask.bool()
                acc = (pred_bins[valid] == target_bins[valid]).float().mean().item()
                entropy = return_bin_distribution_entropy(bin_logits[valid]).mean().item()
            else:
                pred_bins = bin_logits.argmax(dim=-1)
                acc = (pred_bins == target_bins).float().mean().item()
                entropy = return_bin_distribution_entropy(bin_logits).mean().item()
        return loss, acc, entropy

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_world_model(self, data: DataProto):
        """SFT-style world model update on the full rollout batch (after PPO). One optimizer.step()."""
        self.actor_module.train()

        loss_coef = float(data.meta_info.get("world_model_loss_coef", 1.0))
        if "temperature" not in data.meta_info:
            raise KeyError(
                "update_world_model requires meta_info['temperature'] "
                "(set on driver from actor_rollout_ref.rollout.temperature)"
            )
        temperature = float(data.meta_info["temperature"])
        wm_micro = data.meta_info.get("world_model_micro_batch_size_per_gpu")
        if wm_micro is None:
            wm_micro = self.config.get(
                "world_model_micro_batch_size_per_gpu",
                self.config.ppo_micro_batch_size_per_gpu,
            )
        wm_micro = int(wm_micro)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(wm_micro)

        self.actor_optimizer.zero_grad()
        metrics: dict = {}
        num_micro = max(len(micro_batches), 1)
        last_wm_loss = None
        acc_correct = 0
        acc_total = 0

        for micro_batch in micro_batches:
            micro_data = {**micro_batch.to(get_torch_device().current_device())}
            responses = micro_data["responses"]
            response_length = responses.size(1)
            attention_mask = micro_data["attention_mask"]
            response_mask = attention_mask[:, -response_length:]

            _, log_prob, acc_stats = self._forward_micro_batch(
                micro_batch=micro_data,
                temperature=temperature,
                calculate_entropy=False,
                compute_accuracy=True,
                response_mask=response_mask,
            )
            if acc_stats is not None:
                acc_correct += int(acc_stats[0].item())
                acc_total += int(acc_stats[1].item())
            wm_loss = -agg_loss(
                loss_mat=log_prob,
                loss_mask=response_mask,
                loss_agg_mode="token-mean",
            )
            last_wm_loss = wm_loss
            (wm_loss * loss_coef / num_micro).backward()

        grad_norm = self._optimizer_step()
        wm_loss_val = last_wm_loss.detach().item() if last_wm_loss is not None else 0.0
        metrics.update({
            "world_model/loss": wm_loss_val,
            "world_model/scaled_loss": wm_loss_val * loss_coef,
            "world_model/loss_coef": loss_coef,
            "world_model/grad_norm": grad_norm.detach().item(),
            "world_model/num_samples": float(batch.batch_size[0]),
        })
        if acc_total > 0:
            metrics["world_model/reward_token_accuracy"] = acc_correct / acc_total
            metrics["world_model/reward_token_correct"] = float(acc_correct)
            metrics["world_model/reward_token_total"] = float(acc_total)
        self.actor_optimizer.zero_grad()
        get_torch_device().empty_cache()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        actor_value_token = bool(self.config.get("actor_value_token", False))
        actor_value_loss_coef = float(self.config.get("actor_value_loss_coef", 1.0))
        actor_value_separate_steps = bool(
            actor_value_token and self.config.get("actor_value_separate_optimizer_steps", False)
        )
        value_warmup = bool(data.meta_info.get("actor_value_warmup", False))
        effective_entropy_coeff = float(
            data.meta_info.get("entropy_coeff", self.config.entropy_coeff)
        )
        from verl.utils.entropy_band import adapt_entropy_coeff, entropy_band_enabled

        entropy_band_cfg = self.config.get("entropy_band") or {}
        entropy_band_on = entropy_band_enabled(entropy_band_cfg)
        if entropy_band_on and self._entropy_band_coef is None:
            self._entropy_band_coef = effective_entropy_coeff
        entropy_bonus_coeff = (
            self._entropy_band_coef if entropy_band_on else effective_entropy_coeff
        )
        metrics = {"actor/entropy_coeff": effective_entropy_coeff}
        if entropy_band_on:
            metrics["actor/entropy_band_coef"] = entropy_bonus_coeff
            metrics["actor/entropy_band_low"] = float(entropy_band_cfg.get("low", 0.7))
            metrics["actor/entropy_band_high"] = float(entropy_band_cfg.get("high", 1.4))
        if actor_value_separate_steps:
            metrics["actor/value_separate_optimizer_steps"] = 1.0

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if actor_value_token:
            select_keys.extend(
                [
                    "actor_value_target_returns",
                    "value_input_ids",
                    "value_attention_mask",
                    "value_position_ids",
                    "returns",
                ]
            )
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

                if actor_value_separate_steps:
                    update_phases = []
                    if not value_warmup:
                        update_phases.append("actor")
                    update_phases.append("value")
                else:
                    update_phases = ["combined"]

                if value_warmup:
                    metrics["actor/value_warmup"] = 1.0
                    append_to_dict(metrics, {"actor/pg_loss": 0.0})

                micro_batches = list(micro_batches)
                n_micro = len(micro_batches)

                for phase in update_phases:
                    if phase == "value" and actor_value_separate_steps:
                        get_torch_device().empty_cache()
                    self.actor_optimizer.zero_grad()
                    if actor_value_token and (
                        not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    ):
                        print(
                            f"[update_policy] phase={phase} epoch={epoch} "
                            f"mini_batch={batch_idx} n_micro={n_micro}",
                            flush=True,
                        )

                    for mb_idx, data in enumerate(micro_batches):
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
                        entropy_coeff = entropy_bonus_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        entropy_over_valid_actions = bool(
                            self.config.get("entropy_over_valid_actions", False)
                        )
                        if self.config.use_dynamic_bsz:
                            loss_scale = len(data) / self.config.ppo_mini_batch_size
                        else:
                            loss_scale = 1.0 / self.gradient_accumulation

                        run_actor = phase in ("actor", "combined") and not value_warmup
                        run_value = (
                            phase in ("value", "combined")
                            and actor_value_token
                            and "actor_value_target_returns" in data
                        )

                        if run_actor:
                            # Action-set entropy: separate backward so 17 candidate forwards are not
                            # on the same autograd graph as PPO (avoids OOM during backward / vLLM wake_up).
                            if entropy_coeff != 0 and entropy_over_valid_actions:
                                from verl.utils.action_set_entropy import entropy_loss_over_action_scores

                                action_logits = self._compute_action_set_log_scores(data, temperature)
                                sample_mask = None
                                if self.config.get("entropy_action_only_valid_rollouts", False):
                                    if "is_action_valid" in data:
                                        sample_mask = torch.tensor(
                                            data["is_action_valid"],
                                            device=action_logits.device,
                                            dtype=torch.bool,
                                        )
                                entropy_loss, entropy_per_sample = entropy_loss_over_action_scores(
                                    action_logits,
                                    temperature=temperature,
                                    sample_mask=sample_mask,
                                )
                                metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                                action_set_h = entropy_per_sample.detach().mean().item()
                                metrics["actor/action_set_entropy"] = action_set_h
                                (-entropy_coeff * entropy_loss * loss_scale).backward()
                                if entropy_band_on:
                                    self._entropy_band_coef = adapt_entropy_coeff(
                                        entropy_coeff,
                                        action_set_h,
                                        entropy_band_cfg,
                                    )
                                    entropy_bonus_coeff = self._entropy_band_coef
                                    metrics["actor/entropy_band_coef"] = self._entropy_band_coef
                                del action_logits, entropy_loss, entropy_per_sample
                                get_torch_device().empty_cache()

                            calculate_entropy = entropy_coeff != 0 and not entropy_over_valid_actions
                            entropy, log_prob, _ = self._forward_micro_batch(
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
                                kld = kl_penalty(
                                    logprob=log_prob,
                                    ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type,
                                )
                                kl_loss = agg_loss(
                                    loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                                )

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                metrics["actor/kl_loss"] = kl_loss.detach().item()
                                metrics["actor/kl_coef"] = self.config.kl_loss_coef

                            append_to_dict(
                                metrics,
                                {
                                    "actor/pg_loss": pg_loss.detach().item(),
                                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                    "actor/ppo_kl": ppo_kl.detach().item(),
                                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                                },
                            )
                            (policy_loss * loss_scale).backward()
                            get_torch_device().empty_cache()

                        if run_value:
                            value_loss, value_acc, value_entropy = self._compute_return_token_ce_loss(
                                micro_batch=data,
                                temperature=temperature,
                            )
                            value_entropy_coef = float(self.config.get("actor_value_entropy_coef", 0.0))
                            append_to_dict(
                                metrics,
                                {
                                    "actor/value_token_loss": value_loss.detach().item(),
                                    "actor/value_token_accuracy": value_acc,
                                    "actor/value_token_entropy": value_entropy,
                                    "actor/value_token_entropy_coef": value_entropy_coef,
                                },
                            )
                            (value_loss * actor_value_loss_coef * loss_scale).backward()
                            get_torch_device().empty_cache()

                    if actor_value_token and (
                        not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    ):
                        print(
                            f"[update_policy] phase={phase} optimizer.step "
                            f"(mini_batch={batch_idx})",
                            flush=True,
                        )
                    get_torch_device().synchronize()
                    grad_norm = self._optimizer_step()
                    grad_norm_key = "actor/value_grad_norm" if phase == "value" else "actor/grad_norm"
                    append_to_dict(metrics, {grad_norm_key: grad_norm.detach().item()})
        get_torch_device().empty_cache()
        self.actor_optimizer.zero_grad()
        return metrics
