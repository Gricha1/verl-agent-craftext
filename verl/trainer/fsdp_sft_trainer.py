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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import tempfile
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.nn import functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        # Optional: reward-WM label ids for per-batch distribution logging.
        self._reward_wm_token_ids = None  # type: ignore[assignment]
        try:
            from agent_system.environments.env_package.caged_craftext.reward_tokens import reward_token_strings

            tok_ids = []
            for t in reward_token_strings():
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if len(ids) == 1:
                    tok_ids.append(int(ids[0]))
            if len(tok_ids) == 4:
                self._reward_wm_token_ids = {
                    -1: tok_ids[0],
                    0: tok_ids[1],
                    1: tok_ids[2],
                    2: tok_ids[3],
                }
        except Exception:
            self._reward_wm_token_ids = None
        self._reward_wm_max_horizon = max(1, int(getattr(self.config.trainer, "reward_horizon", 1) or 1))
        inv_cfg = getattr(self.config.trainer, "inverse_action_wm", None)
        self._inverse_action_wm_enabled = (
            inv_cfg is not None and self._config_bool(getattr(inv_cfg, "enable", False))
        )
        self._inverse_loss_coef = float(getattr(inv_cfg, "loss_coef", 1.0) or 1.0) if inv_cfg is not None else 1.0
        plan_cfg = getattr(self.config.trainer, "planning_wm", None)
        self._planning_wm_enabled = (
            plan_cfg is not None and self._config_bool(getattr(plan_cfg, "enable", False))
        )
        self._planning_loss_coef = float(getattr(plan_cfg, "loss_coef", 1.0) or 1.0) if plan_cfg is not None else 1.0
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        if self._inverse_action_wm_enabled and not getattr(train_dataset, "has_inverse_columns", False):
            raise ValueError(
                "trainer.inverse_action_wm.enable=true but train parquet lacks "
                "state/state_after/action_token columns. Recollect:\n"
                f"  REWARD_HORIZON={self._reward_wm_max_horizon} "
                "bash examples/world_model/collect_reward_wm_dataset_debug_square.sh"
            )
        if self._inverse_action_wm_enabled and self.device_mesh.get_rank() == 0:
            print(
                f"Inverse-action WM SFT enabled (same batch as reward): "
                f"loss_coef={self._inverse_loss_coef}",
                flush=True,
            )
        if self._planning_wm_enabled and not getattr(train_dataset, "has_planning_columns", False):
            raise ValueError(
                "trainer.planning_wm.enable=true but dataset lacks planning columns "
                "(horizon, state, future_rewards, future_actions). Recollect with matching "
                f"REWARD_HORIZON={self._reward_wm_max_horizon} and enable planning in data config."
            )
        if self._planning_wm_enabled and self.device_mesh.get_rank() == 0:
            print(
                f"Return-conditioned planning WM enabled (DT-style, H={self._reward_wm_max_horizon}): "
                f"loss_coef={self._planning_loss_coef}",
                flush=True,
            )
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    @staticmethod
    def _config_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        dl_workers = int(getattr(config.data, "num_workers", 8) or 0)

        _bal = getattr(config.trainer, "reward_wm_balance_horizon_batches", True)
        if isinstance(_bal, str):
            balance_batches = _bal.strip().lower() in ("1", "true", "yes", "on")
        else:
            balance_batches = bool(_bal) if _bal is not None else True
        self._use_horizon_balanced_batches = (
            self._reward_wm_max_horizon > 1
            and getattr(train_dataset, "horizons", None) is not None
            and balance_batches
        )
        if self._reward_wm_max_horizon > 1 and getattr(train_dataset, "horizons", None) is not None:
            if self.device_mesh.get_rank() == 0 and not balance_batches:
                print(
                    "Reward WM: horizon-balanced batches OFF — "
                    "using random DistributedSampler (h mix varies per batch).",
                    flush=True,
                )
        if self._use_horizon_balanced_batches:
            from collections import Counter

            hz_counts = Counter(int(h) for h in train_dataset.horizons)
            needed = set(range(1, int(self._reward_wm_max_horizon) + 1))
            missing = sorted(needed - set(hz_counts.keys()))
            if missing:
                if self.device_mesh.get_rank() == 0:
                    print(
                        "[reward_wm] Train dataset horizon counts: "
                        + " ".join(f"h{k}={v}" for k, v in sorted(hz_counts.items())),
                        flush=True,
                    )
                    print(
                        f"[reward_wm] trainer.reward_horizon={self._reward_wm_max_horizon} "
                        f"but parquet has no rows for horizon(s) {missing}.",
                        flush=True,
                    )
                    print(
                        "Recollect the dataset, then train with the same REWARD_HORIZON:\n"
                        f"  REWARD_HORIZON={self._reward_wm_max_horizon} "
                        "bash examples/world_model/collect_reward_wm_dataset_debug_square.sh\n"
                        f"  REWARD_HORIZON={self._reward_wm_max_horizon} "
                        "bash examples/world_model/train_reward_wm_sft.sh\n"
                        "Or use legacy single-step data: REWARD_HORIZON=1 bash examples/world_model/train_reward_wm_sft.sh",
                        flush=True,
                    )
                raise ValueError(
                    f"No training samples with horizon={missing[0]}. "
                    f"Dataset has {dict(sorted(hz_counts.items()))}; "
                    f"recollect with REWARD_HORIZON={self._reward_wm_max_horizon}."
                )

            from verl.utils.dataset.horizon_batch_sampler import HorizonBalancedBatchSampler

            seed = int(getattr(config.trainer, "seed", 1) or 1)
            self.train_batch_sampler = HorizonBalancedBatchSampler(
                train_dataset,
                batch_size=int(config.data.train_batch_size),
                num_horizons=int(self._reward_wm_max_horizon),
                micro_batch_size=int(config.data.micro_batch_size_per_gpu),
                rank=rank,
                world_size=world_size,
                seed=seed,
                drop_last=True,
            )
            effective_bs = int(self.train_batch_sampler.batch_size)
            if effective_bs != int(config.data.train_batch_size):
                if self.device_mesh.get_rank() == 0:
                    print(
                        f"Reward WM: adjusted train_batch_size "
                        f"{config.data.train_batch_size} -> {effective_bs} "
                        f"(divisible by horizon={self._reward_wm_max_horizon} "
                        f"and micro_batch={config.data.micro_batch_size_per_gpu})",
                        flush=True,
                    )
                self.config.data.train_batch_size = effective_bs
            if self.device_mesh.get_rank() == 0:
                bs = self.train_batch_sampler.batch_size
                ph = self.train_batch_sampler.per_horizon
                print(
                    f"Reward WM: horizon-balanced train batches "
                    f"(batch_size={bs}, per_horizon={ph}, H={self._reward_wm_max_horizon})",
                    flush=True,
                )
            self.train_sampler = None
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                num_workers=dl_workers,
                pin_memory=True,
            )
        else:
            self.train_batch_sampler = None
            self.train_sampler = DistributedSampler(
                self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
            )
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=config.data.train_batch_size,
                sampler=self.train_sampler,
                num_workers=dl_workers,
                pin_memory=True,
                drop_last=True,
            )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=dl_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _extract_inverse_batch(self, batch: TensorDict) -> TensorDict | None:
        """Inverse-action tensors from the same reward batch (zero loss_mask when invalid)."""
        if "inverse_input_ids" not in batch:
            return None
        inv = TensorDict(
            {
                "input_ids": batch["inverse_input_ids"].clone(),
                "attention_mask": batch["inverse_attention_mask"].clone(),
                "position_ids": batch["inverse_position_ids"].clone(),
                "loss_mask": batch["inverse_loss_mask"].clone(),
            },
            batch_size=batch.batch_size,
        )
        loss_mask = inv["loss_mask"][:, :-1]
        if not (loss_mask > 0).any():
            return None
        return inv

    def _extract_planning_batch(self, batch: TensorDict) -> TensorDict | None:
        if "planning_input_ids" not in batch:
            return None
        plan = TensorDict(
            {
                "input_ids": batch["planning_input_ids"].clone(),
                "attention_mask": batch["planning_attention_mask"].clone(),
                "position_ids": batch["planning_position_ids"].clone(),
                "loss_mask": batch["planning_loss_mask"].clone(),
            },
            batch_size=batch.batch_size,
        )
        loss_mask = plan["loss_mask"][:, :-1]
        if not (loss_mask > 0).any():
            return None
        return plan

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "constant":
            from torch.optim.lr_scheduler import LambdaLR

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _step: 1.0)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _reward_wm_horizon_metrics(self, prefix: str, h_corr, h_tot) -> dict:
        """Build ``{prefix}/reward_token_accuracy_h{1..H}`` from per-horizon counts."""
        metrics = {}
        if h_corr is None or h_tot is None:
            return metrics
        max_h = max(1, int(self._reward_wm_max_horizon))
        for h in range(1, max_h + 1):
            tot = float(h_tot[h].item()) if hasattr(h_tot[h], "item") else float(h_tot[h])
            corr = float(h_corr[h].item()) if hasattr(h_corr[h], "item") else float(h_corr[h])
            metrics[f"{prefix}/reward_token_count_h{h}"] = tot
            if tot > 0:
                metrics[f"{prefix}/reward_token_accuracy_h{h}"] = corr / tot
        return metrics

    def _reduce_horizon_counts(self, h_corr, h_tot):
        if h_corr is None or h_tot is None:
            return h_corr, h_tot
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(h_corr, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(h_tot, op=torch.distributed.ReduceOp.SUM)
        return h_corr, h_tot

    def _batch_horizons(self, batch):
        """TensorDict.get(key) raises KeyError; use explicit default."""
        try:
            if "horizon" not in batch.keys():
                return None
        except Exception:
            return None
        horizons = batch.get("horizon", default=None)
        if horizons is None:
            return None
        return horizons.to(self.device_name)

    @staticmethod
    def _infer_batch_size(data) -> int:
        for value in data.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("Cannot infer batch size from dataloader batch")

    def _to_tensor_dict(self, data) -> TensorDict:
        return TensorDict(data, batch_size=self._infer_batch_size(data)).to(self.device_name)

    def _compute_loss_and_backward(
        self,
        batch,
        do_backward: bool = True,
        return_accuracy: bool = False,
        return_entropy: bool = False,
        return_horizon_accuracy: bool = False,
    ):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        horizons = self._batch_horizons(batch)
        loss_mask_2d = batch.pop("loss_mask")[:, :-1].to(self.device_name)
        loss_mask = loss_mask_2d.reshape(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        correct = None
        total = None
        entropy_sum = None
        entropy_count = None
        horizon_correct = None
        horizon_total = None

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
                preds_flat = None
                if return_accuracy or return_horizon_accuracy:
                    with torch.no_grad():
                        preds_flat = torch.argmax(shift_logits, dim=-1)
                if return_accuracy:
                    with torch.no_grad():
                        mask = loss_mask > 0
                        total = mask.sum()
                        if total.item() > 0:
                            correct = (preds_flat.eq(shift_labels) & mask).sum()
                        else:
                            correct = torch.zeros((), device=shift_logits.device, dtype=torch.long)
                            total = torch.zeros((), device=shift_logits.device, dtype=torch.long)
                if (
                    return_horizon_accuracy
                    and horizons is not None
                    and self._reward_wm_token_ids is not None
                    and preds_flat is not None
                ):
                    with torch.no_grad():
                        reward_ids = set(int(x) for x in self._reward_wm_token_ids.values())
                        bsz, slm1 = loss_mask_2d.shape
                        preds_2d = preds_flat.view(bsz, slm1)
                        labels_2d = shift_labels.view(bsz, slm1)
                        max_h = max(1, int(self._reward_wm_max_horizon))
                        horizon_correct = torch.zeros(max_h + 1, device=shift_logits.device, dtype=torch.long)
                        horizon_total = torch.zeros(max_h + 1, device=shift_logits.device, dtype=torch.long)
                        for b in range(bsz):
                            h = int(horizons[b].item())
                            if h < 1 or h > max_h:
                                continue
                            for p in range(slm1):
                                if loss_mask_2d[b, p] <= 0:
                                    continue
                                lid = int(labels_2d[b, p].item())
                                if lid not in reward_ids:
                                    continue
                                horizon_total[h] += 1
                                if int(preds_2d[b, p].item()) == lid:
                                    horizon_correct[h] += 1
                if return_entropy:
                    with torch.no_grad():
                        mask = loss_mask > 0
                        entropy_count = mask.sum()
                        if entropy_count.item() > 0:
                            # Entropy over full vocabulary at supervised positions.
                            logp = F.log_softmax(shift_logits[mask], dim=-1)
                            p = torch.exp(logp)
                            ent = -(p * logp).sum(dim=-1)  # (n_tokens,)
                            entropy_sum = ent.sum()
                        else:
                            entropy_sum = torch.zeros((), device=shift_logits.device, dtype=torch.float32)
                            entropy_count = torch.zeros((), device=shift_logits.device, dtype=torch.long)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            if (return_accuracy or return_entropy or return_horizon_accuracy):
                out = [loss]
                if return_accuracy:
                    out.extend([correct, total])
                if return_entropy:
                    out.extend([entropy_sum, entropy_count])
                if return_horizon_accuracy:
                    out.extend([horizon_correct, horizon_total])
                return tuple(out)
            return loss

    def _accumulate_batch_grads(
        self,
        batch: TensorDict,
        *,
        use_horizon_acc: bool,
        loss_coef: float = 1.0,
    ) -> dict:
        """Forward + backward on one batch; gradients accumulate (no optimizer step)."""
        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0.0
        step_correct = 0
        step_total = 0
        step_entropy_sum = 0.0
        step_entropy_count = 0
        step_h_corr = None
        step_h_tot = None
        if use_horizon_acc:
            step_h_corr = torch.zeros(
                self._reward_wm_max_horizon + 1, device=self.device_name, dtype=torch.long
            )
            step_h_tot = torch.zeros(
                self._reward_wm_max_horizon + 1, device=self.device_name, dtype=torch.long
            )
        for micro_batch in micro_batches:
            out = self._compute_loss_and_backward(
                batch=micro_batch,
                do_backward=False,
                return_accuracy=True,
                return_entropy=True,
                return_horizon_accuracy=use_horizon_acc,
            )
            loss = out[0] / n_micro_batches
            backward_loss = loss * float(loss_coef) if float(loss_coef) != 1.0 else loss
            backward_loss.backward()
            step_loss += float(loss.detach().item())
            try:
                step_correct += int(out[1].item())
                step_total += int(out[2].item())
                step_entropy_sum += float(out[3].item())
                step_entropy_count += int(out[4].item())
                if use_horizon_acc:
                    step_h_corr += out[5]
                    step_h_tot += out[6]
            except Exception:
                pass
        return {
            "loss": step_loss,
            "correct": step_correct,
            "total": step_total,
            "entropy_sum": step_entropy_sum,
            "entropy_count": step_entropy_count,
            "horizon_correct": step_h_corr,
            "horizon_total": step_h_tot,
            "batch": batch,
        }

    @staticmethod
    def _is_inverse_action_metric_prefix(prefix: str) -> bool:
        return "inverse_action" in str(prefix or "")

    @staticmethod
    def _is_planning_metric_prefix(prefix: str) -> bool:
        return "planning" in str(prefix or "")

    @staticmethod
    def _accuracy_metric_key(prefix: str) -> str:
        if prefix == "train":
            return "train/reward_token_accuracy"
        if FSDPSFTTrainer._is_inverse_action_metric_prefix(prefix) or FSDPSFTTrainer._is_planning_metric_prefix(
            prefix
        ):
            return f"{prefix}/accuracy"
        return f"{prefix}/token_accuracy"

    def _metrics_from_accum(
        self,
        accum: dict,
        *,
        prefix: str,
        use_horizon_acc: bool,
        log_reward_fracs: bool = False,
    ) -> dict:
        metrics = {}
        loss_t = torch.tensor(float(accum["loss"]), device=self.device_name)
        if torch.distributed.is_initialized():
            if is_cuda_available:
                torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.AVG)
            else:
                torch.distributed.all_reduce(loss_t)
                if is_npu_available:
                    loss_t /= self.ulysses_device_mesh.size(0)
        metrics[f"{prefix}/loss"] = float(loss_t.detach().item())

        try:
            corr_t = torch.tensor(float(accum["correct"]), device=self.device_name)
            tot_t = torch.tensor(float(accum["total"]), device=self.device_name)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(corr_t, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(tot_t, op=torch.distributed.ReduceOp.SUM)
            if float(tot_t.item()) > 0:
                acc = float((corr_t / tot_t).item())
                metrics[self._accuracy_metric_key(prefix)] = acc
                if self._is_inverse_action_metric_prefix(prefix):
                    metrics[f"{prefix}/token_accuracy"] = acc
                if self._is_planning_metric_prefix(prefix):
                    metrics[f"{prefix}/token_accuracy"] = acc
            if self._is_inverse_action_metric_prefix(prefix) or self._is_planning_metric_prefix(prefix):
                metrics[f"{prefix}/n_supervised_tokens"] = float(tot_t.item())
        except Exception:
            pass

        if use_horizon_acc and accum.get("horizon_correct") is not None:
            try:
                h_corr, h_tot = self._reduce_horizon_counts(
                    accum["horizon_correct"], accum["horizon_total"]
                )
                metrics.update(self._reward_wm_horizon_metrics(prefix, h_corr, h_tot))
                batch = accum.get("batch")
                max_h = int(self._reward_wm_max_horizon)
                if batch is not None and "horizon" in batch:
                    hz = batch["horizon"].detach().cpu().tolist()
                    if not isinstance(hz, list):
                        hz = [hz]
                    for h in range(1, max_h + 1):
                        metrics[f"{prefix}/batch_samples_h{h}"] = float(
                            sum(1 for x in hz if int(x) == h)
                        )
            except Exception:
                pass

        try:
            ent_sum_t = torch.tensor(float(accum["entropy_sum"]), device=self.device_name)
            ent_cnt_t = torch.tensor(float(accum["entropy_count"]), device=self.device_name)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(ent_sum_t, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(ent_cnt_t, op=torch.distributed.ReduceOp.SUM)
            if float(ent_cnt_t.item()) > 0:
                metrics[f"{prefix}/entropy_vocab"] = float((ent_sum_t / ent_cnt_t).item())
        except Exception:
            pass

        if log_reward_fracs and self._reward_wm_token_ids is not None:
            batch = accum.get("batch")
            try:
                if batch is not None and "loss_mask" in batch and "input_ids" in batch:
                    input_ids = batch["input_ids"]
                    loss_mask = batch["loss_mask"][:, :-1]
                    labels = input_ids[:, 1:]
                    mask = loss_mask > 0
                    if mask.any():
                        target_ids = labels[mask]
                        for val, tid in self._reward_wm_token_ids.items():
                            frac = float((target_ids == int(tid)).to(dtype=torch.float32).mean().item())
                            metrics[f"{prefix}/reward_frac_{val}"] = frac
            except Exception:
                pass
        return metrics

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        use_horizon_acc = self._reward_wm_max_horizon > 1
        inverse_batch = None
        if self._inverse_action_wm_enabled:
            inverse_batch = self._extract_inverse_batch(batch)
        planning_batch = None
        if self._planning_wm_enabled:
            planning_batch = self._extract_planning_batch(batch)
        rm_accum = self._accumulate_batch_grads(
            batch, use_horizon_acc=use_horizon_acc, loss_coef=1.0
        )
        inv_accum = None
        if inverse_batch is not None:
            inv_accum = self._accumulate_batch_grads(
                inverse_batch,
                use_horizon_acc=False,
                loss_coef=self._inverse_loss_coef,
            )
        plan_accum = None
        if planning_batch is not None:
            plan_accum = self._accumulate_batch_grads(
                planning_batch,
                use_horizon_acc=False,
                loss_coef=self._planning_loss_coef,
            )

        if self.config.model.strategy == 'fsdp':
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == 'fsdp2':
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        log_gpu_memory_usage("After offload weights", logger=logger)

        metrics = self._metrics_from_accum(
            rm_accum, prefix="train", use_horizon_acc=use_horizon_acc, log_reward_fracs=True
        )
        metrics["train/lr(1e-3)"] = lr * 1e3
        if self._inverse_action_wm_enabled:
            if inv_accum is not None:
                metrics.update(
                    self._metrics_from_accum(
                        inv_accum, prefix="train/inverse_action", use_horizon_acc=False
                    )
                )
            else:
                metrics["train/inverse_action/n_supervised_tokens"] = 0.0
        if self._planning_wm_enabled:
            if plan_accum is not None:
                metrics.update(
                    self._metrics_from_accum(
                        plan_accum, prefix="train/planning", use_horizon_acc=False
                    )
                )
            else:
                metrics["train/planning/n_supervised_tokens"] = 0.0
        return metrics

    def _greedy_decode_reward_response(self, prompt_text: str, max_reward_tokens: int) -> str:
        """Autoregressive decode reward token string (e.g. ``ijk``) after the user prompt."""
        from agent_system.environments.env_package.caged_craftext.reward_tokens import reward_token_strings
        from verl.utils.model import compute_position_id_with_mask

        reward_toks = list(reward_token_strings())
        reward_tok_ids = []
        for t in reward_toks:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                reward_tok_ids.append(int(ids[0]))
        reward_id_set = set(reward_tok_ids)
        eos_id = self.tokenizer.eos_token_id

        chat = [{"role": "user", "content": str(prompt_text)}]
        prompt_str = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device_name)
        attention_mask = tok["attention_mask"].to(self.device_name)
        position_ids = compute_position_id_with_mask(attention_mask).to(self.device_name)

        id_to_reward_tok = {int(rid): tok for tok, rid in zip(reward_toks, reward_tok_ids)}
        decoded_chars: list[str] = []

        self.fsdp_model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            for _ in range(max(1, int(max_reward_tokens))):
                out = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                seq_len = int(attention_mask.sum(dim=1).item()) - 1
                next_id = int(out.logits[0, seq_len, :].argmax(dim=-1).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                if next_id not in reward_id_set:
                    break
                decoded_chars.append(id_to_reward_tok[next_id])
                next_t = torch.tensor([[next_id]], device=self.device_name, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device_name, dtype=attention_mask.dtype)],
                    dim=1,
                )
                position_ids = compute_position_id_with_mask(attention_mask).to(self.device_name)

        return "".join(decoded_chars)

    def _greedy_decode_action_response(self, prompt_text: str) -> str:
        """Greedy decode a single action token after the inverse-action prompt."""
        from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings
        from verl.utils.model import compute_position_id_with_mask

        action_toks = list(action_token_strings())
        action_tok_ids = []
        id_to_action_tok: dict[int, str] = {}
        for t in action_toks:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                tid = int(ids[0])
                action_tok_ids.append(tid)
                id_to_action_tok[tid] = t
        action_id_set = set(action_tok_ids)
        eos_id = self.tokenizer.eos_token_id

        chat = [{"role": "user", "content": str(prompt_text)}]
        prompt_str = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device_name)
        attention_mask = tok["attention_mask"].to(self.device_name)
        position_ids = compute_position_id_with_mask(attention_mask).to(self.device_name)

        self.fsdp_model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            out = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            seq_len = int(attention_mask.sum(dim=1).item()) - 1
            next_id = int(out.logits[0, seq_len, :].argmax(dim=-1).item())
            if eos_id is not None and next_id == int(eos_id):
                return ""
            if next_id in action_id_set:
                return id_to_action_tok[next_id]
            return self.tokenizer.decode([next_id], skip_special_tokens=True).strip()

    def _greedy_decode_planning_action_sequence(self, prompt_text: str, num_actions: int) -> str:
        """Greedy decode H action tokens for return-conditioned planning."""
        from agent_system.environments.env_package.caged_craftext.action_tokens import action_token_strings
        from verl.utils.model import compute_position_id_with_mask

        action_toks = list(action_token_strings())
        action_tok_ids = []
        id_to_action_tok: dict[int, str] = {}
        for t in action_toks:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                tid = int(ids[0])
                action_tok_ids.append(tid)
                id_to_action_tok[tid] = t
        action_id_set = set(action_tok_ids)
        eos_id = self.tokenizer.eos_token_id

        chat = [{"role": "user", "content": str(prompt_text)}]
        prompt_str = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        tok = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tok["input_ids"].to(self.device_name)
        attention_mask = tok["attention_mask"].to(self.device_name)
        position_ids = compute_position_id_with_mask(attention_mask).to(self.device_name)

        decoded: list[str] = []
        self.fsdp_model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            for _ in range(max(1, int(num_actions))):
                out = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                seq_len = int(attention_mask.sum(dim=1).item()) - 1
                next_id = int(out.logits[0, seq_len, :].argmax(dim=-1).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                if next_id not in action_id_set:
                    break
                decoded.append(id_to_action_tok[next_id])
                next_t = torch.tensor([[next_id]], device=self.device_name, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device_name, dtype=attention_mask.dtype)],
                    dim=1,
                )
                position_ids = compute_position_id_with_mask(attention_mask).to(self.device_name)
        return "".join(decoded)

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.ulysses_device_mesh.size(0)
        return loss

    def validation_metrics(self, batch: TensorDict, return_entropy: bool = False):
        """Return (loss, correct, total) or (+ entropy_sum, entropy_count, h_corr, h_tot)."""
        use_horizon_acc = self._reward_wm_max_horizon > 1
        self.fsdp_model.eval()
        with torch.no_grad():
            out = self._compute_loss_and_backward(
                batch,
                do_backward=False,
                return_accuracy=True,
                return_entropy=return_entropy,
                return_horizon_accuracy=use_horizon_acc,
            )
            loss, correct, total = out[0], out[1], out[2]
            idx = 3
            entropy_sum = None
            entropy_count = None
            horizon_correct = None
            horizon_total = None
            if return_entropy:
                entropy_sum = out[idx]
                entropy_count = out[idx + 1]
                idx += 2
            if use_horizon_acc:
                horizon_correct = out[idx]
                horizon_total = out[idx + 1]
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                if correct is not None and total is not None:
                    torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
                if return_entropy and entropy_sum is not None and entropy_count is not None:
                    torch.distributed.all_reduce(entropy_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(entropy_count, op=torch.distributed.ReduceOp.SUM)
                if use_horizon_acc and horizon_correct is not None and horizon_total is not None:
                    torch.distributed.all_reduce(horizon_correct, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(horizon_total, op=torch.distributed.ReduceOp.SUM)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.ulysses_device_mesh.size(0)
                if correct is not None and total is not None:
                    torch.distributed.all_reduce(correct)
                    torch.distributed.all_reduce(total)
                if return_entropy and entropy_sum is not None and entropy_count is not None:
                    torch.distributed.all_reduce(entropy_sum)
                    torch.distributed.all_reduce(entropy_count)
                if use_horizon_acc and horizon_correct is not None and horizon_total is not None:
                    torch.distributed.all_reduce(horizon_correct)
                    torch.distributed.all_reduce(horizon_total)
            if return_entropy:
                if use_horizon_acc:
                    return loss, correct, total, entropy_sum, entropy_count, horizon_correct, horizon_total
                return loss, correct, total, entropy_sum, entropy_count
            if use_horizon_acc:
                return loss, correct, total, horizon_correct, horizon_total
            return loss, correct, total

    def _val_max_batches(self) -> int:
        """Cap val dataloader iterations (0 = full val set)."""
        n = int(getattr(self.config.trainer, "val_max_batches", 0) or 0)
        if n <= 0:
            cfg = getattr(self.config.trainer, "reward_wm_val", None)
            if cfg is not None:
                n = int(getattr(cfg, "max_val_batches", 0) or 0)
        return max(0, n)

    def _evaluate_validation(self, return_entropy: bool = True) -> dict:
        """Val dataloader: loss, reward-token accuracy, optional vocab entropy."""
        use_horizon_acc = self._reward_wm_max_horizon > 1
        max_batches = self._val_max_batches()
        val_losses = []
        val_correct = 0.0
        val_total = 0.0
        val_entropy_sum = 0.0
        val_entropy_count = 0.0
        val_h_corr = None
        val_h_tot = None
        if use_horizon_acc:
            val_h_corr = torch.zeros(
                self._reward_wm_max_horizon + 1, device=self.device_name, dtype=torch.long
            )
            val_h_tot = torch.zeros(
                self._reward_wm_max_horizon + 1, device=self.device_name, dtype=torch.long
            )
        for batch_idx, val_data in enumerate(self.val_dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if self.device_mesh.get_rank() == 0 and batch_idx > 0 and batch_idx % 20 == 0:
                print(
                    f"[val] metrics batch {batch_idx}" + (f"/{max_batches}" if max_batches else ""),
                    flush=True,
                )
            val_data = self._to_tensor_dict(val_data)
            if return_entropy:
                out = self.validation_metrics(val_data, return_entropy=True)
                vloss, vcorr, vtot = out[0], out[1], out[2]
                esum, ecnt = out[3], out[4]
                if use_horizon_acc:
                    val_h_corr += out[5]
                    val_h_tot += out[6]
                if esum is not None and ecnt is not None:
                    val_entropy_sum += float(esum.item())
                    val_entropy_count += float(ecnt.item())
            else:
                out = self.validation_metrics(val_data, return_entropy=False)
                vloss, vcorr, vtot = out[0], out[1], out[2]
                if use_horizon_acc:
                    val_h_corr += out[3]
                    val_h_tot += out[4]
            val_losses.append(vloss)
            if vcorr is not None and vtot is not None:
                val_correct += float(vcorr.item())
                val_total += float(vtot.item())
        metrics = {"val/loss": torch.mean(torch.stack(val_losses)).detach().item()}
        if max_batches > 0:
            metrics["val/metrics_batches"] = float(len(val_losses))
        if val_total > 0:
            metrics["val/reward_token_accuracy"] = float(val_correct / val_total)
        if return_entropy and val_entropy_count > 0:
            metrics["val/entropy_vocab"] = float(val_entropy_sum / val_entropy_count)
        if use_horizon_acc and val_h_corr is not None and val_h_tot is not None:
            metrics.update(self._reward_wm_horizon_metrics("val", val_h_corr, val_h_tot))
        if self._inverse_action_wm_enabled:
            metrics.update(self._evaluate_inverse_validation(return_entropy=return_entropy))
        if self._planning_wm_enabled:
            metrics.update(self._evaluate_planning_validation(return_entropy=return_entropy))
        return metrics

    def _evaluate_planning_validation(self, return_entropy: bool = True) -> dict:
        max_batches = self._val_max_batches()
        plan_losses = []
        plan_correct = 0.0
        plan_total = 0.0
        plan_entropy_sum = 0.0
        plan_entropy_count = 0.0
        for batch_idx, val_data in enumerate(self.val_dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            val_data = self._to_tensor_dict(val_data)
            plan_batch = self._extract_planning_batch(val_data)
            if plan_batch is None:
                continue
            if return_entropy:
                out = self.validation_metrics(plan_batch, return_entropy=True)
                vloss, vcorr, vtot = out[0], out[1], out[2]
                esum, ecnt = out[3], out[4]
                if esum is not None and ecnt is not None:
                    plan_entropy_sum += float(esum.item())
                    plan_entropy_count += float(ecnt.item())
            else:
                out = self.validation_metrics(plan_batch, return_entropy=False)
                vloss, vcorr, vtot = out[0], out[1], out[2]
            plan_losses.append(vloss)
            if vcorr is not None and vtot is not None:
                plan_correct += float(vcorr.item())
                plan_total += float(vtot.item())
        metrics: dict = {}
        if plan_losses:
            metrics["val/planning/loss"] = torch.mean(torch.stack(plan_losses)).detach().item()
        if plan_total > 0:
            acc = float(plan_correct / plan_total)
            metrics["val/planning/accuracy"] = acc
            metrics["val/planning/token_accuracy"] = acc
            metrics["val/planning/n_supervised_tokens"] = float(plan_total)
        if return_entropy and plan_entropy_count > 0:
            metrics["val/planning/entropy_vocab"] = float(plan_entropy_sum / plan_entropy_count)
        return metrics

    def _evaluate_inverse_validation(self, return_entropy: bool = True) -> dict:
        """Val loss/accuracy on inverse-action tensors from the same val batches."""
        max_batches = self._val_max_batches()
        inv_losses = []
        inv_correct = 0.0
        inv_total = 0.0
        inv_entropy_sum = 0.0
        inv_entropy_count = 0.0
        for batch_idx, val_data in enumerate(self.val_dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            val_data = self._to_tensor_dict(val_data)
            inv_batch = self._extract_inverse_batch(val_data)
            if inv_batch is None:
                continue
            if return_entropy:
                out = self.validation_metrics(inv_batch, return_entropy=True)
                vloss, vcorr, vtot = out[0], out[1], out[2]
                esum, ecnt = out[3], out[4]
                if esum is not None and ecnt is not None:
                    inv_entropy_sum += float(esum.item())
                    inv_entropy_count += float(ecnt.item())
            else:
                out = self.validation_metrics(inv_batch, return_entropy=False)
                vloss, vcorr, vtot = out[0], out[1], out[2]
            inv_losses.append(vloss)
            if vcorr is not None and vtot is not None:
                inv_correct += float(vcorr.item())
                inv_total += float(vtot.item())
        metrics: dict = {}
        if inv_losses:
            metrics["val/inverse_action/loss"] = torch.mean(torch.stack(inv_losses)).detach().item()
        if inv_total > 0:
            acc = float(inv_correct / inv_total)
            metrics["val/inverse_action/accuracy"] = acc
            metrics["val/inverse_action/token_accuracy"] = acc
            metrics["val/inverse_action/n_supervised_tokens"] = float(inv_total)
        if return_entropy and inv_entropy_count > 0:
            metrics["val/inverse_action/entropy_vocab"] = float(inv_entropy_sum / inv_entropy_count)
        return metrics

    def _sample_inverse_val_rows(self, n_rows: int, seed: int = 0):
        """Sample valid inverse-action rows from val parquet."""
        try:
            import pandas as pd
        except Exception:
            return None

        val_path = self.config.data.val_files
        if isinstance(val_path, (list, tuple)):
            val_path = val_path[0]
        try:
            df = pd.read_parquet(val_path)
        except Exception:
            return None

        required = {"state", "state_after", "action_token"}
        if not required.issubset(set(df.columns)):
            return None

        from agent_system.environments.prompts.world_model_inverse_action import (
            format_inverse_action_prompt,
            is_unchanged_transition,
        )

        def _valid_row(row) -> bool:
            s0 = str(row.get("state", "") or "").strip()
            s1 = str(row.get("state_after", "") or "").strip()
            act = str(row.get("action_token", "") or "").strip().split()[0]
            return bool(s0 and s1 and act and not is_unchanged_transition(s0, s1))

        valid_df = df[df.apply(_valid_row, axis=1)].copy()
        if valid_df.empty:
            return None

        rng = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(valid_df), generator=rng)[: min(int(n_rows), len(valid_df))].tolist()
        sub = valid_df.iloc[idx].copy()

        prompts = []
        gt_actions = []
        for _, row in sub.iterrows():
            prompts.append(
                format_inverse_action_prompt(str(row["state"]), str(row["state_after"]))
            )
            gt_actions.append(str(row["action_token"]).strip().split()[0])
        return sub, prompts, gt_actions

    def _compute_inverse_greedy_val_accuracy(self, n_rows: int | None = None, seed: int = 0) -> dict:
        """Greedy-decode accuracy on a val subset (logged as Comet time series)."""
        inv_cfg = getattr(self.config.trainer, "inverse_action_wm", None)
        if n_rows is None:
            n_rows = int(getattr(inv_cfg, "val_table_n", 40) or 40) if inv_cfg is not None else 40
        if seed == 0 and inv_cfg is not None:
            seed = int(getattr(inv_cfg, "val_seed", getattr(inv_cfg, "seed", 0)) or 0)

        sampled = self._sample_inverse_val_rows(n_rows=n_rows, seed=seed)
        if sampled is None:
            return {}
        _, prompts, gt_actions = sampled

        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            parse_single_token_action,
        )

        pred_tokens = [self._greedy_decode_action_response(p) for p in prompts]
        matches = []
        for gt, pred in zip(gt_actions, pred_tokens):
            gt_id = parse_single_token_action(gt)
            pred_id = parse_single_token_action(pred)
            matches.append(gt_id >= 0 and pred_id >= 0 and gt_id == pred_id)
        if not matches:
            return {}
        acc = float(sum(matches) / len(matches))
        return {
            "val/inverse_action/greedy_accuracy": acc,
            "val/inverse_action/greedy_n": float(len(matches)),
        }

    def _sample_planning_val_rows(self, n_rows: int, seed: int = 0):
        """Sample val rows with full planning horizon H."""
        try:
            import pandas as pd
        except Exception:
            return None

        val_path = self.config.data.val_files
        if isinstance(val_path, (list, tuple)):
            val_path = val_path[0]
        try:
            df = pd.read_parquet(val_path)
        except Exception:
            return None

        required = {"state", "future_rewards", "future_actions", "horizon"}
        if not required.issubset(set(df.columns)):
            return None

        h_plan = int(self._reward_wm_max_horizon)
        from agent_system.environments.prompts.world_model_planning import (
            cumulative_return_from_rewards,
            format_planning_prompt,
            format_planning_target,
            parse_future_json_list,
            planning_row_valid,
        )

        valid_idx = []
        for idx, row in df.iterrows():
            horizon = int(row.get("horizon", 0))
            future_acts = parse_future_json_list(row.get("future_actions"))
            future_rews = parse_future_json_list(row.get("future_rewards"))
            if planning_row_valid(
                horizon=horizon,
                planning_horizon=h_plan,
                future_actions=future_acts,
                future_rewards=future_rews,
            ):
                valid_idx.append(idx)
        if not valid_idx:
            return None

        valid_df = df.loc[valid_idx].copy()
        rng = torch.Generator().manual_seed(int(seed))
        pick = torch.randperm(len(valid_df), generator=rng)[: min(int(n_rows), len(valid_df))].tolist()
        sub = valid_df.iloc[pick].copy()

        prompts = []
        gt_seqs = []
        target_returns = []
        for _, row in sub.iterrows():
            future_acts = parse_future_json_list(row.get("future_actions"))
            future_rews = parse_future_json_list(row.get("future_rewards"))
            target_return = cumulative_return_from_rewards(future_rews)
            task = str(row.get("instruction", "") or "").strip()
            prompts.append(
                format_planning_prompt(
                    str(row["state"]),
                    target_return=target_return,
                    task=task,
                    horizon=h_plan,
                )
            )
            gt_seqs.append(format_planning_target(future_acts))
            target_returns.append(int(target_return))
        return sub, prompts, gt_seqs, target_returns

    def _compute_planning_greedy_val_accuracy(self, n_rows: int | None = None, seed: int = 0) -> dict:
        plan_cfg = getattr(self.config.trainer, "planning_wm", None)
        h_plan = int(self._reward_wm_max_horizon)
        if n_rows is None:
            n_rows = int(getattr(plan_cfg, "val_table_n", 40) or 40) if plan_cfg is not None else 40
        if seed == 0 and plan_cfg is not None:
            seed = int(getattr(plan_cfg, "val_seed", getattr(plan_cfg, "seed", 0)) or 0)

        sampled = self._sample_planning_val_rows(n_rows=n_rows, seed=seed)
        if sampled is None:
            return {}
        _, prompts, gt_seqs, _ = sampled

        from agent_system.environments.env_package.caged_craftext.action_tokens import (
            parse_action_token_sequence,
        )

        correct_tokens = 0
        total_tokens = 0
        seq_matches = 0
        for prompt, gt in zip(prompts, gt_seqs):
            pred = self._greedy_decode_planning_action_sequence(prompt, num_actions=h_plan)
            gt_toks = parse_action_token_sequence(gt)
            pred_toks = parse_action_token_sequence(pred)
            for j in range(h_plan):
                total_tokens += 1
                if j < len(gt_toks) and j < len(pred_toks) and gt_toks[j] == pred_toks[j]:
                    correct_tokens += 1
            if (
                len(gt_toks) >= h_plan
                and len(pred_toks) >= h_plan
                and gt_toks[:h_plan] == pred_toks[:h_plan]
            ):
                seq_matches += 1
        n = len(gt_seqs)
        if n == 0:
            return {}
        tok_acc = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0
        return {
            "val/planning/greedy_token_accuracy": tok_acc,
            "val/planning/greedy_sequence_accuracy": float(seq_matches / n),
            "val/planning/greedy_n": float(n),
        }

    def _run_and_log_validation(self, tracking: Tracking, step: int, tag: str = "") -> None:
        """Run full val, log metrics/tables to Comet, print to console."""
        label = f" ({tag})" if tag else ""
        if self.device_mesh.get_rank() == 0:
            cap = self._val_max_batches()
            cap_s = f"max {cap} batches" if cap > 0 else "full val set"
            print(f"[val] step={step}{label} metrics ({cap_s})...", flush=True)
        metric = self._evaluate_validation(return_entropy=True)
        if self.device_mesh.get_rank() == 0 and self._inverse_action_wm_enabled:
            metric.update(self._compute_inverse_greedy_val_accuracy())
        if self.device_mesh.get_rank() == 0 and self._planning_wm_enabled:
            metric.update(self._compute_planning_greedy_val_accuracy())
        if self.device_mesh.get_rank() == 0:
            tracking.log(data=metric, step=step)
            inv_parts = [
                f"{k}={v:.4g}"
                for k, v in sorted(metric.items())
                if k.startswith("val/inverse_action/")
            ]
            plan_parts = [
                f"{k}={v:.4g}"
                for k, v in sorted(metric.items())
                if k.startswith("val/planning/")
            ]
            print(
                f"[val] step={step}{label} "
                + " ".join(f"{k}={v:.4g}" for k, v in sorted(metric.items())),
                flush=True,
            )
            if inv_parts:
                print(f"[val] step={step}{label} inverse: " + " ".join(inv_parts), flush=True)
            if plan_parts:
                print(f"[val] step={step}{label} planning: " + " ".join(plan_parts), flush=True)
            print(f"[val] step={step}{label} tables/images...", flush=True)
            self._log_reward_wm_validation_examples(tracking=tracking, step=step)
            if self._inverse_action_wm_enabled:
                self._log_inverse_action_validation_examples(tracking=tracking, step=step)
            if self._planning_wm_enabled:
                self._log_planning_validation_examples(tracking=tracking, step=step)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _log_reward_wm_validation_examples(self, tracking: Tracking, step: int) -> None:
        """
        Reward-WM specific validation artifacts for offline SFT:
        - 1 table (PNG) for N validation rows (default 100)
        - 1 detailed prompt panel (PNG) for a single sample

        Requires val parquet to have at least: prompt, response.
        If extra columns exist (instruction/action_token/reward_q), we use them for nicer tables.
        """
        cfg = getattr(self.config.trainer, "reward_wm_val", None)
        enable = bool(getattr(cfg, "enable", False)) if cfg is not None else False
        if not enable:
            return

        try:
            import pandas as pd
        except Exception:
            print("[reward_wm_val] pandas not available; skipping reward WM tables.", flush=True)
            return

        val_path = self.config.data.val_files
        if isinstance(val_path, (list, tuple)):
            # only first file for artifact logging
            val_path = val_path[0]

        try:
            df = pd.read_parquet(val_path)
        except Exception as e:
            print(f"[reward_wm_val] failed to read val parquet {val_path}: {e}", flush=True)
            return

        if "prompt" not in df.columns or "response" not in df.columns:
            print("[reward_wm_val] val parquet missing prompt/response; skipping.", flush=True)
            return

        # Config defaults
        n_rows = int(getattr(cfg, "table_n", None) or getattr(cfg, "table_n2", 100))
        seed = int(getattr(cfg, "seed", 0))
        stratify = bool(getattr(cfg, "table_stratify_horizon", True))

        # Stratified sample: equal rows per horizon when reward_horizon > 1.
        max_h = max(1, int(self._reward_wm_max_horizon))
        if stratify and max_h > 1 and "horizon" in df.columns:
            import numpy as np

            per_h = max(1, n_rows // max_h)
            remainder = max(0, n_rows - per_h * max_h)
            idx: list = []
            rng_np = np.random.RandomState(seed)
            for h in range(1, max_h + 1):
                pool = df.index[df["horizon"] == h].tolist()
                n_take = per_h + (1 if h <= remainder else 0)
                n_take = min(n_take, len(pool))
                if n_take > 0:
                    chosen = rng_np.choice(pool, size=n_take, replace=False).tolist()
                    idx.extend(chosen)
            sub = df.loc[idx].copy()
        else:
            rng = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(df), generator=rng)[: min(n_rows, len(df))].tolist()
            sub = df.iloc[idx].copy()

        # Greedy autoregressive decode of reward token sequence after each prompt.
        try:
            from collections import Counter

            from agent_system.environments.env_package.caged_craftext.reward_tokens import (
                format_reward_sequence_display,
                parse_reward_token_sequence,
            )
            from agent_system.environments.env_package.caged_craftext.action_tokens import (
                parse_single_token_action,
            )
            from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT
            from agent_system.environments.env_package.caged_craftext.utility import (
                render_world_model_reward_panel,
            )
        except Exception as e:
            print(f"[reward_wm_val] import error: {e}", flush=True)
            return

        prompts = [str(x) for x in sub["prompt"].tolist()]

        horizons = []
        if "horizon" in sub.columns:
            horizons = [max(1, int(x)) for x in sub["horizon"].tolist()]
        else:
            horizons = [max(1, len(str(sub["response"].iloc[i]).strip())) for i in range(len(sub))]

        tasks = []
        if "instruction" in sub.columns:
            tasks = [str(x) for x in sub["instruction"].tolist()]
        elif "task_slug" in sub.columns:
            tasks = [str(x) for x in sub["task_slug"].tolist()]
        else:
            tasks = ["" for _ in prompts]

        import json

        # Prefer rebuilt prompts (full action sequence for h>1) when extra columns exist.
        try:
            from agent_system.environments.prompts.world_model_reward import format_reward_prompt

            rebuilt = []
            for i in range(len(sub)):
                h = horizons[i]
                if (
                    h > 1
                    and "state" in sub.columns
                    and "future_actions" in sub.columns
                ):
                    try:
                        acts = json.loads(str(sub["future_actions"].iloc[i]))
                        if isinstance(acts, list) and len(acts) >= h:
                            rebuilt.append(
                                format_reward_prompt(
                                    state=str(sub["state"].iloc[i]),
                                    action=str(sub["action_token"].iloc[i])
                                    if "action_token" in sub.columns
                                    else str(acts[0]),
                                    task=tasks[i],
                                    horizon=h,
                                    actions=acts[:h],
                                )
                            )
                            continue
                    except Exception:
                        pass
                rebuilt.append(prompts[i])
            prompts = rebuilt
        except Exception:
            pass

        pred_tokens = []
        for prompt, max_h in zip(prompts, horizons):
            pred_tokens.append(self._greedy_decode_reward_response(prompt, max_reward_tokens=max_h))

        # Build display columns
        def _fmt_action_raw(tok_str: str) -> str:
            raw = (tok_str or "").strip().split()[0] if (tok_str or "").strip() else ""
            aid = parse_single_token_action(raw)
            name = ACTION_TO_TEXT[aid] if 0 <= int(aid) < len(ACTION_TO_TEXT) else "?"
            return f"{name}({raw})" if raw else "?"

        def _fmt_action_seq(row_i: int) -> str:
            if "future_actions" in sub.columns:
                raw = sub["future_actions"].iloc[row_i]
                if raw and str(raw).strip():
                    try:
                        acts = json.loads(str(raw))
                        if isinstance(acts, list) and acts:
                            return " → ".join(_fmt_action_raw(str(a)) for a in acts)
                    except Exception:
                        pass
            if "action_token" in sub.columns:
                return _fmt_action_raw(str(sub["action_token"].iloc[row_i]))
            return ""

        actions = [_fmt_action_seq(i) for i in range(len(sub))]

        gt_responses = [str(x).strip() for x in sub["response"].tolist()]
        gt_vals_list = [parse_reward_token_sequence(t) for t in gt_responses]
        pred_vals_list = [parse_reward_token_sequence(t) for t in pred_tokens]

        gt_disp = []
        if "future_rewards" in sub.columns:
            for raw in sub["future_rewards"].tolist():
                try:
                    rewards = json.loads(str(raw))
                    gt_disp.append(format_reward_sequence_display(rewards))
                except Exception:
                    gt_disp.append(str(raw))
        elif gt_vals_list and any(gt_vals_list):
            gt_disp = [
                ", ".join(f"{v} ({ch})" for v, ch in zip(vals, resp))
                if vals
                else resp
                for vals, resp in zip(gt_vals_list, gt_responses)
            ]
        else:
            gt_disp = list(gt_responses)
        pred_disp = [
            ", ".join(f"{v} ({ch})" for v, ch in zip(vals, tok))
            if vals
            else tok
            for vals, tok in zip(pred_vals_list, pred_tokens)
        ]

        def _reward_wm_subset_stats(n: int):
            """Token-level accuracy overall and per horizon on table rows."""
            n = min(n, len(gt_responses))
            correct = 0
            total = 0
            all_gt_vals = []
            all_pred_vals = []
            per_h = {h: [0, 0] for h in range(1, max_h + 1)}  # correct, total
            for i in range(n):
                gts = gt_vals_list[i]
                preds = pred_vals_list[i]
                h = int(horizons[i]) if i < len(horizons) else len(gts)
                all_gt_vals.extend(gts)
                all_pred_vals.extend(preds)
                for j, g in enumerate(gts):
                    total += 1
                    if 1 <= h <= max_h:
                        per_h[h][1] += 1
                    if j < len(preds) and preds[j] == g:
                        correct += 1
                        if 1 <= h <= max_h:
                            per_h[h][0] += 1
            acc = (correct / total) if total > 0 else 0.0
            per_h_acc = {
                h: (per_h[h][0] / per_h[h][1] if per_h[h][1] > 0 else float("nan"))
                for h in range(1, max_h + 1)
            }

            def _dist_str(vals):
                if not vals:
                    return "n/a"
                c = Counter(vals)
                return " ".join(f"{v}:{100 * c.get(v, 0) / len(vals):.0f}%" for v in (-1, 0, 1, 2))

            return acc, per_h_acc, _dist_str(all_gt_vals), _dist_str(all_pred_vals)

        def _reward_wm_table_title(n: int) -> str:
            acc, per_h_acc, true_dist, pred_dist = _reward_wm_subset_stats(n)
            h_parts = [
                f"h{h}={per_h_acc[h]:.0%}" if per_h_acc[h] == per_h_acc[h] else f"h{h}=n/a"
                for h in range(1, max_h + 1)
            ]
            sample_note = (
                f"stratified h=1..{max_h}, ~{n // max_h}/horizon"
                if stratify and max_h > 1 and "horizon" in df.columns
                else "random sample"
            )
            return (
                f"Reward WM validation (N={n}, acc={acc:.1%}, {sample_note})\n"
                f"per-horizon [{' '.join(h_parts)}]\n"
                f"true [{true_dist}]  pred [{pred_dist}]"
            )

        # Render 2 table images.
        def _render_table(rows, title: str, out_path: str) -> None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_h = max(2.5, 0.32 * len(rows) + 2.2)
            fig, ax = plt.subplots(figsize=(16, fig_h), dpi=140)
            ax.axis("off")
            col_labels = ["horizon", "task", "actions (a_t..)", "rewards (true)", "rewards (pred)"]
            table = ax.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.15)
            ax.set_title(title, fontsize=10, pad=12)
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        def _rows(n: int):
            n = min(n, len(prompts))
            out_rows = []
            for i in range(n):
                out_rows.append([
                    str(horizons[i]) if i < len(horizons) else "",
                    tasks[i],
                    actions[i],
                    gt_disp[i],
                    pred_disp[i],
                ])
            return out_rows

        p_table = os.path.join(tempfile.gettempdir(), f"val_reward_wm_table{n_rows}_step{step}.png")
        _render_table(_rows(len(sub)), _reward_wm_table_title(len(sub)), p_table)
        tracking.log_image(p_table, step=step, name=f"validation_reward_wm_table_{n_rows}")

        acc_table, per_h_acc, _, _ = _reward_wm_subset_stats(len(sub))
        table_metrics = {
            f"val/table_{n_rows}_accuracy": float(acc_table),
        }
        for h in range(1, max_h + 1):
            if per_h_acc[h] == per_h_acc[h]:
                table_metrics[f"val/table_{n_rows}_accuracy_h{h}"] = float(per_h_acc[h])
        tracking.log(data=table_metrics, step=step)

        # Detailed single example panel (full prompt + action + gt/pred).
        pick = min(len(prompts) // 2, len(prompts) - 1)
        panel = render_world_model_reward_panel(
            step=int(pick),
            world_model_input=prompts[pick],
            policy_action=actions[pick],
            ground_truth_reward=gt_disp[pick],
            world_model_output=pred_tokens[pick],
        )
        try:
            from PIL import Image

            p_ex = os.path.join(tempfile.gettempdir(), f"val_reward_wm_example_step{step}.png")
            Image.fromarray(panel).save(p_ex)
            tracking.log_image(p_ex, step=step, name="validation_reward_wm_example")
        except Exception as e:
            print(f"[reward_wm_val] failed to save/log example panel: {e}", flush=True)

    def _log_inverse_action_validation_examples(self, tracking: Tracking, step: int) -> None:
        """
        Inverse-action WM validation artifacts:
        - table (PNG): prompt, ground-truth action, model prediction
        - detailed panel (PNG) for one sample
        """
        inv_cfg = getattr(self.config.trainer, "inverse_action_wm", None)
        enable = self._inverse_action_wm_enabled and (
            bool(getattr(inv_cfg, "val_enable", True)) if inv_cfg is not None else False
        )
        if not enable:
            return

        n_rows = int(getattr(inv_cfg, "val_table_n", 40) or 40)
        seed = int(getattr(inv_cfg, "val_seed", getattr(inv_cfg, "seed", 0)) or 0)
        sampled = self._sample_inverse_val_rows(n_rows=n_rows, seed=seed)
        if sampled is None:
            print("[inverse_action_val] no valid inverse transitions in val parquet.", flush=True)
            return
        sub, prompts, gt_actions = sampled

        try:
            from agent_system.environments.env_package.caged_craftext.action_tokens import (
                format_single_token_action_display,
                parse_single_token_action,
            )
            from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT
            from agent_system.environments.env_package.caged_craftext.utility import (
                render_world_model_transition_panel,
            )
        except Exception as e:
            print(f"[inverse_action_val] import error: {e}", flush=True)
            return

        gt_disp = []
        for act_raw in gt_actions:
            aid = parse_single_token_action(act_raw)
            name = ACTION_TO_TEXT[aid] if 0 <= int(aid) < len(ACTION_TO_TEXT) else "?"
            gt_disp.append(f"{name}({act_raw})")

        pred_tokens = [self._greedy_decode_action_response(p) for p in prompts]
        pred_disp = []
        for tok in pred_tokens:
            disp, _ = format_single_token_action_display(tok)
            pred_disp.append(disp or tok)

        def _truncate(text: str, max_len: int = 220) -> str:
            text = str(text or "").replace("\n", " ")
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."

        matches = []
        for gt, pred in zip(gt_actions, pred_tokens):
            gt_id = parse_single_token_action(gt)
            pred_id = parse_single_token_action(pred)
            matches.append(gt_id >= 0 and pred_id >= 0 and gt_id == pred_id)
        acc = float(sum(matches) / len(matches)) if matches else 0.0

        def _render_table(rows, title: str, out_path: str) -> None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_h = max(2.5, 0.32 * len(rows) + 2.2)
            fig, ax = plt.subplots(figsize=(16, fig_h), dpi=140)
            ax.axis("off")
            col_labels = ["prompt", "action (true)", "action (pred)", "match"]
            table = ax.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.15)
            ax.set_title(title, fontsize=10, pad=12)
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        table_rows = []
        for i in range(len(sub)):
            table_rows.append([
                _truncate(prompts[i]),
                gt_disp[i],
                pred_disp[i],
                "✓" if matches[i] else "✗",
            ])

        title = (
            f"Inverse-action WM validation (N={len(sub)}, acc={acc:.1%})\n"
            f"(s_t, s_{{t+1}}) -> a_t — greedy decode"
        )
        p_table = os.path.join(
            tempfile.gettempdir(), f"val_inverse_action_table{len(sub)}_step{step}.png"
        )
        _render_table(table_rows, title, p_table)
        tracking.log_image(p_table, step=step, name=f"validation_inverse_action_table_{len(sub)}")

        pick = min(len(prompts) // 2, len(prompts) - 1)
        panel = render_world_model_transition_panel(
            step=int(pick),
            world_model_input=prompts[pick],
            policy_action=gt_disp[pick],
            world_model_output=pred_disp[pick] or pred_tokens[pick],
        )
        try:
            from PIL import Image

            p_ex = os.path.join(tempfile.gettempdir(), f"val_inverse_action_example_step{step}.png")
            Image.fromarray(panel).save(p_ex)
            tracking.log_image(p_ex, step=step, name="validation_inverse_action_example")
        except Exception as e:
            print(f"[inverse_action_val] failed to save/log example panel: {e}", flush=True)

    def _log_planning_validation_examples(self, tracking: Tracking, step: int) -> None:
        """Planning WM val table: prompt, target return, gt/pred action trajectories."""
        plan_cfg = getattr(self.config.trainer, "planning_wm", None)
        enable = self._planning_wm_enabled and (
            bool(getattr(plan_cfg, "val_enable", True)) if plan_cfg is not None else False
        )
        if not enable:
            return

        n_rows = int(getattr(plan_cfg, "val_table_n", 40) or 40)
        seed = int(getattr(plan_cfg, "val_seed", getattr(plan_cfg, "seed", 0)) or 0)
        h_plan = int(self._reward_wm_max_horizon)
        sampled = self._sample_planning_val_rows(n_rows=n_rows, seed=seed)
        if sampled is None:
            print("[planning_val] no full-horizon rows in val parquet.", flush=True)
            return
        sub, prompts, gt_seqs, target_returns = sampled

        try:
            from agent_system.environments.env_package.caged_craftext.action_tokens import (
                parse_action_token_sequence,
            )
            from agent_system.environments.env_package.caged_craftext.projection import ACTION_TO_TEXT
            from agent_system.environments.env_package.caged_craftext.action_tokens import (
                parse_single_token_action,
            )
        except Exception as e:
            print(f"[planning_val] import error: {e}", flush=True)
            return

        def _fmt_seq(seq: str) -> str:
            toks = parse_action_token_sequence(seq)
            parts = []
            for t in toks:
                aid = parse_single_token_action(t)
                name = ACTION_TO_TEXT[aid] if 0 <= int(aid) < len(ACTION_TO_TEXT) else "?"
                parts.append(f"{name}({t})")
            return " → ".join(parts) if parts else seq

        pred_seqs = [
            self._greedy_decode_planning_action_sequence(p, num_actions=h_plan) for p in prompts
        ]

        def _truncate(text: str, max_len: int = 200) -> str:
            text = str(text or "").replace("\n", " ")
            return text if len(text) <= max_len else text[: max_len - 3] + "..."

        seq_matches = []
        for gt, pred in zip(gt_seqs, pred_seqs):
            gt_t = parse_action_token_sequence(gt)
            pr_t = parse_action_token_sequence(pred)
            seq_matches.append(
                len(gt_t) >= h_plan and len(pr_t) >= h_plan and gt_t[:h_plan] == pr_t[:h_plan]
            )
        seq_acc = float(sum(seq_matches) / len(seq_matches)) if seq_matches else 0.0

        def _render_table(rows, title: str, out_path: str) -> None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_h = max(2.5, 0.32 * len(rows) + 2.2)
            fig, ax = plt.subplots(figsize=(16, fig_h), dpi=140)
            ax.axis("off")
            col_labels = ["R̂", "actions (true)", "actions (pred)", "match", "prompt"]
            table = ax.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.15)
            ax.set_title(title, fontsize=10, pad=12)
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        table_rows = []
        for i in range(len(sub)):
            table_rows.append([
                str(target_returns[i]),
                _fmt_seq(gt_seqs[i]),
                _fmt_seq(pred_seqs[i]),
                "✓" if seq_matches[i] else "✗",
                _truncate(prompts[i]),
            ])
        title = (
            f"Planning WM validation (N={len(sub)}, H={h_plan}, seq_acc={seq_acc:.1%})\n"
            f"return-conditioned action trajectory (DT-style)"
        )
        p_table = os.path.join(
            tempfile.gettempdir(), f"val_planning_table{len(sub)}_step{step}.png"
        )
        _render_table(table_rows, title, p_table)
        tracking.log_image(p_table, step=step, name=f"validation_planning_table_{len(sub)}")

        try:
            from agent_system.environments.env_package.caged_craftext.utility import (
                render_world_model_planning_panel,
            )
            from agent_system.environments.prompts.world_model_planning import (
                describe_planning_prompt_schema,
            )
            from PIL import Image

            pick = min(len(prompts) // 2, len(prompts) - 1)
            panel = render_world_model_planning_panel(
                step=int(pick),
                prompt_schema=describe_planning_prompt_schema(horizon=h_plan),
                world_model_input=prompts[pick],
                ground_truth_plan=_fmt_seq(gt_seqs[pick]),
                world_model_output=_fmt_seq(pred_seqs[pick]) or pred_seqs[pick],
                target_return=int(target_returns[pick]) if pick < len(target_returns) else None,
                horizon=h_plan,
            )
            p_ex = os.path.join(tempfile.gettempdir(), f"val_planning_example_step{step}.png")
            Image.fromarray(panel).save(p_ex)
            tracking.log_image(p_ex, step=step, name="validation_planning_example")
        except Exception as e:
            print(f"[planning_val] failed to save/log example panel: {e}", flush=True)

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        val_every_steps = int(getattr(self.config.trainer, "val_every_steps", 0) or 0)
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        if rank == 0 and val_every_steps > 0:
            print(f"Validation every {val_every_steps} train steps (+ initial at step 0)", flush=True)

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        if rank == 0 and val_every_steps > 0:
            self._run_and_log_validation(tracking=tracking, step=0, tag="initial")

        for epoch in range(self.config.trainer.total_epochs):
            if self.train_batch_sampler is not None:
                self.train_batch_sampler.set_epoch(epoch=epoch)
            elif self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0
            ):
                global_step += 1
                data = self._to_tensor_dict(data)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                    # Periodic validation (optional).
                    if val_every_steps > 0 and (global_step % val_every_steps == 0):
                        self._run_and_log_validation(tracking=tracking, step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    if rank == 0:
                        self._run_and_log_validation(
                            tracking=tracking, step=global_step, tag="final"
                        )

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

            # validation at epoch end
            if rank == 0:
                self._run_and_log_validation(
                    tracking=tracking, step=global_step, tag="epoch end"
                )

            # save checkpoint
            self.save_checkpoint(step=global_step)


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
