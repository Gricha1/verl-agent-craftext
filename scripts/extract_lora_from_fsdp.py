#!/usr/bin/env python3
"""
Скрипт для извлечения LoRA адаптера из FSDP чекпоинта для использования в vLLM инференсе.

Использование:
    python3 scripts/extract_lora_from_fsdp.py \
        --checkpoint_path /path/to/fsdp_checkpoint/global_step_0 \
        --output_path /path/to/output/lora_adapter \
        --lora_rank 64 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --base_model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
import json
from collections import OrderedDict

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, layered_summon_lora_params
from verl.utils.distributed import initialize_global_process_group
from verl.utils.py_functional import convert_to_regular_types


def parse_args():
    parser = argparse.ArgumentParser(description="Extract LoRA adapter from FSDP checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to FSDP checkpoint (global_step_N)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for LoRA adapter")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path (HuggingFace)")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--target_modules", type=str, default="all-linear", help="Target modules for LoRA")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Инициализация distributed
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Скрипт должен быть запущен через torchrun. "
            "Используйте: torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/extract_lora_from_fsdp.py ..."
        )
    
    if not dist.is_initialized():
        initialize_global_process_group()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Извлечение LoRA адаптера из FSDP чекпоинта:")
        print(f"  Checkpoint: {args.checkpoint_path}")
        print(f"  Output: {args.output_path}")
        print(f"  Base model: {args.base_model}")
        print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    
    # Проверка существования чекпоинта
    actor_path = os.path.join(args.checkpoint_path, "actor")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    
    # Инициализация device mesh
    if torch.cuda.is_available():
        device_mesh = init_device_mesh("cuda", (world_size,))
    else:
        device_mesh = init_device_mesh("cpu", (world_size,))
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка конфигурации модели
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    
    # Инициализация модели с LoRA
    init_context = get_init_weight_context_manager(use_meta_tensor=False, mesh=device_mesh)
    
    with init_context():
        # Загружаем базовую модель
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=args.trust_remote_code,
        )
        
        # Перемещаем модель на GPU
        if torch.cuda.is_available():
            model = model.to(torch.cuda.current_device())
        
        # Применяем LoRA (создаем структуру)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(",") if "," in args.target_modules else args.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    # Настройка FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    # Получаем wrap policy для LoRA
    wrap_policy_config = {"min_num_params": 0}
    auto_wrap_policy = get_fsdp_wrap_policy(model, config=wrap_policy_config, is_lora=True)
    
    # Обертываем модель в FSDP
    fsdp_model = FSDP(
        model,
        cpu_offload=None,
        param_init_fn=None,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh,
        forward_prefetch=False,
    )
    
    if rank == 0:
        print("Модель обернута в FSDP")
    
    # Создаем фиктивный оптимизатор и scheduler для загрузки чекпоинта
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Создаем checkpoint manager
    checkpoint_manager = FSDPCheckpointManager(
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        processing_class=tokenizer,
        checkpoint_contents=["model", "optimizer", "extra"],
    )
    
    # Загружаем чекпоинт
    if rank == 0:
        print(f"Загрузка FSDP чекпоинта из {actor_path}")
    
    checkpoint_manager.load_checkpoint(local_path=actor_path)
    
    if rank == 0:
        print("Чекпоинт загружен")
    
    # Извлекаем LoRA параметры
    if rank == 0:
        print("Извлечение LoRA параметров...")
    
    if torch.cuda.is_available():
        fsdp_model = fsdp_model.cuda()
    
    lora_params = layered_summon_lora_params(fsdp_model)
    
    # Сохраняем LoRA адаптер
    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        
        # Сохраняем LoRA веса
        save_file(lora_params, os.path.join(args.output_path, "adapter_model.safetensors"))
        
        # Сохраняем конфигурацию LoRA
        peft_config = {
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules.split(",") if "," in args.target_modules else [args.target_modules],
            "bias": "none",
        }
        
        with open(os.path.join(args.output_path, "adapter_config.json"), "w", encoding='utf-8') as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        
        print(f"LoRA адаптер сохранен в {args.output_path}")
        print(f"  Размер адаптера: {sum(p.numel() * 2 for p in lora_params.values()) / 1024**2:.2f} MB")
    
    dist.barrier()
    
    if rank == 0:
        print("Извлечение завершено!")


if __name__ == "__main__":
    main()

