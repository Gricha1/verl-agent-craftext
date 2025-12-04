#!/usr/bin/env python3
"""
Скрипт для конвертации HuggingFace LoRA чекпоинта (из SFT обучения) в FSDP формат для RL обучения.

Использование:
    python3 scripts/convert_sft_to_fsdp.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_path /path/to/sft_checkpoint/global_step_N \
        --output_path /path/to/output_fsdp_checkpoint_global_step_0 \
        --lora_rank 64 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --num_gpus 1
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import json

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager
from verl.utils.distributed import initialize_global_process_group


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HuggingFace LoRA checkpoint to FSDP format")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path (HuggingFace)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for FSDP checkpoint")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--target_modules", type=str, default="all-linear", help="Target modules for LoRA")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Инициализация distributed
    # Для корректной работы FSDP всегда используем torchrun (даже для одного GPU)
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Скрипт должен быть запущен через torchrun. "
            "Используйте: torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/convert_sft_to_fsdp.py ..."
        )
    
    if not dist.is_initialized():
        initialize_global_process_group()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Конвертация чекпоинта:")
        print(f"  Base model: {args.base_model}")
        print(f"  Adapter path: {args.adapter_path}")
        print(f"  Output path: {args.output_path}")
        print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
        print(f"  World size: {world_size}")
    
    # Проверка существования adapter_path
    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    
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
        
        # Перемещаем модель на GPU перед применением LoRA (для Flash Attention)
        if torch.cuda.is_available():
            model = model.to(torch.cuda.current_device())
            if rank == 0:
                print("Модель перемещена на GPU")
        
        # Применяем LoRA
        if os.path.exists(os.path.join(args.adapter_path, "adapter_model.safetensors")) or \
           os.path.exists(os.path.join(args.adapter_path, "adapter_model.bin")):
            # Загружаем обученный LoRA адаптер
            model = PeftModel.from_pretrained(model, args.adapter_path)
            if rank == 0:
                print(f"Загружен LoRA адаптер из {args.adapter_path}")
        else:
            # Создаем новый LoRA адаптер (если адаптер не найден, создаем пустой)
            if rank == 0:
                print(f"Адаптер не найден, создаем новый LoRA адаптер")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules.split(",") if "," in args.target_modules else args.target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
    
    # Убеждаемся, что все параметры имеют одинаковый dtype (bfloat16)
    # Это критично для FSDP, который требует uniform dtype для всех параметров в одном модуле
    if rank == 0:
        print("Приведение всех параметров к bfloat16 для совместимости с FSDP...")
    
    dtype_changes = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            with torch.no_grad():
                param.data = param.data.to(torch.bfloat16)
            dtype_changes += 1
            if rank == 0 and dtype_changes <= 5:  # Показываем только первые 5 для краткости
                print(f"  Приведен {name} к bfloat16 (был {param.dtype})")
    
    if rank == 0:
        if dtype_changes > 0:
            print(f"  Всего приведено {dtype_changes} параметров к bfloat16")
        else:
            print("  Все параметры уже в bfloat16")
    
    # НЕ объединяем LoRA веса - сохраняем модель с LoRA структурой для RL обучения
    # В RL обучении модель инициализируется с LoRA, поэтому чекпоинт должен содержать LoRA структуру
    is_lora_model = isinstance(model, PeftModel)
    if rank == 0:
        if is_lora_model:
            print("Модель сохранена с LoRA структурой (не объединяем веса)")
        else:
            print("Модель без LoRA структуры")
    
    # Настройка FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    # Получаем wrap policy (для LoRA нужна специальная политика)
    wrap_policy_config = {"min_num_params": 0}
    auto_wrap_policy = get_fsdp_wrap_policy(model, config=wrap_policy_config, is_lora=is_lora_model)
    
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
    
    # Создаем фиктивный оптимизатор и scheduler для сохранения чекпоинта
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Создаем выходную директорию
    output_dir = os.path.join(args.output_path, "actor")
    os.makedirs(output_dir, exist_ok=True)
    
    if rank == 0:
        print(f"Сохранение FSDP чекпоинта в {output_dir}")
    
    # Создаем checkpoint manager
    checkpoint_manager = FSDPCheckpointManager(
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        processing_class=tokenizer,
        checkpoint_contents=["model", "optimizer", "extra"],
    )
    # Сохраняем чекпоинт
    checkpoint_manager.save_checkpoint(local_path=output_dir, global_step=0)
    
    # Сохраняем токенизатор и конфигурацию на rank 0
    if rank == 0:
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        
        # Копируем файлы токенизатора из adapter_path, если они есть
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", 
                          "vocab.json", "merges.txt", "added_tokens.json"]
        for file in tokenizer_files:
            src = os.path.join(args.adapter_path, file)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, output_dir)
        
        print(f"Чекпоинт успешно сохранен в {output_dir}")
    
    dist.barrier()
    
    if rank == 0:
        print("Конвертация завершена!")


if __name__ == "__main__":
    main()

