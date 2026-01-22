import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# NOTE: these env vars must be set before importing jax in most setups
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from notebooks.simple_agent import SimpleAgent

from agent_system.environments import make_envs
from agent_system.environments.env_package.caged_craftext.envs import CagedCraftextWorker
from agent_system.environments.env_package.caged_craftext.projection import craftext_projection
from agent_system.environments.env_package.caged_craftext.utility import VisualizerWithLLM
from omegaconf import OmegaConf


@dataclass
class CheckpointChoice:
    checkpoint_path: str
    base_model_path: str
    lora_rank: int
    lora_alpha: int
    target_modules: Optional[str]


def _pick_checkpoint_model(checkpoint_path: str) -> CheckpointChoice:
    """
    Decide which model to load for inference:
    - prefer merged HF export if present
    - else base model + LoRA adapter if present
    - else fallback to base model only
    """
    hf_merged_path = os.path.join(checkpoint_path, "actor", "actor_hf_merged")
    if not os.path.exists(hf_merged_path):
        hf_merged_path = os.path.join(checkpoint_path, "actor_hf_merged")

    lora_path = os.path.join(checkpoint_path, "actor", "lora_adapter")

    if os.path.exists(hf_merged_path):
        return CheckpointChoice(
            checkpoint_path=checkpoint_path,
            base_model_path=hf_merged_path,
            lora_rank=0,
            lora_alpha=0,
            target_modules=None,
        )

    if os.path.exists(lora_path):
        return CheckpointChoice(
            checkpoint_path=checkpoint_path,
            base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            lora_rank=64,
            lora_alpha=64,
            target_modules="all-linear",
        )

    return CheckpointChoice(
        checkpoint_path=checkpoint_path,
        base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
        lora_rank=0,
        lora_alpha=0,
        target_modules=None,
    )

def evaluation_caged_craftext(config) -> None:
    """
    Валидация обученной модели на Caged Craftext с визуализацией.
    
    Args:
        config: Конфигурация с полями:
            - run_name: Имя запуска (например, "run_ppo_qwen2.5_1.5b_caged_craftext_budgetary_water_20250120-120000")
            - global_step: Номер шага обучения (например, 100)
            - craftext_settings: Настройки датасета (например, "achievements_safe_budget_drink")
            - num_episodes: Количество эпизодов для валидации (по умолчанию 10)
            - output_dir: Директория для сохранения GIF (по умолчанию "./runs")
    """

    if getattr(config, "cuda_visible_devices", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    simple_cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "caged_craftext/CagedCraftextEnv",
                "craftext_settings": config.craftext_settings,
                "seed": config.seed,
                "rollout": {"n": 1},
                "max_steps": config.max_steps,
                "resources_per_worker": {  # required by make_envs
                    "num_cpus": 0.1,
                    "num_gpus": 0,
                },
                "history_length": 0,
                "observation_type": "ascii",
            },
            "data": {
                "val_batch_size": 1,
                "train_batch_size": 1,
            },
        }
    )

    _, val_envs = make_envs(simple_cfg)

    # --- checkpoint resolution ---
    checkpoint_path = config.checkpoint_path
    if checkpoint_path is None:
        if config.run_name is None or config.global_step is None:
            raise ValueError("Either --checkpoint_path or both --run_name and --global_step must be provided.")
        checkpoint_path = os.path.join(
            "checkpoints",
            config.project_name,
            config.run_name,
            f"global_step_{config.global_step}",
        )

    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Checkpoint: {checkpoint_path} (exists={os.path.exists(checkpoint_path)})")

    choice = _pick_checkpoint_model(checkpoint_path)
    print("Inference model choice:")
    print(f"  base_model_path: {choice.base_model_path}")
    print(f"  use_lora: {choice.lora_rank > 0}")

    agent = SimpleAgent(
        checkpoint_path=choice.checkpoint_path,
        base_model_path=choice.base_model_path,
        lora_rank=choice.lora_rank,
        lora_alpha=choice.lora_alpha,
        target_modules=choice.target_modules,
        max_response_length=512,
        do_sample=config.do_sample,
        temperature=config.temperature,
    )

    # temp worker only to get env_params + wrapper for VisualizerWithLLM
    # IMPORTANT: do NOT pass encode_form as a string (worker expects EncodeForm Enum)
    temp_worker = CagedCraftextWorker(
        seed=config.seed,
        env_kwargs={
            "config_name": config.craftext_settings,
            "observation_type": "ascii",
        },
    )
    env_params = temp_worker.env_params
    env_for_viz = temp_worker.wrapper  # Получаем объект env для визуализатора

    # Инициализируем визуализатор
    visualizer = VisualizerWithLLM(
        env=env_for_viz,
        env_params=env_params,
        pixel_render_size=6,
        llm_panel_width=2000,  # Ширина панели с промптом и выводом
        banner_height=250,
        font_size=40,
        policy_panel_height=250,
    )

    # Создаем директорию для сохранения результатов
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    num_episodes = config.num_episodes

    for i in range(num_episodes):
        # Проходим по эпизоду и визуализируем
        obs, info = val_envs.reset({})
        visualizer.clear_frames()

        step = 0
        done = False
        total_reward = 0.0
        total_cost = 0.0  # Для Caged Craftext - накопление cost
        max_steps = config.max_steps

        # Получаем доступ к воркеру для получения env_state
        # val_envs содержит список воркеров в self._workers
        # Access first ray worker to fetch raw env_state for pixel rendering
        import ray
        worker = val_envs.envs._workers[0]
        _ = ray.get(worker.get_state.remote())  # force state materialization

        instruction_text = info[0].get('instruction', 'Unknown instruction')
        constraint_text = info[0].get('constraint', '')  # Для Caged Craftext

        while not done and step < max_steps:
            step += 1  # Увеличиваем счетчик шага ПЕРЕД визуализацией
            
            # Получаем промпт и вывод модели используя новый метод
            prediction_result = agent.predict_with_prompt(obs['text'][0])
            prompt = prediction_result['prompt']
            model_output = prediction_result['model_output']
            action_text = prediction_result['action']
            
            if config.verbose:
                print(f"Step {step}:")
                print(f"  Instruction: {instruction_text[:100]}")
                if constraint_text:
                    print(f"  Constraint: {constraint_text[:100]}")
                print(f"  Action: {action_text[:100]}")
            
            # Проверяем валидность действия
            action_ids, valids = craftext_projection([action_text])
            is_valid = bool(valids[0])
            action_id = action_ids[0] if valids[0] else -1

            if config.verbose:
                print(f"  Action ID: {action_id}, Valid: {is_valid}")
            
            # Выполняем действие в среде
            next_obs, rewards, dones, infos = val_envs.step([action_text])
            
            reward = float(rewards[0])
            done = bool(dones[0])
            total_reward += reward
            
            # Извлекаем cost из info (для Caged Craftext)
            cost = float(infos[0].get("cost", 0.0) or 0.0)
            episode_cost = infos[0].get("episode_cost", None)
            if episode_cost is None:
                total_cost += cost
            else:
                total_cost = float(episode_cost)

            # constraint is constant for the episode, but keep it updated if present
            constraint_text = infos[0].get("constraint", constraint_text) or constraint_text
            
            # Получаем новое состояние для визуализации ПОСЛЕ применения действия
            new_state = ray.get(worker.get_state.remote())
            env_state_for_viz = new_state.env_state
            
            # Получаем дополнительные данные из info
            success_rate = getattr(new_state, 'success_rate', 0.0)
            instruction_done = float(getattr(new_state, 'instruction_done', 0.0))
            
            # Визуализируем шаг (состояние ПОСЛЕ применения действия)
            # Для Caged Craftext добавляем информацию о cost
            visualizer.render_step(
                env_state=env_state_for_viz,
                action=action_id,
                prompt=prompt[:2000],  # Увеличена длина для лучшего отображения
                model_output=model_output[:1000],  # Увеличена длина для лучшего отображения
                step_num=step,
                reward=total_reward,
                instruction=instruction_text[:200],  # Увеличена длина инструкции
                success_rate=success_rate,
                instruction_done=instruction_done,
                policy_distribution=None,  # Можно добавить распределение политики, если доступно
                value=None,
                action_text=action_text[:200],
                is_action_valid=is_valid,
                cost=total_cost,  # Добавляем cost для визуализации
                constraint=constraint_text[:200] if constraint_text else None,  # Добавляем constraint
            )
            
            obs = next_obs
            current_state = new_state

        print(f"\nЭпизод {i+1} завершен:")
        print(f"  Шагов: {step}")
        print(f"  Общая награда: {total_reward:.4f}")
        print(f"  Общая стоимость (cost): {total_cost:.4f}")
        print(f"  Success rate: {success_rate:.4f}")
        print(f"  Instruction done: {instruction_done:.2f}")

        # Сохраняем GIF
        run_name_for_file = config.run_name or os.path.basename(os.path.dirname(checkpoint_path))
        gif_filename = os.path.join(output_dir, f"{run_name_for_file}_ep{i+1:03d}_sr{instruction_done:.2f}_cost{total_cost:.2f}.gif")
        visualizer.save_gif(gif_filename, duration=500)
        print(f"  Визуализация сохранена: {gif_filename}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Валидация обученной модели на Caged Craftext с визуализацией")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Полный путь к чекпоинту (global_step_*)")
    parser.add_argument("--project_name", type=str, default="verl_agent_caged_craftext", help="trainer.project_name")
    parser.add_argument("--run_name", type=str, default=None, help="trainer.experiment_name (если checkpoint_path не задан)")
    parser.add_argument("--global_step", type=int, default=None, help="номер global_step (если checkpoint_path не задан)")
    parser.add_argument("--craftext_settings", type=str, default="achievements_safe_budget_drink",
                        help="Настройки датасета (по умолчанию: achievements_safe_budget_drink)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Количество эпизодов для валидации (по умолчанию: 10)")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Директория для сохранения GIF (по умолчанию: ./runs)")
    parser.add_argument("--max_steps", type=int, default=50, help="максимум шагов в эпизоде")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true", help="sampling like in training val_kwargs.do_sample")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help='e.g. "0" or "0,1"')
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluation_caged_craftext(args)
