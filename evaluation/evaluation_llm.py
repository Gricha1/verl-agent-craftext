import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from notebooks.simple_agent import SimpleAgent

# Пример использования VisualizerWithLLM
from agent_system.environments.env_package.craftext.utility import VisualizerWithLLM
from agent_system.environments.env_package.craftext.projection import craftext_projection
import jax
import ray

# Создаем визуализатор
# Для получения env_params нужно обратиться к воркеру
# Здесь мы создаем временный воркер для получения env_params
from agent_system.environments.env_package.craftext.envs import CraftextWorker


# Создаем простую среду (один env)
from agent_system.environments import make_envs
from omegaconf import OmegaConf, open_dict

# Создаем минимальный конфиг для среды
simple_cfg = OmegaConf.create({
    "env": {
        "env_name": "craftext/CraftextEnv",
        "craftext_settings": "achievements_collect_wood",
        "seed": 42,
        "rollout": {"n": 1},
        "max_steps": 100,
        "resources_per_worker": {  # Ресурсы для каждого env worker (обязательный параметр для make_envs)
            "num_cpus": 0.1,
            "num_gpus": 0,
        },
        "history_length": 1,
        "observation_type": "text",
        # "observation_type": "ascii",
    },
    "data": {
        "val_batch_size": 1,
        "train_batch_size": 1,
    },
})

_, val_envs = make_envs(simple_cfg)


# Пример использования SimpleAgent

# Создаем Agent
checkpoint_path = "/home/n.sorokin/verl-agent/checkpoints/verl_agent_craftext/run_gigpo_default_collect_wood_20251110-160909/global_step_500"
# checkpoint_path = "/home/n.sorokin/verl-agent/checkpoints/verl_agent_craftext/run_sft_qwen2.5_1.5b_fsdp_achievements_wood_20251203-194330/global_step_0"
# checkpoint_path = ""

# Проверяем наличие чекпоинта и LoRA адаптера
import os
print(f"Проверка чекпоинта: {checkpoint_path}")
print(f"  Чекпоинт существует: {os.path.exists(checkpoint_path)}")

lora_path = os.path.join(checkpoint_path, "actor", "lora_adapter")
print(f"  LoRA путь: {lora_path}")
print(f"  LoRA существует: {os.path.exists(lora_path)}")

if os.path.exists(lora_path):
    adapter_files = os.listdir(lora_path)
    print(f"  LoRA файлы: {adapter_files}")

agent = SimpleAgent(
    checkpoint_path=checkpoint_path,
    base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank=64,
    lora_alpha=64,
    target_modules="all-linear",
    max_response_length=512,
    do_sample=True,  # ИСПРАВЛЕНО: должно совпадать с обучением (val_kwargs.do_sample=True)
    temperature=0.4,  # ИСПРАВЛЕНО: должно совпадать с обучением (val_kwargs.temperature=0.4)
)

# Проверяем, что LoRA загружен
print(f"\nLoRA загружен: {agent.use_lora}")
print(f"LoRA имя: {agent.lora_name}")


# Создаем временный воркер для получения env_params (используется только для инициализации визуализатора)
temp_worker = CraftextWorker(seed=42, env_kwargs=simple_cfg.env.env_kwargs if hasattr(simple_cfg.env, 'env_kwargs') else {})
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

for i in range(10):
    # Проходим по эпизоду и визуализируем
    obs, info = val_envs.reset({})
    visualizer.clear_frames()

    step = 0
    done = False
    total_reward = 0.0
    max_steps = 20

    # Получаем доступ к воркеру для получения env_state
    # val_envs содержит список воркеров в self._workers
    worker = val_envs.envs._workers[0]  # Берем первый воркер
    current_state = ray.get(worker.get_state.remote())  # Получаем текущее состояние

    instruction_text = info[0].get('instruction', 'Unknown instruction')

    while not done and step < max_steps:
        step += 1  # Увеличиваем счетчик шага ПЕРЕД визуализацией
        
        # Получаем промпт и вывод модели используя новый метод
        prediction_result = agent.predict_with_prompt(obs['text'][0])
        prompt = prediction_result['prompt']
        model_output = prediction_result['model_output']
        action_text = prediction_result['action']
        
        print(f"Prompt: \n {prompt}")
        print()
        print(f"Action text: \n {action_text}")
        print()
        
        # Проверяем валидность действия
        action_ids, valids = craftext_projection([action_text])
        is_valid = bool(valids[0])
        action_id = action_ids[0] if valids[0] else -1

        print(f"Action ID: {action_id}, Valid: {is_valid}")
        print()
        
        # Выполняем действие в среде
        next_obs, rewards, dones, infos = val_envs.step([action_text])
        
        reward = float(rewards[0])
        done = bool(dones[0])
        total_reward += reward
        
        # Получаем новое состояние для визуализации ПОСЛЕ применения действия
        new_state = ray.get(worker.get_state.remote())
        env_state_for_viz = new_state.env_state
        
        # Получаем дополнительные данные из info
        success_rate = getattr(new_state, 'success_rate', 0.0)
        instruction_done = float(getattr(new_state, 'instruction_done', 0.0))
        
        # Визуализируем шаг (состояние ПОСЛЕ применения действия)
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
        )
        
        obs = next_obs
        current_state = new_state

    print(f"Эпизод завершен: {step} шагов, общая награда: {total_reward:.4f}")

    # Сохраняем GIF
    visualizer.save_gif(f"episode_llm_{i}.gif", duration=500)
    # print("Визуализация сохранена в episode_llm.gif")