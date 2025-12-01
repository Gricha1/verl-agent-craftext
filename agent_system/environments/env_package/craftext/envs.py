import gc  # <--- ДОБАВЬТЕ ЭТОТ ИМПОРТ

import gymnasium as gym  # verl-agent, скорее всего, использует gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import ray
from .utility import render_craftax_text, render_craftax_ascii
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN


class CraftextWorker:
    """
    Этот класс компилирует JIT-функции на уровне экземпляра (одна компиляция на воркер).
    Также добавлен отладочный вывод форм, чтобы увидеть, какие части env_state меняют shape.
    """
    def __init__(self, seed: int, env_kwargs: dict):
        from craftax.craftax_env import make_craftax_env_from_name

        from craftext.enviroment.craftext_wrapper import InstructionWrapper
        from craftext.enviroment.encoders.craftext_base_model_encoder import EncodeForm
        from craftext.enviroment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
        from craftext.enviroment.scenarious.manager import ScenariosNoLambda

        env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
        self.wrapper = InstructionWrapper(
            env=env,
            config_name=env_kwargs.get('config_name', 'achievements_collect_wood'),
            sample_range=[0, 10_000],
            scenario_handler_class=ScenariosNoLambda,
            encode_model_class=DistilBertEncode,
            encode_form=env_kwargs.get('encode_form', EncodeForm.EMBEDDING)
        )
        self.env_params = self.wrapper.env.default_params
        self.key = jax.random.PRNGKey(seed)
        self.base_seed = seed
        self.state = None

        # debug counters
        self._debug_step_count = 0
        self._debug_reset_count = 0
        self._reset_counter = 0  # Счетчик для добавления случайности при каждом reset

        # --- Компиляция на уровне экземпляра ---
        # Компилируем bound-функции, не делая wrapper/env_params static_arg.
        # НЕ указываем action/instruction_idx в static_argnames, если они меняются
        self._jitted_reset = jax.jit(self.wrapper.reset, static_argnames=['env_params'])
        self._jitted_step = jax.jit(self.wrapper.step, static_argnames=['env_params'])

    def _shape_or_type(self, x):
        # helper for debug: return shape if array-like, else type
        try:
            return getattr(x, 'shape', type(x))
        except Exception:
            return type(x)
    
    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")

        self.key, step_key = jax.random.split(self.key)

        # --- DEBUG: лог форм для первых N шагов ---
        if self._debug_step_count < 5:
            try:
                shapes = jax.tree_util.tree_map(self._shape_or_type, self.state)
            except Exception:
                shapes = str(type(self.state))
            print(f"[DEBUG] Calling _jitted_step. action={action}, state_shapes={shapes}")
            self._debug_step_count += 1


        # Вызываем скомпилированную функцию, передавая только числовой env_state
        obs_jax, new_state_jax, reward_jax, done_jax, info_jax = self._jitted_step(
            step_key,
            self.state,
            action,
            env_params=self.env_params
        )

        # Заметь: если ты хочешь хранить состояние на host, можно делать jax.device_get здесь
        self.state = new_state_jax

        # Render the observation using render_classic and convert to numpy
        obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        obs = np.asarray(obs_jax_rendered)

        reward = float(reward_jax)
        done = bool(done_jax)

        info = {}  # Создаем пустой info, так как info_jax может быть None
        info['won'] = done and reward > 0
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = render_craftax_ascii(env_state_cpu)
        instruction_idx = new_state_jax.idx
        instruction_text = self.wrapper.scenario_handler.scenario_data.instructions_list[instruction_idx]
        info['text_render'] = text_render
        info['instruction'] = instruction_text

        return obs, reward, done, info
    
    def reset(self, scenario_idx: int):
        # Увеличиваем счетчик reset'ов для добавления случайности
        self._reset_counter += 1
        
        # Создаем уникальный ключ для каждого reset, комбинируя:
        # - базовый seed воркера
        # - scenario_idx (инструкция)
        # - счетчик reset'ов (для разных миров при одинаковой инструкции)
        # Это гарантирует, что для одного scenario_idx будут разные миры при каждом reset
        reset_seed = self.base_seed + scenario_idx * 1000000 + self._reset_counter * 1000
        reset_key = jax.random.PRNGKey(reset_seed)
        
        # Также обновляем основной ключ для step()
        self.key, _ = jax.random.split(self.key)

        # --- DEBUG: лог форм для первых N reset'ов ---
        if self._debug_reset_count < 5:
            print(f"[DEBUG] Calling _jitted_reset. instruction_idx={scenario_idx}, reset_counter={self._reset_counter}")
            self._debug_reset_count += 1
        obs_jax, new_state_jax = self._jitted_reset(
            reset_key,
            instruction_idx=scenario_idx,
            env_params=self.env_params
        )

        self.state = new_state_jax

        # Render the observation using render_classic and convert to numpy
        obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        obs = np.asarray(obs_jax_rendered)
        info = {'won': False}
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = render_craftax_ascii(env_state_cpu)
        instruction_text = self.wrapper.scenario_handler.scenario_data.instructions_list[scenario_idx]
        info['text_render'] = text_render
        info['instruction'] = instruction_text

        return obs, info

    def get_scenarios(self):
        return len(self.wrapper.scenario_handler.scenario_data.instructions_list)

    def get_state(self):
        """Get the current state of the environment. Returns None if reset() hasn't been called yet."""
        if self.state is None:
            return None
        # Convert JAX arrays to CPU/numpy for serialization
        return jax.device_get(self.state)

    def close(self):
        pass


# -----------------------------------------------------------------------------
# Векторизованная Ray-среда ДЛЯ CRAFTEXT --------------------------------------
# -----------------------------------------------------------------------------

class CraftextMultiProcessEnv(gym.Env):
    """
    Векторизованная обертка на базе Ray для Craftext.
    Этот класс почти идентичен WebshopMultiProcessEnv, заменены только воркеры.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        
        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}

        # Создаем Ray-воркеры CraftextWorker
        env_worker = ray.remote(**resources_per_worker)(CraftextWorker)
        self._workers = [
            env_worker.remote(seed + i, self._env_kwargs) 
            for i in range(self.num_processes)
        ]

        # Получаем количество сценариев от первого воркера (аналог goals)
        num_scenarios = ray.get(self._workers[0].get_scenarios.remote())

        # Делим сценарии на train/eval
        # Пример: 80% на train, 20% на eval (адаптируйте под себя)
        split_idx = int(num_scenarios * 0.8)
        all_indices = np.arange(num_scenarios)

        # get all indices either way
        if is_train:
            self.scenario_idxs = all_indices[:]
            print(f"Craftext Training with {len(self.scenario_idxs)} scenarios.")
        else:
            self.scenario_idxs = all_indices[:]
            print(f"Craftext Evaluating with {len(self.scenario_idxs)} scenarios.")

    def step(self, actions: list[int]):
        futures = [
            worker.step.remote(action) 
            for worker, action in zip(self._workers, actions)
        ]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def reset(self):
        # Выбираем случайные индексы сценариев для каждого env в группе
        idxs = self._rng.choice(self.scenario_idxs, size=self.env_num, replace=True)
        # Повторяем индексы для каждой среды внутри группы (требование group-based RL)
        idxs = np.repeat(idxs, self.group_n).tolist()

        futures = [
            worker.reset.remote(idx) for worker, idx in zip(self._workers, idxs)
        ]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def close(self):
        if getattr(self, '_closed', False): return
        ray.get([worker.close.remote() for worker in self._workers])
        [ray.kill(worker) for worker in self._workers]
        self._closed = True

    def __del__(self):
        self.close()

# -----------------------------------------------------------------------------
# Фабрика-хелпер --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_craftext_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Фабрика для создания CraftextMultiProcessEnv."""
    return CraftextMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )