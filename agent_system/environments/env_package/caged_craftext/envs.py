import gc
import os
import sys

os.environ.setdefault("CRAFTAX_RELOAD_TEXTURES", "True")


def _prepend_local_craftax_on_path():
    """Prefer caged_craftext/Craftax over pip craftax (debug map + spawn fixes)."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
    caged_root = os.environ.get("CAGED_CRAFTEXT_PATH", os.path.join(project_root, "caged_craftext"))
    craftax_root = os.path.join(caged_root, "Craftax")
    if os.path.isdir(craftax_root) and craftax_root not in sys.path:
        sys.path.insert(0, craftax_root)


_prepend_local_craftax_on_path()

import gymnasium as gym  # verl-agent, скорее всего, использует gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import ray
from .utility import (
    render_craftax_text,
    render_craftax_ascii,
    render_craftax_ascii_v2,
    overlay_episode_cumulative_stats,
)
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN


def _pick_text_render_fn(observation_type: str):
    if observation_type == "ascii":
        return render_craftax_ascii
    if observation_type == "ascii_v2":
        return render_craftax_ascii_v2
    if observation_type == "text":
        return render_craftax_text
    raise ValueError(f"Invalid observation type: {observation_type}")


def _import_generate_debug_square_world():
    """Load debug map generator from repo (pip craftax does not ship this module)."""
    import importlib.util

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "../../../.."))
    caged_root = os.environ.get("CAGED_CRAFTEXT_PATH", os.path.join(project_root, "caged_craftext"))
    module_path = os.path.join(caged_root, "Craftax/craftax/craftax_classic/debug_square_world_gen.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"debug_square_world_gen.py not found at {module_path}. "
            "Set CAGED_CRAFTEXT_PATH to the caged_craftext repo root."
        )
    spec = importlib.util.spec_from_file_location("caged_debug_square_world_gen", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_debug_square_world


def _craftax_env_params(env):
    """EnvParams for craftax env (supports debug override without mutating @property)."""
    return getattr(env, "_caged_debug_env_params", env.default_params)


def _make_craftax_classic_pixels_env(env_kwargs: dict):
    """Classic Craftax env; optional fixed 8x8 debug square map."""
    if env_kwargs.get("use_debug_square_map", False):
        from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnvNoAutoReset
        from craftax.craftax_classic.envs.craftax_state import StaticEnvParams

        generate_debug_square_world = _import_generate_debug_square_world()
        static_params = StaticEnvParams(map_size=(8, 8))
        env = CraftaxClassicPixelsEnvNoAutoReset(static_env_params=static_params)

        # No mob spawns on the tiny debug arena (pip craftax may lack map_size hook in default_params).
        env._caged_debug_env_params = _craftax_env_params(env).replace(
            spawn_cow_chance=0.0,
            spawn_zombie_base_chance=0.0,
            spawn_zombie_night_chance=0.0,
            spawn_skeleton_chance=0.0,
        )
        print(
            "[CagedCraftext] debug_square_8x8: fixed 8x8 map, no mob spawn, "
            "no random grass sapling drops (use repo Craftax on sys.path)."
        )

        # Pip-installed craftax has no debug generator; patch reset (procedural gen breaks on 8x8).
        def reset_env(rng, params):
            state = generate_debug_square_world(rng, params, env.static_env_params)
            return env.get_obs(state), state

        env.reset_env = reset_env

        # Belt-and-suspenders: strip mobs after each step even if pip craftax lacks debug_square hooks.
        import craftax.craftax_classic.game_logic as _craftax_gl

        _orig_craftax_step = _craftax_gl.craftax_step

        def _craftax_step_debug_safe(rng, state, action, params, static_params):
            state, reward = _orig_craftax_step(rng, state, action, params, static_params)
            if tuple(static_params.map_size) == (8, 8) and hasattr(_craftax_gl, "_strip_all_mobs"):
                state = _craftax_gl._strip_all_mobs(state, static_params)
            return state, reward

        _craftax_gl.craftax_step = _craftax_step_debug_safe
        return env
    from craftax.craftax_env import make_craftax_env_from_name

    return make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)


class CagedCraftextTextRenderActor:
    """
    Lightweight Ray actor: only ASCII/text render of a single craftax env_state (numpy pytree).
    No env, no BERT, no CMDP wrapper — avoids duplicating heavy per-env RAM.
    Wrapped with ray.remote(...) when spawned (see CagedCraftextOptimisticVecEnv).
    """

    def __init__(self, observation_type: str):
        self._render = _pick_text_render_fn(observation_type)

    def render_craftax_state(self, state_numpy_tree):
        return self._render(state_numpy_tree)


class CagedCraftextWorker:
    """
    Worker для Caged Craftext - безопасной версии Craftext с CMDP (Constrained Markov Decision Process).
    Этот класс компилирует JIT-функции на уровне экземпляра (одна компиляция на воркер).
    """
    def __init__(self, seed: int, env_kwargs: dict):
        # Добавляем путь к caged_craftext в sys.path, если его там еще нет
        # Это нужно для импорта модуля craftext из caged_craftext
        caged_craftext_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../../..', 'caged_craftext')
        )
        if caged_craftext_path not in sys.path:
            sys.path.insert(0, caged_craftext_path)

        # Используем caged_craftext версию wrapper с CMDP поддержкой
        from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
        from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
        from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
        from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

        env = _make_craftax_classic_pixels_env(env_kwargs)
        self.wrapper = CMDPInstructionWrapper(
            env=env,
            config_name=env_kwargs.get('config_name', 'achievements_safe_caged'),
            scenario_handler_class=ScenariosNoLambdaCMDP,
            encode_model_class=DistilBertEncode,
            encode_form=env_kwargs.get('encode_form', EncodeForm.EMBEDDING)
        )
        self.env_params = _craftax_env_params(self.wrapper.env)
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

        self.observation_type = env_kwargs.get('observation_type', 'ascii')
        # If False, we avoid rendering/returning pixel observations on each step/reset.
        # This is important for RAM when running many parallel env workers.
        self.use_pixel_obs = bool(env_kwargs.get("use_pixel_obs", False))
        if self.observation_type == 'ascii':
            self.render_func = render_craftax_ascii
        elif self.observation_type == 'ascii_v2':
            self.render_func = render_craftax_ascii_v2
        elif self.observation_type == 'text':
            self.render_func = render_craftax_text
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self._episode_return_cum = 0.0
        self._episode_step = 0

    def _shape_or_type(self, x):
        # helper for debug: return shape if array-like, else type
        try:
            return getattr(x, 'shape', type(x))
        except Exception:
            return type(x)
    
    def step(self, action: int, return_render: bool = False):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")

        self.key, step_key = jax.random.split(self.key)

        # --- DEBUG: лог форм для первых N шагов ---
        if self._debug_step_count < 5:
            try:
                shapes = jax.tree_util.tree_map(self._shape_or_type, self.state)
            except Exception:
                shapes = str(type(self.state))
            print(f"[DEBUG CagedCraftext] Calling _jitted_step. action={action}, state_shapes={shapes}")
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

        # Render pixel observations only when needed:
        # - for VLEnv (use_pixel_obs=True)
        # - or when recording a validation video (return_render=True)
        obs = None
        if return_render or self.use_pixel_obs:
            obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
            obs = np.asarray(obs_jax_rendered)

        reward = float(reward_jax)
        done = bool(done_jax)

        info = {}  # Создаем пустой info, так как info_jax может быть None
        info['won'] = done and reward > 0
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = self.render_func(env_state_cpu)
        instruction_idx = new_state_jax.idx
        instruction_text = self.wrapper.scenario_handler.scenario_data.instructions_list[instruction_idx]
        
        # Добавляем информацию о constraint (если есть)
        if hasattr(new_state_jax, 'textual_constraint') and new_state_jax.textual_constraint is not None:
            constraint_idx = new_state_jax.idx
            if hasattr(self.wrapper.scenario_handler.scenario_data, 'constraints_list'):
                constraint_text = self.wrapper.scenario_handler.scenario_data.constraints_list[constraint_idx]
                info['constraint'] = constraint_text
        
        # Добавляем информацию о cost (если есть)
        if hasattr(new_state_jax, 'cost'):
            info['cost'] = float(new_state_jax.cost)
        if hasattr(new_state_jax, 'episode_cost'):
            info['episode_cost'] = float(new_state_jax.episode_cost)
            
        info['text_render'] = text_render
        info['instruction'] = instruction_text
        info['instruction_done'] = bool(getattr(new_state_jax, 'instruction_done', False))
        info['done'] = done
        self._episode_return_cum += reward
        info['episode_return_cum'] = float(self._episode_return_cum)

        self._episode_step += 1
        if return_render and obs is not None:
            ep_cost = float(info.get('episode_cost', 0.0))
            info['render_frame'] = overlay_episode_cumulative_stats(
                obs, self._episode_return_cum, ep_cost, step=self._episode_step
            )
            info['env_step'] = self._episode_step

        if done:
            self._episode_return_cum = 0.0
            self._episode_step = 0

        return obs, reward, done, info
    
    def reset(self, scenario_idx: int, return_render: bool = False):
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
            print(f"[DEBUG CagedCraftext] Calling _jitted_reset. instruction_idx={scenario_idx}, reset_counter={self._reset_counter}")
            self._debug_reset_count += 1
        obs_jax, new_state_jax = self._jitted_reset(
            reset_key,
            instruction_idx=scenario_idx,
            env_params=self.env_params
        )

        self.state = new_state_jax

        # Render pixel observations only when needed (see step()).
        obs = None
        if return_render or self.use_pixel_obs:
            obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
            obs = np.asarray(obs_jax_rendered)
        info = {'won': False}
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = self.render_func(env_state_cpu)
        instruction_text = self.wrapper.scenario_handler.scenario_data.instructions_list[scenario_idx]
        
        # Добавляем информацию о constraint (если есть)
        if hasattr(new_state_jax, 'textual_constraint') and new_state_jax.textual_constraint is not None:
            if hasattr(self.wrapper.scenario_handler.scenario_data, 'constraints_list'):
                constraint_text = self.wrapper.scenario_handler.scenario_data.constraints_list[scenario_idx]
                info['constraint'] = constraint_text
        
        info['text_render'] = text_render
        info['instruction'] = instruction_text
        info['instruction_done'] = False
        self._episode_return_cum = 0.0
        self._episode_step = 0
        info['episode_return_cum'] = 0.0

        if return_render and obs is not None:
            ep_cost = float(info.get('episode_cost', 0.0))
            info['render_frame'] = overlay_episode_cumulative_stats(
                obs, 0.0, ep_cost, step=self._episode_step
            )
            info['env_step'] = self._episode_step

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
# Векторизованная Ray-среда ДЛЯ CAGED CRAFTEXT ---------------------------------
# -----------------------------------------------------------------------------

class CagedCraftextMultiProcessEnv(gym.Env):
    """
    Векторизованная обертка на базе Ray для Caged Craftext.
    Этот класс почти идентичен CraftextMultiProcessEnv, но использует CagedCraftextWorker.
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

        # Передаем переменные окружения в Ray workers
        # Вычисляем путь относительно корня проекта
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '../../../../'))
        default_caged_path = os.path.join(project_root, 'caged_craftext')
        
        # Используем переменную окружения или вычисленный путь
        caged_craftext_path = os.environ.get('CAGED_CRAFTEXT_PATH', default_caged_path)
        caged_craftext_path = os.path.abspath(caged_craftext_path)
        
        # Собираем все необходимые переменные окружения для Ray workers
        # НЕ устанавливаем CUDA_VISIBLE_DEVICES="" здесь, так как это может мешать Ray
        # Encoder'ы уже исправлены и проверяют torch.cuda.is_available() перед использованием CUDA
        jax_platforms = os.environ.get("JAX_PLATFORMS")
        env_vars = {
            "CAGED_CRAFTEXT_PATH": caged_craftext_path,
            "PYTHONPATH": f"{caged_craftext_path}:{os.environ.get('PYTHONPATH', '')}",
            "CRAFTAX_RELOAD_TEXTURES": os.environ.get("CRAFTAX_RELOAD_TEXTURES", "True"),
        }
        if jax_platforms:
            env_vars["JAX_PLATFORMS"] = jax_platforms
        elif not resources_per_worker.get("num_gpus"):
            # CPU Ray env workers: keep JAX off GPU (vLLM/FSDP own the GPUs).
            env_vars.setdefault("JAX_PLATFORMS", "cpu")

        # Передаем CUDA_VISIBLE_DEVICES только если он явно установлен в окружении
        # Не устанавливаем пустое значение, чтобы не мешать Ray
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']:
            env_vars["CUDA_VISIBLE_DEVICES"] = os.environ['CUDA_VISIBLE_DEVICES']
        
        runtime_env = {
            "env_vars": env_vars
        }
        print(f"[DEBUG] CagedCraftextMultiProcessEnv: Передаем в Ray workers:")
        print(f"  CAGED_CRAFTEXT_PATH={caged_craftext_path}")
        print(f"  JAX_PLATFORMS={env_vars.get('JAX_PLATFORMS', '(not set)')}")
        if 'CUDA_VISIBLE_DEVICES' in env_vars:
            print(f"  CUDA_VISIBLE_DEVICES={env_vars['CUDA_VISIBLE_DEVICES']}")
        else:
            print(f"  CUDA_VISIBLE_DEVICES=не установлен (encoder'ы будут проверять torch.cuda.is_available())")

        self._workers = []
        self._closed = False

        # Создаем Ray-воркеры CagedCraftextWorker с runtime_env
        env_worker = ray.remote(**resources_per_worker, runtime_env=runtime_env)(CagedCraftextWorker)
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
            print(f"CagedCraftext Training with {len(self.scenario_idxs)} scenarios.")
        else:
            self.scenario_idxs = all_indices[:]
            print(f"CagedCraftext Evaluating with {len(self.scenario_idxs)} scenarios.")

        # Индексы воркеров, для которых при step/reset возвращать render_frame (для записи видео)
        self._record_video_worker_idxs = set()

    def set_record_video_worker_idxs(self, idxs):
        """Установить индексы воркеров, для которых возвращать render_frame в info (для записи видео)."""
        self._record_video_worker_idxs = set(idxs) if idxs is not None else set()

    def step(self, actions: list[int]):
        return_render_flags = [i in self._record_video_worker_idxs for i in range(len(self._workers))]
        futures = [
            worker.step.remote(action, return_render=return_render)
            for worker, action, return_render in zip(self._workers, actions, return_render_flags)
        ]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def reset(self):
        # Выбираем случайные индексы сценариев для каждого env в группе
        idxs = self._rng.choice(self.scenario_idxs, size=self.env_num, replace=True)
        # Повторяем индексы для каждой среды внутри группы (требование group-based RL)
        idxs = np.repeat(idxs, self.group_n).tolist()

        return_render_flags = [i in self._record_video_worker_idxs for i in range(len(self._workers))]
        futures = [
            worker.reset.remote(idx, return_render=return_render)
            for worker, idx, return_render in zip(self._workers, idxs, return_render_flags)
        ]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def close(self):
        if getattr(self, "_closed", False):
            return
        workers = getattr(self, "_workers", None)
        if workers:
            try:
                ray.get([worker.close.remote() for worker in workers])
            except Exception:
                pass
            for worker in workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Фабрика-хелпер --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_caged_craftext_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Фабрика для создания CagedCraftextMultiProcessEnv."""
    return CagedCraftextMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )


# -----------------------------------------------------------------------------
# Batched optimistic-reset env (NO ray env workers)
# -----------------------------------------------------------------------------

class CagedCraftextOptimisticVecEnv(gym.Env):
    """
    Batched CagedCraftext env that runs NUM_ENVS environments inside a single process
    using JAX vmap + OptimisticResetVecEnvWrapper (like caged_craftext/baselines/ppo_lag*).

    This avoids spawning env_num Ray actors for environments. It is intended to be created
    once and stepped with batched actions of length env_num.
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        env_kwargs: dict | None = None,
        reset_ratio: int | None = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        # Make sure caged_craftext is importable
        caged_craftext_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", "caged_craftext")
        )
        if caged_craftext_path not in sys.path:
            sys.path.insert(0, caged_craftext_path)

        from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
        from craftext.environment.encoders.craftext_base_model_encoder import EncodeForm
        from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode
        from craftext.environment.scenarious.manager_cmdp import ScenariosNoLambdaCMDP

        # Optimistic wrapper from caged_craftext baselines
        # baselines/ is not a Python package (no __init__.py), so import via direct path.
        baselines_path = os.path.join(caged_craftext_path, "baselines")
        if baselines_path not in sys.path:
            sys.path.insert(0, baselines_path)
        from wrappers_cmdp import OptimisticResetVecEnvWrapper

        self.env_num = int(env_num)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}
        self._rng = np.random.RandomState(seed)

        env = _make_craftax_classic_pixels_env(self._env_kwargs)
        self.wrapper = CMDPInstructionWrapper(
            env=env,
            config_name=self._env_kwargs.get("config_name", "achievements_safe_caged"),
            scenario_handler_class=ScenariosNoLambdaCMDP,
            encode_model_class=DistilBertEncode,
            encode_form=self._env_kwargs.get("encode_form", EncodeForm.EMBEDDING),
        )
        self.env_params = _craftax_env_params(self.wrapper.env)

        # Batched optimistic-reset wrapper
        if reset_ratio is None:
            reset_ratio = min(16, self.env_num)
        reset_ratio = int(reset_ratio)
        if reset_ratio <= 0:
            reset_ratio = 1
        # Must divide perfectly (OptimisticResetVecEnvWrapper requirement)
        if self.env_num % reset_ratio != 0:
            # fallback to 1 (always divides)
            reset_ratio = 1
        self._vec_env = OptimisticResetVecEnvWrapper(
            self.wrapper, num_envs=self.env_num, reset_ratio=reset_ratio
        )

        self.key = jax.random.PRNGKey(seed)
        self.state = None

        self.observation_type = self._env_kwargs.get("observation_type", "ascii")
        self.use_pixel_obs = bool(self._env_kwargs.get("use_pixel_obs", False))
        if self.observation_type == "ascii":
            self.render_func = render_craftax_ascii
        elif self.observation_type == "ascii_v2":
            self.render_func = render_craftax_ascii_v2
        elif self.observation_type == "text":
            self.render_func = render_craftax_text
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self._record_video_env_idxs: set[int] = set()
        self._is_train = bool(is_train)
        self._episode_return_cum = np.zeros(self.env_num, dtype=np.float32)
        self._episode_steps = np.zeros(self.env_num, dtype=np.int32)
        # Monotonic step index within one rollout (for video); only reset on env.reset().
        self._rollout_steps = np.zeros(self.env_num, dtype=np.int32)

        # Optional: one Ray actor per env — parallel text_render only (no duplicate env/BERT).
        # Train only: val + Ray ships env_state through the object store every step; with tiny
        # num_cpus Ray often runs renders quasi-serially while still paying full serialize/IPC cost.
        self._use_ray_text_render = bool(self._env_kwargs.get("use_ray_text_render_workers", False))
        self._text_render_actors: list | None = None
        if self._use_ray_text_render and self._is_train:
            if not ray.is_initialized():
                ray.init()
            ncpus = float(self._env_kwargs.get("text_render_ray_num_cpus", 0.25))
            ncpus = max(0.0, ncpus)
            RemoteCls = ray.remote(num_cpus=ncpus)(CagedCraftextTextRenderActor)
            self._text_render_actors = [
                RemoteCls.remote(self.observation_type) for _ in range(self.env_num)
            ]
            print(
                f"[CagedCraftextOptimisticVecEnv] Ray text-render actors: {self.env_num} "
                f"(num_cpus={ncpus} each)"
            )
        elif self._use_ray_text_render and not self._is_train:
            print(
                "[CagedCraftextOptimisticVecEnv] use_ray_text_render_workers=True ignored for val env "
                "(sync text_render; Ray on val is slow due to object-store IPC per step)."
            )

    def set_record_video_worker_idxs(self, idxs):
        # Keep same API as MultiProcess env, but here idxs refer to env indices inside the batch.
        self._record_video_env_idxs = set(idxs) if idxs is not None else set()

    def _build_info_list(
        self,
        state_batched,
        reward_batched,
        done_batched,
        render_frames: dict[int, np.ndarray] | None = None,
    ):
        # Pull CPU state for text rendering + instruction/constraint lookup
        render_frames = render_frames or {}
        infos: list[dict] = []

        # In this optimistic env, state_batched is typically TextEnvStateCMDP (batched),
        # where instruction index lives in state_batched.idx (NOT in state_batched.env_state.idx).
        state_cpu = jax.device_get(state_batched)

        idxs_arr = getattr(state_cpu, "idx", None)
        idxs = np.asarray(idxs_arr).tolist() if idxs_arr is not None else [0] * self.env_num

        craftax_state_batched = getattr(state_cpu, "env_state", None)
        cost_batched = getattr(state_cpu, "cost", None)
        episode_cost_batched = getattr(state_cpu, "episode_cost", None)
        instruction_done_batched = getattr(state_cpu, "instruction_done", None)
        total_sr_batched = getattr(state_cpu, "total_success_rate", None)

        # Parallel ASCII/text render via one Ray actor per env (optional).
        if self._text_render_actors is not None and craftax_state_batched is not None:
            futures = []
            for i in range(self.env_num):
                craftax_state_i = jax.tree_util.tree_map(
                    lambda x: np.asarray(x[i]), craftax_state_batched
                )
                futures.append(self._text_render_actors[i].render_craftax_state.remote(craftax_state_i))
            text_renders = ray.get(futures)
        else:
            text_renders = []
            for i in range(self.env_num):
                try:
                    if craftax_state_batched is not None:
                        craftax_state_i = jax.tree_util.tree_map(lambda x: x[i], craftax_state_batched)
                        text_renders.append(self.render_func(craftax_state_i))
                    else:
                        text_renders.append("The world is empty.")
                except Exception:
                    text_renders.append("The world is empty.")

        for i in range(self.env_num):
            info = {}
            r = float(np.asarray(jax.device_get(reward_batched[i])))
            d = bool(np.asarray(jax.device_get(done_batched[i])))
            info["won"] = d and r > 0
            info["text_render"] = text_renders[i]

            # instruction/constraint (string) from scenario handler lists
            try:
                idx = int(idxs[i])
                info["instruction"] = self.wrapper.scenario_handler.scenario_data.instructions_list[idx]
                if hasattr(self.wrapper.scenario_handler.scenario_data, "constraints_list"):
                    info["constraint"] = self.wrapper.scenario_handler.scenario_data.constraints_list[idx]
            except Exception:
                pass

            # cost fields if present
            try:
                if cost_batched is not None:
                    info["cost"] = float(np.asarray(cost_batched[i]))
            except Exception:
                pass
            try:
                if episode_cost_batched is not None:
                    info["episode_cost"] = float(np.asarray(episode_cost_batched[i]))
            except Exception:
                pass

            info["episode_return_cum"] = float(self._episode_return_cum[i])
            info["env_step"] = int(self._rollout_steps[i])
            info["episode_step"] = int(self._episode_steps[i])
            info["done"] = d

            try:
                if instruction_done_batched is not None:
                    info["instruction_done"] = bool(np.asarray(instruction_done_batched[i]))
            except Exception:
                pass
            try:
                if total_sr_batched is not None:
                    info["success_rate"] = float(np.asarray(total_sr_batched[i]))
            except Exception:
                pass

            if i in render_frames:
                ep_cost = float(info.get("episode_cost", 0.0))
                info["render_frame"] = overlay_episode_cumulative_stats(
                    render_frames[i],
                    self._episode_return_cum[i],
                    ep_cost,
                    step=int(self._rollout_steps[i]),
                )

            infos.append(info)

        return infos

    def reset(self):
        self.key, reset_key = jax.random.split(self.key)
        obs, state = self._vec_env.reset(reset_key, self.env_params)
        self.state = state
        self._episode_return_cum[:] = 0.0
        self._episode_steps[:] = 0
        self._rollout_steps[:] = 0

        render_frames: dict[int, np.ndarray] = {}
        if self._record_video_env_idxs:
            # Only render selected env indices
            state_cpu = jax.device_get(state)
            env_state_cpu = getattr(state_cpu, "env_state", None)
            for i in self._record_video_env_idxs:
                if i < 0 or i >= self.env_num:
                    continue
                try:
                    craftax_state_i = jax.tree_util.tree_map(lambda x: x[i], env_state_cpu)
                    obs_jax_rendered = render_classic(craftax_state_i, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
                    render_frames[i] = np.asarray(obs_jax_rendered).copy()
                except Exception:
                    pass

        # For compatibility with existing managers, return list of obs (pixel obs optional) and list of infos
        obs_list = [None] * self.env_num
        if self.use_pixel_obs:
            # If pixel obs is required (VLEnv), render for all envs
            state_cpu = jax.device_get(state)
            env_state_cpu = getattr(state_cpu, "env_state", None)
            for i in range(self.env_num):
                try:
                    craftax_state_i = jax.tree_util.tree_map(lambda x: x[i], env_state_cpu)
                    obs_jax_rendered = render_classic(craftax_state_i, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
                    obs_list[i] = np.asarray(obs_jax_rendered)
                except Exception:
                    obs_list[i] = None

        infos = self._build_info_list(state, jnp.zeros((self.env_num,)), jnp.zeros((self.env_num,), dtype=bool), render_frames=render_frames)
        return obs_list, infos

    def step(self, actions: list[int]):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if len(actions) != self.env_num:
            raise ValueError(f"Expected {self.env_num} actions, got {len(actions)}")

        self.key, step_key = jax.random.split(self.key)
        action_arr = jnp.asarray(actions, dtype=jnp.int32)
        obs, new_state, reward, done, info, state_pre_reset = self._vec_env.step(
            step_key, self.state, action_arr, self.env_params
        )
        self.state = new_state

        render_frames: dict[int, np.ndarray] = {}
        if self._record_video_env_idxs:
            state_cpu = jax.device_get(new_state)
            state_pre_reset_cpu = jax.device_get(state_pre_reset)
            done_batched = jnp.asarray(done)
            for i in self._record_video_env_idxs:
                if i < 0 or i >= self.env_num:
                    continue
                try:
                    # On done, optimistic wrapper already swapped in a fresh episode — render terminal pose.
                    use_terminal = bool(np.asarray(done_batched[i]))
                    src = state_pre_reset_cpu if use_terminal else state_cpu
                    env_state_cpu = getattr(src, "env_state", None)
                    if env_state_cpu is None:
                        continue
                    craftax_state_i = jax.tree_util.tree_map(lambda x: x[i], env_state_cpu)
                    obs_jax_rendered = render_classic(craftax_state_i, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
                    render_frames[i] = np.asarray(obs_jax_rendered).copy()
                except Exception:
                    pass

        obs_list = [None] * self.env_num
        if self.use_pixel_obs:
            state_cpu = jax.device_get(new_state)
            env_state_cpu = getattr(state_cpu, "env_state", None)
            for i in range(self.env_num):
                try:
                    craftax_state_i = jax.tree_util.tree_map(lambda x: x[i], env_state_cpu)
                    obs_jax_rendered = render_classic(craftax_state_i, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
                    obs_list[i] = np.asarray(obs_jax_rendered)
                except Exception:
                    obs_list[i] = None

        reward_list = [float(x) for x in np.asarray(jax.device_get(reward)).tolist()]
        done_list = [bool(x) for x in np.asarray(jax.device_get(done)).tolist()]
        for i, r in enumerate(reward_list):
            self._episode_return_cum[i] += r
        self._rollout_steps += 1
        self._episode_steps += 1
        infos = self._build_info_list(new_state, reward, done, render_frames=render_frames)
        for i, d in enumerate(done_list):
            if d:
                self._episode_return_cum[i] = 0.0
                self._episode_steps[i] = 0
        return obs_list, reward_list, done_list, infos

    def close(self):
        if getattr(self, "_text_render_actors", None):
            for actor in self._text_render_actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            self._text_render_actors = None


def build_caged_craftext_envs_optimistic(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    reset_ratio: int | None = None,
):
    # group_n is ignored (this env already batches internally)
    _ = resources_per_worker
    return CagedCraftextOptimisticVecEnv(
        seed=seed,
        env_num=env_num,
        env_kwargs=env_kwargs,
        reset_ratio=reset_ratio,
        is_train=is_train,
    )
