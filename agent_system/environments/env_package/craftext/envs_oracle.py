"""
Модификация Craftext среды с поддержкой агента-оракла.
Агент может задавать вопросы оракулу, который отвечает на них.
"""
import gc
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import ray
from typing import Optional, Dict, Any
from .utility import render_craftax_text, render_craftax_ascii, render_classic
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
from .oracle import CraftextOracle


class CraftextWorkerOracle:
    """
    Worker для Craftext с поддержкой оракла.
    Модифицированная версия CraftextWorker, которая может обрабатывать вопросы к оракулу.
    """
    def __init__(self, seed: int, env_kwargs: dict, oracle_handle: Optional[ray.ObjectRef] = None):
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

        # Oracle handle - ссылка на Ray remote actor
        self.oracle_handle = oracle_handle

        # debug counters
        self._debug_step_count = 0
        self._debug_reset_count = 0
        self._reset_counter = 0

        # Компиляция JIT-функций
        self._jitted_reset = jax.jit(self.wrapper.reset, static_argnames=['env_params'])
        self._jitted_step = jax.jit(self.wrapper.step, static_argnames=['env_params'])

        # Выбор типа наблюдения (как в CraftextWorker)
        self.observation_type = env_kwargs.get('observation_type', 'ascii')
        if self.observation_type == 'ascii':
            self.render_func = render_craftax_ascii
        elif self.observation_type == 'text':
            self.render_func = render_craftax_text
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

    def _shape_or_type(self, x):
        try:
            return getattr(x, 'shape', type(x))
        except Exception:
            return type(x)
    
    def step(self, action: int, question: Optional[str] = None):
        """
        Выполняет шаг в среде и обрабатывает вопрос к оракулу (если есть).
        
        Args:
            action: Числовое действие
            question: Опциональный вопрос для оракула
            
        Returns:
            Tuple of (obs, reward, done, info)
            info содержит 'oracle_answer' если был задан вопрос
        """
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")

        self.key, step_key = jax.random.split(self.key)

        # --- DEBUG: лог форм для первых N шагов ---
        if self._debug_step_count < 5:
            try:
                shapes = jax.tree_util.tree_map(self._shape_or_type, self.state)
            except Exception:
                shapes = str(type(self.state))
            print(f"[DEBUG] Calling _jitted_step. action={action}, question={question}, state_shapes={shapes}")
            self._debug_step_count += 1

        # Выполняем базовый шаг в среде
        obs_jax, new_state_jax, reward_jax, done_jax, info_jax = self._jitted_step(
            step_key,
            self.state,
            action,
            env_params=self.env_params
        )

        self.state = new_state_jax

        # Render the observation using render_classic and convert to numpy
        obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        obs = np.asarray(obs_jax_rendered)

        reward = float(reward_jax)
        done = bool(done_jax)

        info = {}
        info['won'] = done and reward > 0
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = self.render_func(env_state_cpu)  # Используем выбранную функцию рендеринга
        instruction_idx = new_state_jax.idx
        instruction_text = self.wrapper.scenario_handler.scenario_data.instructions_list[instruction_idx]
        info['text_render'] = text_render
        info['instruction'] = instruction_text

        # Сохраняем информацию о вопросе для последующей батч-обработки
        # Вместо вызова оракула напрямую, сохраняем данные вопроса в info
        if question and self.oracle_handle is not None:
            # Получаем контекст текущего состояния для оракла
            context = f"Task: {instruction_text}\nCurrent state: {text_render}"
            
            # Получаем env_state на CPU для передачи в оракла (для расширенного состояния)
            env_state_cpu_for_oracle = None
            try:
                env_state_cpu_for_oracle = jax.device_get(new_state_jax.env_state)
            except Exception as e:
                # Если не получилось получить env_state, это не критично
                env_state_cpu_for_oracle = None
            
            # Сохраняем данные вопроса в info для последующей батч-обработки
            info['oracle_question'] = question
            info['oracle_context'] = context
            info['oracle_env_state'] = env_state_cpu_for_oracle
            info['oracle_answer'] = None  # Будет заполнено после батч-обработки
        elif question:
            # Если вопрос есть, но оракла нет, возвращаем сообщение об ошибке
            info['oracle_answer'] = "Oracle not available"

        return obs, reward, done, info
    
    def reset(self, scenario_idx: int):
        """Сброс среды."""
        self._reset_counter += 1
        
        reset_seed = self.base_seed + scenario_idx * 1000000 + self._reset_counter * 1000
        reset_key = jax.random.PRNGKey(reset_seed)
        
        self.key, _ = jax.random.split(self.key)

        if self._debug_reset_count < 5:
            print(f"[DEBUG] Calling _jitted_reset. instruction_idx={scenario_idx}, reset_counter={self._reset_counter}")
            self._debug_reset_count += 1
            
        obs_jax, new_state_jax = self._jitted_reset(
            reset_key,
            instruction_idx=scenario_idx,
            env_params=self.env_params
        )

        self.state = new_state_jax

        obs_jax_rendered = render_classic(new_state_jax.env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        obs = np.asarray(obs_jax_rendered)
        info = {'won': False}
        env_state_cpu = jax.device_get(new_state_jax.env_state)
        text_render = self.render_func(env_state_cpu)  # Используем выбранную функцию рендеринга
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
        return jax.device_get(self.state)

    def close(self):
        pass


# -----------------------------------------------------------------------------
# Векторизованная Ray-среда ДЛЯ CRAFTEXT С ОРАКЛОМ ----------------------------
# -----------------------------------------------------------------------------

class CraftextMultiProcessEnvOracle(gym.Env):
    """
    Векторизованная обертка на базе Ray для Craftext с поддержкой оракла.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        oracle_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        
        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}

        # Создаем оракла как Ray remote actor (один на все воркеры)
        print(f"[CraftextMultiProcessEnvOracle] Creating oracle with model {oracle_model_name}...")
        self.oracle_handle = CraftextOracle.remote(oracle_model_name)

        # Создаем Ray-воркеры CraftextWorkerOracle
        env_worker = ray.remote(**resources_per_worker)(CraftextWorkerOracle)
        self._workers = [
            env_worker.remote(seed + i, self._env_kwargs, self.oracle_handle) 
            for i in range(self.num_processes)
        ]

        # Получаем количество сценариев от первого воркера
        num_scenarios = ray.get(self._workers[0].get_scenarios.remote())

        # Делим сценарии на train/eval
        split_idx = int(num_scenarios * 0.8)
        all_indices = np.arange(num_scenarios)

        if is_train:
            self.scenario_idxs = all_indices[:]
            print(f"CraftextOracle Training with {len(self.scenario_idxs)} scenarios.")
        else:
            self.scenario_idxs = all_indices[:]
            print(f"CraftextOracle Evaluating with {len(self.scenario_idxs)} scenarios.")

    def step(self, actions: list[int], questions: Optional[list[Optional[str]]] = None):
        """
        Выполняет шаг во всех средах с батч-обработкой вопросов к оракулу.
        
        Args:
            actions: Список числовых действий
            questions: Опциональный список вопросов для оракла (может быть None)
        """
        if questions is None:
            questions = [None] * len(actions)
        
        # Убеждаемся, что questions имеет правильную длину
        if len(questions) != len(actions):
            questions = questions[:len(actions)] + [None] * (len(actions) - len(questions))
        
        # Выполняем шаги во всех средах (без вызова оракула)
        futures = [
            worker.step.remote(action, question) 
            for worker, action, question in zip(self._workers, actions, questions)
        ]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        info_list = list(info_list)
        
        # Собираем все вопросы для батч-обработки
        questions_to_process = []
        question_indices = []
        for i, info in enumerate(info_list):
            if 'oracle_question' in info and info['oracle_question'] is not None:
                questions_to_process.append({
                    'question': info['oracle_question'],
                    'context': info.get('oracle_context'),
                    'env_state': info.get('oracle_env_state')
                })
                question_indices.append(i)
        
        # Обрабатываем все вопросы батчем через оракула
        if questions_to_process and self.oracle_handle is not None:
            try:
                oracle_answers = ray.get(
                    self.oracle_handle.answer_questions_batch.remote(questions_to_process)
                )
                # Добавляем ответы в соответствующие info и очищаем временные поля
                for idx, answer in zip(question_indices, oracle_answers):
                    info_list[idx]['oracle_answer'] = answer
                    # Очищаем временные поля, которые использовались для передачи данных
                    info_list[idx].pop('oracle_question', None)
                    info_list[idx].pop('oracle_context', None)
                    info_list[idx].pop('oracle_env_state', None)
            except Exception as e:
                print(f"[CraftextMultiProcessEnvOracle] Error in batch oracle processing: {e}")
                import traceback
                traceback.print_exc()
                # В случае ошибки, заполняем ответы об ошибкой
                for idx in question_indices:
                    if info_list[idx].get('oracle_answer') is None:
                        info_list[idx]['oracle_answer'] = f"Oracle batch error: {str(e)}"
                    # Очищаем временные поля даже в случае ошибки
                    info_list[idx].pop('oracle_question', None)
                    info_list[idx].pop('oracle_context', None)
                    info_list[idx].pop('oracle_env_state', None)
        
        return list(obs_list), list(reward_list), list(done_list), info_list

    def reset(self):
        """Сброс всех сред."""
        idxs = self._rng.choice(self.scenario_idxs, size=self.env_num, replace=True)
        idxs = np.repeat(idxs, self.group_n).tolist()

        futures = [
            worker.reset.remote(idx) for worker, idx in zip(self._workers, idxs)
        ]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def close(self):
        """Закрытие всех воркеров и оракла."""
        if getattr(self, '_closed', False):
            return
        
        # Закрываем воркеры
        ray.get([worker.close.remote() for worker in self._workers])
        [ray.kill(worker) for worker in self._workers]
        
        # Закрываем оракла
        if hasattr(self, 'oracle_handle') and self.oracle_handle is not None:
            ray.get(self.oracle_handle.close.remote())
            ray.kill(self.oracle_handle)
        
        self._closed = True

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Фабрика-хелпер --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_craftext_envs_oracle(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    oracle_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
):
    """Фабрика для создания CraftextMultiProcessEnvOracle."""
    return CraftextMultiProcessEnvOracle(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        oracle_model_name=oracle_model_name,
    )

