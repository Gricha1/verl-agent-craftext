import os
import multiprocessing
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
import numpy as np


class SimpleAgent:
    """
    Простой класс Agent для инференса модели с gym-подобным интерфейсом.
    
    Пример использования:
        agent = SimpleAgent(
            checkpoint_path="/path/to/checkpoint",
            base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            lora_rank=64,
            lora_alpha=64,
            target_modules="all-linear",
        )
        
        obs, info = env.reset()
        action = agent.predict(obs['text'][0])
        next_obs, reward, done, info = env.step([action])
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str = None,
        lora_rank: int = 0,
        lora_alpha: int = 0,
        target_modules: str = None,
        max_response_length: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        device: str = "cuda",
    ):
        """
        Инициализация Agent.
        
        Args:
            checkpoint_path: Путь к чекпоинту модели (например, global_step_500)
            base_model_path: Путь к базовой модели (если None, берется из чекпоинта)
            lora_rank: Ранг LoRA (0 если не используется LoRA)
            lora_alpha: Alpha для LoRA
            target_modules: Модули для LoRA (например, "all-linear")
            max_response_length: Максимальная длина ответа
            do_sample: Использовать ли сэмплирование
            temperature: Температура для сэмплирования
            device: Устройство для инференса
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.max_response_length = max_response_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.device = device
        
        # Определяем путь к модели для загрузки токенизатора
        model_path = base_model_path if base_model_path else checkpoint_path
        
        # Определяем, является ли модель VLM
        self.is_vlm = self._is_vlm_model(model_path)
        
        if self.is_vlm:
            # Для VLM используем transformers напрямую
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            self.device = device
            print(f"Инициализация VLM модели {model_path}...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            self.llm = None  # Не используем vLLM для VLM
            print("VLM модель загружена!")
        else:
            # Для обычных LLM используем vLLM
            # Загружаем токенизатор из transformers (не vLLM!)
            # Этот токенизатор будет использоваться для apply_chat_template
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=True
            )
            
            # Загружаем конфиг модели
            self.model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        

         # Настраиваем LoRA если нужно
        self.lora_path = None  # ДОБАВЛЯЕМ это
        if lora_rank > 0:
            lora_path = os.path.join(checkpoint_path, "actor", "lora_adapter")
            if not os.path.exists(lora_path):
                possible_paths = [
                    os.path.join(checkpoint_path, "adapter_model"),
                    os.path.join(checkpoint_path, "lora_adapter"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        lora_path = path
                        break
                
                if not lora_path or not os.path.exists(lora_path):
                    print(f"Предупреждение: LoRA адаптер не найден в {checkpoint_path}")
                    print("Используется базовая модель без LoRA")
                    lora_rank = 0
                else:
                    self.lora_path = lora_path  # СОХРАНЯЕМ путь
            else:
                self.lora_path = lora_path  # СОХРАНЯЕМ путь

        
        if not self.is_vlm:
            # Инициализируем vLLM engine только для LLM
            print("Инициализация vLLM engine...")
            if not base_model_path:
                raise ValueError("base_model_path обязателен для загрузки модели в vLLM")
            
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            vllm_kwargs = {
                "model": base_model_path,
                "tensor_parallel_size": 1,
                "dtype": "auto",
                "enforce_eager": True,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 2048 + max_response_length,
                "trust_remote_code": True,
                "disable_custom_all_reduce": True,
            }
            
            # Добавляем поддержку LoRA если нужно
            if lora_rank > 0 and lora_path and os.path.exists(lora_path):
                adapter_config = os.path.join(lora_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    vllm_kwargs["enable_lora"] = True
                    vllm_kwargs["max_lora_rank"] = lora_rank
                    vllm_kwargs["max_loras"] = 1
                    print(f"Включена поддержка LoRA (rank={lora_rank})")
            
            self.llm = LLM(**vllm_kwargs)
            
            # НЕ обновляем self.tokenizer! Оставляем тот, что загрузили из transformers
            
            # Загружаем LoRA адаптер если нужно
            self.use_lora = False
            self.lora_name = None
            if lora_rank > 0 and lora_path and os.path.exists(lora_path):
                try:
                    from vllm.lora.request import LoRARequest
                    adapter_config = os.path.join(lora_path, "adapter_config.json")
                    if not os.path.exists(adapter_config):
                        print(f"Предупреждение: adapter_config.json не найден в {lora_path}")
                        print("Продолжаем без LoRA")
                    else:
                        lora_request = LoRARequest(
                            lora_name="agent_lora",
                            lora_int_id=1,
                            lora_path=lora_path,
                        )
                        self.llm.llm_engine.add_lora(lora_request)
                        self.use_lora = True
                        self.lora_name = "agent_lora"
                        print(f"LoRA адаптер загружен из {lora_path}")
                except Exception as e:
                    import traceback
                    print(f"Ошибка при загрузке LoRA: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    print("Продолжаем без LoRA")
            
            # Настраиваем параметры сэмплирования
            self.sampling_params = SamplingParams(
                n=1,
                max_tokens=max_response_length,
                temperature=temperature if do_sample else 0.0,
                stop=None,
            )
        else:
            # Для VLM не используем LoRA и vLLM
            self.use_lora = False
            self.lora_name = None
        
        print("SimpleAgent инициализирован!")
    
    def _is_vlm_model(self, model_path: str) -> bool:
        """
        Определяет, является ли модель VLM (Vision-Language Model).
        
        Args:
            model_path: путь к модели
        
        Returns:
            True если модель VLM, False иначе
        """
        # Проверяем по названию модели
        vlm_keywords = ["vl", "vision", "qwen2-vl", "qwen2.5-vl"]
        model_path_lower = model_path.lower()
        for keyword in vlm_keywords:
            if keyword in model_path_lower:
                return True
        
        # Проверяем по конфигу модели
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(config, 'model_type', '').lower()
            if 'vl' in model_type or 'vision' in model_type:
                return True
        except:
            pass
        
        return False
    

    def predict(self, observation: str, image=None) -> str:
        """
        Предсказывает действие на основе наблюдения.
        
        Args:
            observation: Текстовое наблюдение из среды
            image: Опциональное изображение (PIL Image или numpy array) для VLM моделей
            
        Returns:
            Текстовое действие
        """
        if self.is_vlm:
            # Для VLM моделей
            if image is None:
                raise ValueError("Для VLM моделей необходимо передать изображение")
            
            return self._predict_vlm(observation, image)
        else:
            # Для обычных LLM моделей
            return self._predict_llm(observation)
    
    def predict_with_prompt(self, observation: str, image=None) -> dict:
        """
        Предсказывает действие на основе наблюдения и возвращает промпт и вывод модели.
        
        Args:
            observation: Текстовое наблюдение из среды
            image: Опциональное изображение (PIL Image или numpy array) для VLM моделей
            
        Returns:
            Словарь с ключами:
                - 'prompt': промпт, отправленный модели
                - 'model_output': полный вывод модели
                - 'action': извлеченное действие (строка)
        """
        if self.is_vlm:
            # Для VLM моделей
            if image is None:
                raise ValueError("Для VLM моделей необходимо передать изображение")
            
            return self._predict_vlm_with_prompt(observation, image)
        else:
            # Для обычных LLM моделей
            return self._predict_llm_with_prompt(observation)
    
    def _predict_llm(self, observation: str) -> str:
        """Предсказание для LLM моделей."""
        # Формируем промпт используя chat template
        chat = [{"role": "user", "content": observation}]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Генерируем ответ
        if self.use_lora and self.lora_path:
            try:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest(
                    lora_name=self.lora_name,
                    lora_int_id=1,
                    lora_path=self.lora_path,
                )
                # lora_request передается в generate(), а НЕ в SamplingParams!
                outputs = self.llm.generate(
                    [prompt], 
                    self.sampling_params,
                    lora_request=lora_request  # Передаем здесь!
                )
            except Exception as e:
                import traceback
                print(f"Ошибка при генерации с LoRA: {e}")
                print(traceback.format_exc())
                print("Используем без LoRA")
                outputs = self.llm.generate([prompt], self.sampling_params)
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
        
        response = outputs[0].outputs[0].text.strip()
        
        return response
    
    def _predict_llm_with_prompt(self, observation: str) -> dict:
        """Предсказание для LLM моделей с возвратом промпта."""
        # Формируем промпт используя chat template
        chat = [{"role": "user", "content": observation}]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Генерируем ответ
        if self.use_lora and self.lora_path:
            try:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest(
                    lora_name=self.lora_name,
                    lora_int_id=1,
                    lora_path=self.lora_path,
                )
                outputs = self.llm.generate(
                    [prompt], 
                    self.sampling_params,
                    lora_request=lora_request
                )
            except Exception as e:
                import traceback
                print(f"Ошибка при генерации с LoRA: {e}")
                print(traceback.format_exc())
                print("Используем без LoRA")
                outputs = self.llm.generate([prompt], self.sampling_params)
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
        
        response = outputs[0].outputs[0].text.strip()
        
        return {
            'prompt': prompt,
            'model_output': response,
            'action': response
        }
    
    def _predict_vlm(self, text_prompt: str, image) -> str:
        """Предсказание для VLM моделей."""
        import torch
        from PIL import Image as PILImage
        from verl.utils.dataset.vision_utils import process_image as process_image_util
        
        # Конвертируем изображение в PIL если нужно
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        elif not isinstance(image, PILImage.Image):
            raise ValueError(f"Неизвестный тип изображения: {type(image)}")
        
        # Обрабатываем изображение через утилиту (как в rl_dataset.py)
        processed_image = process_image_util(image)
        
        # Формируем сообщения
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Применяем chat template (как в rl_dataset.py)
        raw_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Обрабатываем через processor напрямую (как в rl_dataset.py)
        # Processor сам обработает изображения и заменит плейсхолдеры
        inputs = self.processor(
            text=[raw_prompt],
            images=[processed_image],
            return_tensors="pt",
        )
        
        # Перемещаем на устройство
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Генерируем ответ
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_response_length,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
            )
        
        # Обрезаем сгенерированные токены (убираем входные)
        input_ids = inputs['input_ids']
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
    
    def _predict_vlm_with_prompt(self, text_prompt: str, image) -> dict:
        """Предсказание для VLM моделей с возвратом промпта."""
        import torch
        from PIL import Image as PILImage
        from verl.utils.dataset.vision_utils import process_image as process_image_util
        
        # Конвертируем изображение в PIL если нужно
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        elif not isinstance(image, PILImage.Image):
            raise ValueError(f"Неизвестный тип изображения: {type(image)}")
        
        # Обрабатываем изображение через утилиту (как в rl_dataset.py)
        processed_image = process_image_util(image)
        
        # Формируем сообщения
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Применяем chat template (как в rl_dataset.py)
        raw_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Обрабатываем через processor напрямую (как в rl_dataset.py)
        inputs = self.processor(
            text=[raw_prompt],
            images=[processed_image],
            return_tensors="pt",
        )
        
        # Перемещаем на устройство
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Генерируем ответ
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_response_length,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
            )
        
        # Обрезаем сгенерированные токены (убираем входные)
        input_ids = inputs['input_ids']
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0].strip()
        
        return {
            'prompt': raw_prompt,
            'model_output': response,
            'action': response
        }


    
    def predict_action_id(self, observation: str) -> int:
        """
        Предсказывает действие и возвращает его ID (для Craftext).
        
        Args:
            observation: Текстовое наблюдение из среды
            
        Returns:
            ID действия (int)
        """
        from agent_system.environments.env_package.craftext.projection import craftext_projection
        
        action_text = self.predict(observation)
        action_ids, valids = craftext_projection([action_text])
        return action_ids[0] if valids[0] else -1