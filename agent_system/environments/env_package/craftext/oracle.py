"""
Oracle agent для Craftext среды.
Использует Qwen 2.5 1.5B для ответов на вопросы агента.
Реализован как Ray remote actor для избежания хранения весов в каждом воркере.
"""
import os
import ray
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

# Промпт с знаниями о игре
ORACLE_GAME_KNOWLEDGE_PROMPT = """
# Role
You are an expert guide for the CrafText (based on Craftax) environment. Your goal is to answer questions accurately and concisely based ONLY on the information provided below.

# IMPORTANT: Answering Rules
- Answer ONLY based on the exact information provided in the "Game Mechanics & Rules" section below.
- If you are not certain about the answer based on the provided information, respond with "I don't have enough information to answer this question accurately."
- Do NOT make up or guess information that is not explicitly stated in the rules below.
- Keep your answers short, direct, and factual.
- If the question asks about something not covered in the rules, say "I don't have information about that in my knowledge base."

# Game Mechanics & Rules

### Progression & Mining Requirements:
- **To collect wood:** No tool required. Stand next to a tree and use the DO action.
- **To mine stone:** Requires a `wood_pickaxe` in inventory. Stand next to stone and use DO.
- **To mine iron:** Requires a `stone_pickaxe` in inventory. Stand next to iron and use DO.
- **To mine diamond:** Requires an `iron_pickaxe` in inventory. Stand next to diamond and use DO.

### Crafting & Placing Requirements:
- **To craft wood items (e.g., wood_pickaxe):** Requires `1 wood` in inventory and a `table` (crafting table) placed nearby. You must be adjacent to the `table` and use the appropriate craft action.
- **To craft stone items (e.g., stone_pickaxe):** Requires `1 wood`, `1 stone` in inventory and a `table` nearby. You must be adjacent to the `table` and use the appropriate craft action.
- **To craft iron items (e.g., iron_pickaxe):** Requires `1 wood`, `1 stone`, `1 coal`, `1 iron` in inventory, and both a `table` and `furnace` placed nearby. You must be adjacent to both and use the appropriate craft action.
- **To place table (crafting table):** Requires `2 wood` in inventory. Use the "place table" action.
- **To place furnace:** Requires `1 stone` in inventory. Use the "place furnace" action.
- **To gather sapling:** You must stand on a `grass` block and use the `DO` action.
"""


# Оракул использует CPU по умолчанию, чтобы избежать конфликтов с trainer
# Trainer требует 1 GPU, и если оракул тоже требует GPU, возникает конфликт ресурсов
# 
# Варианты:
# - num_gpus=0 (CPU) - стабильно, но медленнее
# - num_gpus=0.05 (часть GPU) - быстрее, но возможны конфликты
# - num_gpus=1 (полный GPU) - нужен отдельный GPU (см. ORACLE_GPU_FIX_OPTIONS.md)
@ray.remote(num_gpus=0.1)  # Использовать CPU для избежания конфликтов
class CraftextOracle:
    """
    Ray remote actor для оракла.
    Загружает модель один раз и обрабатывает вопросы от разных воркеров.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Инициализация оракла.
        
        Args:
            model_name: Имя модели из HuggingFace
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library is required for oracle. Install with: pip install transformers")
        
        if torch is None:
            raise ImportError("torch library is required for oracle. Install with: pip install torch")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Диагностика GPU
        if torch.cuda.is_available():
            print(f"[Oracle] CUDA available: {torch.cuda.is_available()}")
            print(f"[Oracle] CUDA device count: {torch.cuda.device_count()}")
            print(f"[Oracle] Current CUDA device: {torch.cuda.current_device()}")
            print(f"[Oracle] CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"[Oracle] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        else:
            print(f"[Oracle] WARNING: CUDA not available, will use CPU (very slow!)")
        
        print(f"[Oracle] Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Загружаем модель с правильной конфигурацией GPU
        if self.device == "cuda":
            # Используем device_map="auto" для автоматического распределения
            # или явно указываем устройство
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # Автоматически разместит на доступном GPU
                trust_remote_code=True
            )
            # Убеждаемся, что модель на GPU
            if hasattr(self.model, 'device'):
                if "cuda" not in str(next(self.model.parameters()).device):
                    print(f"[Oracle] WARNING: Model not on CUDA, moving manually...")
                    self.model = self.model.to("cuda")
        else:
            # CPU режим
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        
        # Проверяем, где реально находится модель
        if hasattr(self.model, 'device'):
            actual_device = str(next(self.model.parameters()).device)
        else:
            actual_device = str(next(self.model.parameters()).device)
        
        self.model.eval()
        print(f"[Oracle] Model loaded successfully")
        print(f"[Oracle] Model device (expected): {self.device}")
        print(f"[Oracle] Model device (actual): {actual_device}")
        
        # Проверяем, что модель действительно на нужном устройстве
        if self.device == "cuda" and "cuda" not in actual_device:
            print(f"[Oracle] WARNING: Model expected on CUDA but is on {actual_device}!")
            print(f"[Oracle] This will be VERY slow. Check GPU availability and memory.")
    
    def _get_extended_state_info(self, env_state: Any) -> str:
        """
        Получает расширенную информацию о состоянии среды используя утилиты.
        
        Args:
            env_state: Состояние среды (env_state, должен быть на CPU)
            
        Returns:
            Строка с расширенной информацией о состоянии
        """
        try:
            import numpy as np
            from craftax.craftax_classic.constants import BlockType
            
            # Проверяем, что env_state не None и имеет нужные атрибуты
            if env_state is None:
                return "Extended state info unavailable: env_state is None"
            
            if not hasattr(env_state, 'map') or not hasattr(env_state, 'player_position'):
                return "Extended state info unavailable: env_state missing required attributes"
            
            # Получаем карту и позицию игрока
            # Преобразуем в numpy массивы, обрабатывая возможные JAX структуры
            try:
                # Получаем map и player_position, обрабатывая возможные вложенные структуры
                map_data = env_state.map
                player_pos_data = env_state.player_position
                
                # Проверяем, что это не объекты EnvState
                if hasattr(map_data, '__class__') and 'EnvState' in str(type(map_data)):
                    return "Extended state info unavailable: map is an EnvState object, not an array"
                if hasattr(player_pos_data, '__class__') and 'EnvState' in str(type(player_pos_data)):
                    return "Extended state info unavailable: player_position is an EnvState object, not an array"
                
                # Если это кортеж или список, берем первый элемент
                if isinstance(map_data, (tuple, list)) and len(map_data) > 0:
                    map_data = map_data[0]
                    # Проверяем еще раз после извлечения
                    if hasattr(map_data, '__class__') and 'EnvState' in str(type(map_data)):
                        return "Extended state info unavailable: map[0] is an EnvState object"
                        
                if isinstance(player_pos_data, (tuple, list)) and len(player_pos_data) > 0:
                    player_pos_data = player_pos_data[0]
                    # Проверяем еще раз после извлечения
                    if hasattr(player_pos_data, '__class__') and 'EnvState' in str(type(player_pos_data)):
                        return "Extended state info unavailable: player_position[0] is an EnvState object"
                
                # Преобразуем в numpy массивы, используя try-except для обработки ошибок
                try:
                    map_array = np.asarray(map_data, dtype=np.int32)
                except (ValueError, TypeError) as e:
                    # Если не удалось преобразовать в int32, пробуем без указания типа
                    try:
                        map_array = np.asarray(map_data)
                        # Проверяем тип элементов
                        if map_array.dtype == object:
                            return "Extended state info unavailable: map array contains objects"
                        map_array = map_array.astype(np.int32)
                    except Exception as e2:
                        return f"Extended state info unavailable: error converting map: {str(e2)}"
                
                try:
                    player_pos = np.asarray(player_pos_data, dtype=np.int32)
                except (ValueError, TypeError) as e:
                    # Если не удалось преобразовать в int32, пробуем без указания типа
                    try:
                        player_pos = np.asarray(player_pos_data)
                        # Проверяем тип элементов
                        if player_pos.dtype == object:
                            return "Extended state info unavailable: player_pos array contains objects"
                        player_pos = player_pos.astype(np.int32)
                    except Exception as e2:
                        return f"Extended state info unavailable: error converting player_pos: {str(e2)}"
                
                # Убеждаемся, что это действительно массивы с числовыми типами
                if not isinstance(map_array, np.ndarray) or not isinstance(player_pos, np.ndarray):
                    return "Extended state info unavailable: failed to convert to numpy arrays"
                
                # Проверяем, что массивы содержат числа, а не объекты
                if map_array.dtype == object or player_pos.dtype == object:
                    return "Extended state info unavailable: arrays contain objects instead of numbers"
                    
            except Exception as e:
                import traceback
                return f"Extended state info unavailable: error converting arrays: {str(e)}\n{traceback.format_exc()}"
            
            extended_info = []
            extended_info.append(f"Map size: {map_array.shape}")
            extended_info.append(f"Player position: {player_pos.tolist()}")
            
            # Функция для поиска ближайшего блока
            def find_nearest_block_local(state, block_type):
                try:
                    block_value = int(block_type.value) if hasattr(block_type, 'value') else int(block_type)
                    block_positions = np.argwhere(map_array == block_value)
                    
                    if len(block_positions) == 0:
                        return None
                    
                    # Вычисляем расстояния (манхэттенское расстояние)
                    distances = np.abs(block_positions - player_pos).sum(axis=1)
                    nearest_idx = np.argmin(distances)
                    nearest_pos = block_positions[nearest_idx]
                    
                    # Преобразуем distance в int
                    try:
                        distance_val = distances[nearest_idx]
                        if hasattr(distance_val, '__class__') and 'EnvState' in str(type(distance_val)):
                            return None
                        distance = int(distance_val) if isinstance(distance_val, (int, float, np.integer, np.floating)) else 0
                    except Exception:
                        return None
                    
                    # Преобразуем в Python int для безопасного сравнения
                    try:
                        diff = nearest_pos - player_pos
                        # Проверяем, что diff не содержит объектов EnvState
                        if hasattr(diff, '__class__') and 'EnvState' in str(type(diff)):
                            return None
                        
                        # Преобразуем в список и проверяем каждый элемент
                        diff_list = diff.tolist() if hasattr(diff, 'tolist') else [diff[0], diff[1]]
                        dx, dy = diff_list[0], diff_list[1]
                        
                        # Проверяем и преобразуем каждый элемент
                        if hasattr(dx, '__class__') and 'EnvState' in str(type(dx)):
                            return None
                        if hasattr(dy, '__class__') and 'EnvState' in str(type(dy)):
                            return None
                            
                        dx = int(dx) if isinstance(dx, (int, float, np.integer, np.floating)) else 0
                        dy = int(dy) if isinstance(dy, (int, float, np.integer, np.floating)) else 0
                    except Exception as e:
                        return None
                    
                    # Определяем направление (теперь безопасно, т.к. dx и dy - это int)
                    direction_parts = []
                    if dy > 0:
                        direction_parts.append("forward")
                    elif dy < 0:
                        direction_parts.append("behind")
                    
                    if dx > 0:
                        direction_parts.append("right")
                    elif dx < 0:
                        direction_parts.append("left")
                    
                    direction = " and ".join(direction_parts) if direction_parts else "at your position"
                    
                    return {
                        'position': nearest_pos.tolist(),
                        'distance': distance,
                        'direction': direction
                    }
                except Exception as e:
                    # Если произошла ошибка, возвращаем None
                    return None
            
            # Ищем ближайшие важные ресурсы
            important_blocks = [
                (BlockType.DIAMOND, "Diamond"),
                (BlockType.IRON, "Iron"),
                (BlockType.COAL, "Coal"),
                (BlockType.STONE, "Stone"),
                (BlockType.TREE, "Tree"),
                (BlockType.CRAFTING_TABLE, "Crafting Table"),
                (BlockType.FURNACE, "Furnace"),
            ]
            
            extended_info.append("\nNearest resources:")
            for block_type, name in important_blocks:
                nearest = find_nearest_block_local(env_state, block_type)
                if nearest:
                    extended_info.append(f"  - {name}: {nearest['direction']}, {nearest['distance']} steps away, position {nearest['position']}")
                else:
                    extended_info.append(f"  - {name}: not found on map")
            
            # Добавляем информацию об инвентаре
            if hasattr(env_state, 'inventory'):
                try:
                    inv = env_state.inventory
                    inventory_items = []
                    for attr in ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 
                                'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                                'wood_sword', 'stone_sword', 'iron_sword']:
                        try:
                            if not hasattr(inv, attr):
                                continue
                                
                            val = getattr(inv, attr)
                            
                            # Пропускаем, если val - это не число (например, объект EnvState)
                            if not isinstance(val, (int, float, np.integer, np.floating, np.ndarray)):
                                # Проверяем, не является ли это объектом (не числом)
                                if hasattr(val, '__class__') and 'EnvState' in str(type(val)):
                                    continue
                                # Пытаемся преобразовать через item() если возможно
                                if hasattr(val, 'item'):
                                    try:
                                        val = val.item()
                                    except:
                                        continue
                                else:
                                    continue
                            
                            # Преобразуем в Python int
                            try:
                                if isinstance(val, (int, float)):
                                    val_int = int(val)
                                elif hasattr(val, 'item'):
                                    val_int = int(val.item())
                                elif isinstance(val, (np.integer, np.floating)):
                                    val_int = int(val)
                                elif isinstance(val, np.ndarray):
                                    if val.size > 0:
                                        val_int = int(val.flat[0])
                                    else:
                                        continue
                                else:
                                    continue
                                
                                # Финальная проверка типа перед сравнением
                                if not isinstance(val_int, (int, float)):
                                    continue
                                
                                # Дополнительная проверка: убеждаемся, что val_int - это действительно число
                                try:
                                    # Теперь безопасно сравниваем
                                    if isinstance(val_int, (int, float)) and val_int > 0:
                                        inventory_items.append(f"{attr}: {val_int}")
                                except (TypeError, ValueError) as e:
                                    # Если сравнение не удалось, пропускаем
                                    continue
                            except (TypeError, ValueError, AttributeError) as e:
                                # Пропускаем атрибуты, которые не удалось обработать
                                continue
                        except Exception as e:
                            # Пропускаем атрибуты, которые не удалось обработать
                            continue
                    
                    if inventory_items:
                        extended_info.append(f"\nInventory: {', '.join(inventory_items)}")
                    else:
                        extended_info.append("\nInventory: empty")
                except Exception as e:
                    extended_info.append(f"\nInventory: error reading inventory ({str(e)})")
            
            return "\n".join(extended_info)
            
        except Exception as e:
            print(f"[Oracle] Error getting extended state: {e}")
            import traceback
            traceback.print_exc()
            return f"Extended state info unavailable: {str(e)}"
    
    def _build_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        extended_state: Optional[str] = None
    ) -> str:
        """
        Формирует промпт для оракула.
        
        Args:
            question: Вопрос от агента
            context: Опциональный контекст (текстовое состояние среды)
            extended_state: Опциональная расширенная информация о состоянии
            
        Returns:
            Сформированный промпт
        """
        answer_instructions = (
            "Answer the question in ONLY one sentence based ONLY on the game mechanics rules provided above"
        )
        
        if context:
            answer_instructions += " and the current game state information"
        
        '''
        answer_instructions += (
            ".\n"
            "- If you can answer accurately based on the provided information, give a short, direct answer.\n"
            "- If the question cannot be answered with certainty using the provided information, "
            "respond with: \"I don't have enough information to answer this question accurately.\"\n"
            "- Do NOT guess or make up information."
        )
        '''

        prompt_parts = [ORACLE_GAME_KNOWLEDGE_PROMPT]
        
        if context:
            prompt_parts.append("\n# Current Game State\n")
            
            if extended_state:
                # prompt_parts.append("## Basic State:\n")
                # prompt_parts.append(context)
                prompt_parts.append("\n\n## Extended State (Map Information):\n")
                prompt_parts.append(extended_state)
            else:
                # prompt_parts.append(context)
                pass
        
        prompt_parts.append("\n\n# Question\n")
        prompt_parts.append(question)
        prompt_parts.append("\n\n# Answer Instructions\n")
        prompt_parts.append(answer_instructions)
        prompt_parts.append("\n\nAnswer:")
        
        return "".join(prompt_parts)
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        env_state: Optional[Any] = None,
        max_length: int = 256
    ) -> str:
        """
        Отвечает на вопрос агента.
        
        Args:
            question: Вопрос от агента
            context: Опциональный контекст (например, текущее состояние среды в текстовом виде)
            env_state: Опциональное состояние среды для получения расширенной информации
            max_length: Максимальная длина ответа
            
        Returns:
            Ответ оракла
        """
        try:
            # Получаем расширенную информацию о состоянии (если доступна)
            extended_state = None
            if env_state is not None:
                extended_state = self._get_extended_state_info(env_state)
            
            # Формируем промпт
            prompt = self._build_prompt(question, context, extended_state)
            
            # Токенизация
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Генерация ответа
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"[Oracle] Error answering question: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error: {str(e)}"
    
    def answer_questions_batch(
        self,
        questions_data: List[Dict[str, Any]],
        max_length: int = 256
    ) -> List[str]:
        """
        Отвечает на батч вопросов одновременно, используя настоящую батч-генерацию.
        Все вопросы обрабатываются одним вызовом model.generate(), что значительно быстрее.
        
        Args:
            questions_data: Список словарей с ключами:
                - 'question': str - вопрос от агента
                - 'context': Optional[str] - опциональный контекст
                - 'env_state': Optional[Any] - опциональное состояние среды
            max_length: Максимальная длина ответа
            
        Returns:
            Список ответов оракла (в том же порядке, что и вопросы)
        """
        import time
        total_start = time.time()
        
        num_questions = len(questions_data)
        if num_questions == 0:
            return []
        
        print(f"[Oracle] Processing batch of {num_questions} questions with true batch generation...")
        
        # Шаг 1: Подготовка промптов для всех вопросов
        prep_start = time.time()
        prompts = []
        
        for i, q_data in enumerate(questions_data):
            question = q_data.get('question', '')
            context = q_data.get('context')
            env_state = q_data.get('env_state')
            
            try:
                # Получаем расширенную информацию о состоянии (если доступна)
                extended_state = None
                if env_state is not None:
                    extended_state = self._get_extended_state_info(env_state)
                
                # Формируем промпт
                prompt = self._build_prompt(question, context, extended_state)
                prompts.append(prompt)
                
            except Exception as e:
                print(f"[Oracle] Error preparing question {i+1}: {e}")
                # Добавляем пустой промпт, чтобы сохранить порядок
                prompts.append("")
        
        prep_time = time.time() - prep_start
        
        # Шаг 2: Токенизация батча (с padding)
        tokenize_start = time.time()
        try:
            # Токенизируем все промпты одновременно с padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # Максимальная длина промпта
            ).to(self.device)
            
            # Сохраняем длины входных последовательностей для декодирования
            input_lengths = [inputs['attention_mask'][i].sum().item() for i in range(len(prompts))]
            
        except Exception as e:
            print(f"[Oracle] Error tokenizing batch: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: обрабатываем по одному
            return self._fallback_sequential_processing(questions_data, max_length)
        
        tokenize_time = time.time() - tokenize_start
        
        # Шаг 3: Генерация батча (один вызов для всех вопросов!)
        gen_start = time.time()
        
        # Диагностика перед генерацией
        actual_device = str(next(self.model.parameters()).device)
        print(f"[Oracle] Generating batch on device: {actual_device}")
        if torch.cuda.is_available():
            print(f"[Oracle] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"[Oracle] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        try:
            with torch.no_grad():
                # attention_mask уже включен в **inputs, не нужно передавать отдельно
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except Exception as e:
            print(f"[Oracle] Error in batch generation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: обрабатываем по одному
            return self._fallback_sequential_processing(questions_data, max_length)
        
        gen_time = time.time() - gen_start
        
        # Шаг 4: Декодирование всех ответов
        decode_start = time.time()
        answers = []
        for i in range(num_questions):
            try:
                # Извлекаем только сгенерированную часть (после промпта)
                input_length = input_lengths[i]
                generated_tokens = outputs[i][input_length:]
                
                # Декодируем
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers.append(response.strip())
            except Exception as e:
                print(f"[Oracle] Error decoding answer {i+1}: {e}")
                answers.append(f"I encountered an error: {str(e)}")
        
        decode_time = time.time() - decode_start
        total_time = time.time() - total_start
        
        # Статистика
        avg_time = total_time / num_questions
        print(f"[Oracle] Batch completed: {num_questions} questions in {total_time:.2f}s "
              f"(prep={prep_time:.2f}s, tokenize={tokenize_time:.2f}s, "
              f"gen={gen_time:.2f}s, decode={decode_time:.2f}s, "
              f"avg {avg_time:.3f}s/question, throughput: {num_questions/total_time:.2f} questions/s)")
        
        return answers
    
    def _fallback_sequential_processing(
        self,
        questions_data: List[Dict[str, Any]],
        max_length: int = 256
    ) -> List[str]:
        """
        Fallback метод: обрабатывает вопросы последовательно, если батч-обработка не удалась.
        """
        print(f"[Oracle] Falling back to sequential processing...")
        answers = []
        for q_data in questions_data:
            question = q_data.get('question', '')
            context = q_data.get('context')
            env_state = q_data.get('env_state')
            answer = self.answer_question(question, context, env_state, max_length)
            answers.append(answer)
        return answers
    
    def close(self):
        """Освобождает ресурсы."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

