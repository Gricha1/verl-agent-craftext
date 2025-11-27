"""
Функции для получения правильных ответов на вопросы о состоянии мира Craftext.
Эти функции работают с craftext_state (env_state) и могут использоваться для
создания датасета с вопросами и правильными ответами.
"""
import numpy as np
from craftax.craftax.constants import MAX_OBS_DIM
from craftax.craftax_classic.constants import OBS_DIM, BlockType


def _get_block_type_from_name(block_name: str):
    """
    Преобразует название блока в значение BlockType enum.
    Поддерживает различные варианты названий.
    """
    block_name_lower = block_name.lower().strip()
    
    # Маппинг названий на BlockType
    name_to_block = {
        "tree": BlockType.TREE,
        "stone": BlockType.STONE,
        "water": BlockType.WATER,
        "sand": BlockType.SAND,
        "iron": BlockType.IRON,
        "iron ore": BlockType.IRON,
        "coal": BlockType.COAL,
        "grass": BlockType.GRASS,
        "wood": BlockType.WOOD,
        "path": BlockType.PATH,
        "diamond": BlockType.DIAMOND,
        "crafting table": BlockType.CRAFTING_TABLE,
        "furnace": BlockType.FURNACE,
        "plant": BlockType.PLANT,
        "ripe plant": BlockType.RIPE_PLANT,
        "lava": BlockType.LAVA,
        "zombie": "zombie",  # Это моб, не блок
        "cow": "cow",
        "skeleton": "skeleton",
        "arrow": "arrow",
    }
    
    if block_name_lower in name_to_block:
        return name_to_block[block_name_lower]
    else:
        # Попробуем найти по частичному совпадению
        for name, block_type in name_to_block.items():
            if name in block_name_lower or block_name_lower in name:
                return block_type
        return None


def _get_visible_map_view(state):
    """
    Извлекает видимую область карты относительно позиции игрока.
    Возвращает map_view (H, W) и mob_map (H, W, 4).
    """
    H, W = OBS_DIM
    
    # Создаем padded grid
    pad_width = MAX_OBS_DIM + 2
    padded_grid = np.pad(
        state.map,
        pad_width=pad_width,
        mode='constant',
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    
    # Вычисляем top-left corner видимой области
    tl_x = int(state.player_position[0] - H // 2 + pad_width)
    tl_y = int(state.player_position[1] - W // 2 + pad_width)
    
    # Извлекаем видимую область
    map_view = padded_grid[tl_x : tl_x + H, tl_y : tl_y + W]
    
    # Создаем mob_map
    mob_map = np.zeros((H, W, 4), dtype=np.uint8)
    
    def add_mobs(mobs, mob_type_idx):
        if not hasattr(mobs, 'position') or len(mobs.position) == 0:
            return
        positions = np.asarray(mobs.position)
        masks = np.asarray(mobs.mask)
        N = len(masks)
        
        local_positions = positions - state.player_position + np.array([H // 2, W // 2])
        
        for i in range(N):
            if not masks[i]:
                continue
            lx, ly = local_positions[i]
            if 0 <= lx < H and 0 <= ly < W:
                mob_map[int(lx), int(ly), mob_type_idx] = 1
    
    add_mobs(state.zombies, 0)
    add_mobs(state.cows, 1)
    add_mobs(state.skeletons, 2)
    add_mobs(state.arrows, 3)
    
    return map_view, mob_map


def _find_nearest_block_in_view(map_view, mob_map, target_type, target_name: str):
    """
    Находит ближайший блок/моб заданного типа в видимой области.
    Возвращает список позиций (x, y) в координатах наблюдения (относительно агента).
    Агент находится в центре наблюдения (H//2, W//2), что соответствует (0, 0) в относительных координатах.
    """
    H, W = OBS_DIM
    center_x, center_y = H // 2, W // 2
    positions = []
    
    # Проверяем, является ли цель мобом
    is_mob = target_name.lower() in ["zombie", "cow", "skeleton", "arrow"]
    
    if is_mob:
        mob_idx_map = {"zombie": 0, "cow": 1, "skeleton": 2, "arrow": 3}
        mob_idx = mob_idx_map.get(target_name.lower(), -1)
        if mob_idx >= 0:
            for x in range(H):
                for y in range(W):
                    if mob_map[x, y, mob_idx] > 0:
                        # Преобразуем в относительные координаты (как в текстовом наблюдении)
                        rel_x = y - center_y  # x в тексте = y в массиве
                        rel_y = -(x - center_x)  # y в тексте = -x в массиве
                        positions.append((rel_x, rel_y))
    else:
        # Ищем блоки
        if target_type is None:
            return []
        
        target_value = target_type.value if hasattr(target_type, 'value') else target_type
        
        for x in range(H):
            for y in range(W):
                block_val = int(map_view[x, y])
                if block_val == target_value:
                    # Преобразуем в относительные координаты (как в текстовом наблюдении)
                    rel_x = y - center_y  # x в тексте = y в массиве
                    rel_y = -(x - center_x)  # y в тексте = -x в массиве
                    positions.append((rel_x, rel_y))
    
    return positions


def answer_existence_question(state, block_type_name: str) -> str:
    """
    Отвечает на вопрос: "Is there any {block_type} visible in the current observation?"
    Возвращает "Yes" или "No".
    
    Args:
        state: craftext_state (env_state) - состояние среды
        block_type_name: название блока или моба (например, "tree", "stone", "zombie")
    
    Returns:
        "Yes" или "No"
    """
    map_view, mob_map = _get_visible_map_view(state)
    target_type = _get_block_type_from_name(block_type_name)
    
    positions = _find_nearest_block_in_view(map_view, mob_map, target_type, block_type_name)
    
    return "Yes" if len(positions) > 0 else "No"


def answer_distance_question(state, block_type_name: str) -> int:
    """
    Отвечает на вопрос: "Calculate the Manhattan distance from the agent (0, 0) to the NEAREST {block_type}."
    Возвращает расстояние (целое число) или -1, если блок не найден.
    
    Args:
        state: craftext_state (env_state) - состояние среды
        block_type_name: название блока или моба
    
    Returns:
        Manhattan distance (целое число) или -1, если не найдено
    """
    map_view, mob_map = _get_visible_map_view(state)
    target_type = _get_block_type_from_name(block_type_name)
    
    positions = _find_nearest_block_in_view(map_view, mob_map, target_type, block_type_name)
    
    if len(positions) == 0:
        return -1
    
    # Вычисляем Manhattan distance для всех позиций и берем минимум
    distances = [abs(x) + abs(y) for x, y in positions]
    return min(distances)


def answer_coordinates_question(state, block_type_name: str) -> list:
    """
    Отвечает на вопрос: "Identify the coordinates (x, y) of the NEAREST {block_type}."
    Возвращает координаты всех ближайших блоков (с одинаковым минимальным расстоянием).
    Формат: список строк ["x1, y1", "x2, y2", ...] или пустой список, если блок не найден.
    
    Args:
        state: craftext_state (env_state) - состояние среды
        block_type_name: название блока или моба
    
    Returns:
        Список строк формата ["x1, y1", "x2, y2", ...] или пустой список
    """
    map_view, mob_map = _get_visible_map_view(state)
    target_type = _get_block_type_from_name(block_type_name)
    
    positions = _find_nearest_block_in_view(map_view, mob_map, target_type, block_type_name)
    
    if len(positions) == 0:
        return []
    
    # Вычисляем расстояния для всех позиций
    distances = [(abs(x) + abs(y), (x, y)) for x, y in positions]
    
    # Находим минимальное расстояние
    min_distance = min(dist for dist, _ in distances)
    
    # Находим все позиции с минимальным расстоянием
    nearest_positions = [pos for dist, pos in distances if dist == min_distance]
    
    # Форматируем результат: список строк ["x1, y1", "x2, y2", ...]
    result = [f"{x}, {y}" for x, y in nearest_positions]
    
    return result


def _get_direction_from_coords(x: int, y: int) -> str:
    """
    Определяет направление по координатам относительно агента (0, 0).
    
    Args:
        x: горизонтальное смещение (положительное = право)
        y: вертикальное смещение (положительное = вверх)
    
    Returns:
        Направление как строка
    """
    if x == 0 and y == 0:
        return "none"
    
    # Определяем основные направления
    if x == 0:
        return "up" if y > 0 else "down"
    elif y == 0:
        return "right" if x > 0 else "left"
    else:
        # Диагональные направления
        if y > 0 and x > 0:
            return "up-right"
        elif y > 0 and x < 0:
            return "up-left"
        elif y < 0 and x > 0:
            return "down-right"
        else:  # y < 0 and x < 0
            return "down-left"


def answer_direction_question(state, block_type_name: str) -> list:
    """
    Отвечает на вопрос: "In which direction is the nearest {block_type} located relative to the agent?"
    Возвращает список направлений для всех ближайших блоков (с одинаковым минимальным расстоянием).
    Возможные значения: "left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right", "none"
    
    Args:
        state: craftext_state (env_state) - состояние среды
        block_type_name: название блока или моба
    
    Returns:
        Список направлений (без дубликатов) или пустой список, если блок не найден
    """
    map_view, mob_map = _get_visible_map_view(state)
    target_type = _get_block_type_from_name(block_type_name)
    
    positions = _find_nearest_block_in_view(map_view, mob_map, target_type, block_type_name)
    
    if len(positions) == 0:
        return []
    
    # Вычисляем расстояния для всех позиций
    distances = [(abs(x) + abs(y), (x, y)) for x, y in positions]
    
    # Находим минимальное расстояние
    min_distance = min(dist for dist, _ in distances)
    
    # Находим все позиции с минимальным расстоянием
    nearest_positions = [pos for dist, pos in distances if dist == min_distance]
    
    # Определяем направления для всех ближайших позиций
    directions = [_get_direction_from_coords(x, y) for x, y in nearest_positions]
    
    # Убираем дубликаты, сохраняя порядок
    unique_directions = []
    seen = set()
    for direction in directions:
        if direction not in seen:
            unique_directions.append(direction)
            seen.add(direction)
    
    return unique_directions

