"""
Утилиты для работы с картой Craftext в ноутбуках.
"""
import numpy as np
from craftax.craftax_classic.constants import BlockType


def get_map_info(state, block_type=None):
    """
    Извлекает информацию о всей карте мира из state.
    
    Args:
        state: состояние среды (env_state, должен быть на CPU через jax.device_get)
        block_type: опционально, тип блока для поиска (например, BlockType.DIAMOND)
    
    Returns:
        dict с информацией о карте:
        - map: numpy array всей карты
        - shape: размеры карты (height, width)
        - player_position: позиция игрока [x, y]
        - block_positions: если block_type указан, список позиций этого блока
        - nearest_block: если block_type указан, позиция ближайшего блока и расстояние
    """
    # Получаем карту и позицию игрока
    map_array = np.asarray(state.map)
    player_pos = np.asarray(state.player_position)
    
    result = {
        'map': map_array,
        'shape': map_array.shape,
        'player_position': player_pos.tolist() if hasattr(player_pos, 'tolist') else [int(player_pos[0]), int(player_pos[1])],
    }
    
    # Если указан тип блока, ищем его позиции
    if block_type is not None:
        block_value = int(block_type.value) if hasattr(block_type, 'value') else int(block_type)
        block_positions = np.argwhere(map_array == block_value)
        
        result['block_positions'] = block_positions.tolist()
        result['block_count'] = len(block_positions)
        
        # Находим ближайший блок
        if len(block_positions) > 0:
            # Вычисляем расстояния (манхэттенское расстояние)
            distances = np.abs(block_positions - player_pos).sum(axis=1)
            nearest_idx = np.argmin(distances)
            nearest_pos = block_positions[nearest_idx]
            distance = int(distances[nearest_idx])
            
            result['nearest_block'] = {
                'position': nearest_pos.tolist(),
                'distance': distance,
                'relative_position': (nearest_pos - player_pos).tolist()
            }
        else:
            result['nearest_block'] = None
    
    return result


def find_nearest_block(state, block_type):
    """
    Находит ближайший блок указанного типа к игроку.
    
    Args:
        state: состояние среды (env_state, должен быть на CPU через jax.device_get)
        block_type: тип блока для поиска (например, BlockType.DIAMOND)
    
    Returns:
        dict с информацией о ближайшем блоке:
        - position: [x, y] позиция блока
        - distance: расстояние до блока (манхэттенское)
        - relative_position: относительная позиция [dx, dy] от игрока
        - direction: текстовое описание направления
    """
    map_info = get_map_info(state, block_type)
    
    if map_info.get('nearest_block') is None:
        return None
    
    nearest = map_info['nearest_block']
    dx, dy = nearest['relative_position']
    
    # Определяем направление
    direction_parts = []
    if dy > 0:
        direction_parts.append("вперед")
    elif dy < 0:
        direction_parts.append("назад")
    
    if dx > 0:
        direction_parts.append("вправо")
    elif dx < 0:
        direction_parts.append("влево")
    
    direction = " и ".join(direction_parts) if direction_parts else "на месте"
    
    result = nearest.copy()
    result['direction'] = direction
    result['block_name'] = BlockType(block_type.value if hasattr(block_type, 'value') else block_type).name.lower()
    
    return result

