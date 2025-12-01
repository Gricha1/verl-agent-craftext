import jax
import jax.numpy as jnp
from craftax.craftax.constants import MAX_OBS_DIM
from craftax.craftax_classic.constants import OBS_DIM, BlockType
import numpy as np

ACTION_TO_DIRECTION = {"1": "Left", "2": "Right", "3": "Up", "4": "Down"}

# def render_craftax_text(state) -> str:
#     text_obs = ""
#     obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

#     padded_grid = jnp.pad(
#         state.map,
#         (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
#         constant_values=BlockType.OUT_OF_BOUNDS.value,
#     )

#     tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
#     map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

#     def block_name(val):
#         try:
#             return BlockType(int(val)).name.lower()
#         except Exception:
#             return "unknown"

#     mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

    
#     def _add_mob_to_map(carry, mob_index):
#         mob_map, mobs, mob_type_index = carry

#         local_position = mobs.position[mob_index] - state.player_position + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
#         on_screen = jnp.logical_and(local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])).all()
#         on_screen *= mobs.mask[mob_index]

#         mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(on_screen.astype(jnp.uint8))

#         return (mob_map, mobs, mob_type_index), None

#     (mob_map, _, _), _ = jax.lax.scan(
#         _add_mob_to_map,
#         (mob_map, state.zombies, 0),
#         jnp.arange(state.zombies.mask.shape[0]),
#     )
#     (mob_map, _, _), _ = jax.lax.scan(_add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0]))
#     (mob_map, _, _), _ = jax.lax.scan(
#         _add_mob_to_map,
#         (mob_map, state.skeletons, 2),
#         jnp.arange(state.skeletons.mask.shape[0]),
#     )
#     (mob_map, _, _), _ = jax.lax.scan(
#         _add_mob_to_map,
#         (mob_map, state.arrows, 3),
#         jnp.arange(state.arrows.mask.shape[0]),
#     )

#     def mob_id_to_name(id):
#         if id == 0:
#             return "zombie"
#         elif id == 1:
#             return "cow"
#         elif id == 2:
#             return "skeleton"
#         elif id == 3:
#             return "arrow"

#     text_obs += "Map: \n"

#     table = ""
#     for x in range(OBS_DIM[0]):
#         row = ""
#         for y in range(OBS_DIM[1]):
#             tx, ty = x - OBS_DIM[0] // 2, y - OBS_DIM[1] // 2

#             if tx == 0 and ty == 0:
#                 description = f"agent {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
#             elif mob_map[x, y].max() > 0.5:
#                 mob_name = mob_id_to_name(mob_map[x, y].argmax())
#                 description = f"{mob_name} {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
#             else:
#                 block = block_name(map_view[x, y])
#                 description = f"{block} {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"

#             row += description + " "
#         table += row
#         table += "\n"

#     # text_obs += "\n" + tabulate(table, tablefmt="github") + "\n\n"
#     text_obs += str(table) + "\n"
#     # text_obs += "To gather a sapling execute 'do' action" + "\n"

#     text_obs += "Inventory: "
#     for field in state.inventory.__class__.__dataclass_fields__:
#         value = getattr(state.inventory, field)
#         formatted_name = field.replace("_", " ").title()

#         text_obs += f"{formatted_name}: {value}; "

#     text_obs += "\n \n" + "Player Direction: "

#     direction = ACTION_TO_DIRECTION[str(state.player_direction)]
#     text_obs += direction

#     return text_obs

def render_craftax_text(state) -> str:
    """
    CPU-only, numpy-based renderer. Input `state` must be host-converted (e.g. via jax.device_get).
    """
    text_obs = ""
    H, W = OBS_DIM

    # --- 1. Map view (numpy padding + slicing) ---
    pad_width = MAX_OBS_DIM + 2
    padded_grid = np.pad(
        state.map,
        pad_width=pad_width,
        mode='constant',
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    
    # tl_corner = player_pos - [H//2, W//2] + pad_width → top-left of view in padded grid
    tl_x = int(state.player_position[0] - H // 2 + pad_width)
    tl_y = int(state.player_position[1] - W // 2 + pad_width)
    
    # Extract view: [tl_x : tl_x+H, tl_y : tl_y+W]
    map_view = padded_grid[tl_x : tl_x + H, tl_y : tl_y + W]  # shape (H, W)

    # --- 2. Block name helper ---
    def block_name(val):
        try:
            return BlockType(int(val)).name.lower()
        except (ValueError, KeyError):
            return "unknown"

    # --- 3. Build mob_map (H, W, 4) on CPU ---
    mob_map = np.zeros((H, W, 4), dtype=np.uint8)

    def add_mobs(mobs, mob_type_idx):
        # mobs: has .position (N, 2), .mask (N,)
        if not hasattr(mobs, 'position') or len(mobs.position) == 0:
            return
        positions = np.asarray(mobs.position)
        masks = np.asarray(mobs.mask)
        N = len(masks)

        # Compute local positions relative to player
        # local = mob_pos - player_pos + [H//2, W//2]
        local_positions = positions - state.player_position + np.array([H // 2, W // 2])

        for i in range(N):
            if not masks[i]:
                continue
            lx, ly = local_positions[i]
            # Check bounds: 0 <= lx < H, 0 <= ly < W
            if 0 <= lx < H and 0 <= ly < W:
                mob_map[int(lx), int(ly), mob_type_idx] = 1

    # Add all mob types
    add_mobs(state.zombies, 0)
    add_mobs(state.cows, 1)
    add_mobs(state.skeletons, 2)
    add_mobs(state.arrows, 3)

    # --- 4. Mob name helper ---
    def mob_id_to_name(idx):
        names = ["zombie", "cow", "skeleton", "arrow"]
        return names[idx] if 0 <= idx < len(names) else "unknown"

    # --- 5. Build text table ---
    text_obs += "Map: \n"
    rows = []

    for x in range(H):
        row_parts = []
        for y in range(W):
            tx = x - H // 2
            ty = y - W // 2

            if tx == 0 and ty == 0:
                # Agent position
                desc = f"agent {ty}, {-tx}"
            elif mob_map[x, y].max() > 0:  # uint8 — достаточно > 0
                mob_idx = int(mob_map[x, y].argmax())
                mob_name = mob_id_to_name(mob_idx)
                desc = f"{mob_name} {ty}, {-tx}"
            else:
                block_val = int(map_view[x, y])
                block = block_name(block_val)
                desc = f"{block} {ty}, {-tx}"
            row_parts.append(desc)
        rows.append(" ".join(row_parts))

    table = "\n".join(rows)
    text_obs += table + "\n"

    # --- 6. Inventory ---
    text_obs += "Inventory: "
    inventory_items = []
    for field in state.inventory.__class__.__dataclass_fields__:
        value = getattr(state.inventory, field)
        # Convert to int if numpy scalar
        if hasattr(value, 'item'):
            value = value.item()
        elif isinstance(value, (np.integer, np.floating)):
            value = int(value)
        # Skip zero or empty
        if value:
            name = field.replace("_", " ").title()
            inventory_items.append(f"{name}: {value}")
    text_obs += "; ".join(inventory_items) + "; "

    # --- 7. Player direction ---
    text_obs += "\n \nPlayer Direction: "
    try:
        dir_key = str(int(state.player_direction))
        direction = ACTION_TO_DIRECTION.get(dir_key, "Unknown")
    except (ValueError, TypeError):
        direction = "Unknown"
    text_obs += direction

    return text_obs


# Обновите этот список, если у вас другие названия в BlockType
ASCII_MAPPING = {
    "out_of_bounds": "#",
    "unknown": "?",
    "water": "~",
    "grass": ".", 
    "stone": "%", 
    "tree": "T",
    "wood": "w",
    "sand": ":",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "gold": "g",
    "path": "_",
    "table": "T", # Crafting table
    "furnace": "F",
    "plant": "*",
    "ripe_plant": "P",
    "bed": "B",     # Добавил кровать на всякий случай
}

# 0: Up, 1: Right, 2: Down, 3: Left
# PLAYER_SYMBOLS = ["^", ">", "v", "<"]
PLAYER_SYMBOLS = ["@", "@", "@", "@"]

# Zombie, Cow, Skeleton, Arrow
# ИЗМЕНЕНО: Arrow теперь 'a', чтобы не путать с игроком '^'
MOB_SYMBOLS = ["Z", "C", "S", "a"] 

def render_craftax_ascii(state) -> str:
    H, W = OBS_DIM
    ascii_output = ""

    # --- 1. Подготовка данных (карта) ---
    pad_width = MAX_OBS_DIM + 2
    padded_grid = np.pad(
        state.map,
        pad_width=pad_width,
        mode='constant',
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    
    tl_x = int(state.player_position[0] - H // 2 + pad_width)
    tl_y = int(state.player_position[1] - W // 2 + pad_width)
    map_view = padded_grid[tl_x : tl_x + H, tl_y : tl_y + W]

    # Хелпер для получения имени и символа блока
    def get_block_info(val):
        try:
            name = BlockType(int(val)).name.lower()
            char = ASCII_MAPPING.get(name, name[0].upper())
            return char, name
        except:
            return "?", "unknown"

    # --- 2. Подготовка мобов ---
    mob_map = np.zeros((H, W), dtype=np.int32) - 1 # -1 means no mob
    
    def add_mobs(mobs, idx):
        if not hasattr(mobs, 'position'): return
        pos = np.asarray(mobs.position)
        mask = np.asarray(mobs.mask)
        local_pos = pos - state.player_position + np.array([H // 2, W // 2])
        
        for i in range(len(mask)):
            if mask[i]:
                lx, ly = local_pos[i]
                if 0 <= lx < H and 0 <= ly < W:
                    mob_map[int(lx), int(ly)] = idx

    add_mobs(state.zombies, 0)
    add_mobs(state.cows, 1)
    add_mobs(state.skeletons, 2)
    add_mobs(state.arrows, 3)

    # --- 3. Отрисовка карты ---
    ascii_output += "+" + "-" * (W * 2 + 1) + "+\n"
    
    visible_blocks = set()
    current_player_char = "@" 

    for x in range(H):
        row_str = "| "
        for y in range(W):
            char_to_draw = " "
            
            # --- ИГРОК ---
            if x == H // 2 and y == W // 2:
                try:
                    p_dir = int(state.player_direction)
                    if 0 <= p_dir < len(PLAYER_SYMBOLS):
                        char_to_draw = PLAYER_SYMBOLS[p_dir]
                    else:
                        char_to_draw = "@"
                except:
                    char_to_draw = "@"
                current_player_char = char_to_draw
            
            # --- МОБЫ ---
            elif mob_map[x, y] != -1:
                char_to_draw = MOB_SYMBOLS[mob_map[x, y]]
            
            # --- БЛОКИ ---
            else:
                val = int(map_view[x, y])
                visible_blocks.add(val)
                char_to_draw, _ = get_block_info(val)
            
            row_str += char_to_draw + " "
        ascii_output += row_str + "|\n"
    
    ascii_output += "+" + "-" * (W * 2 + 1) + "+\n"

    # --- 4. Статус игрока (Stats) ---
    def get_val(x):
        return x.item() if hasattr(x, 'item') else x

    # Список полей статистики
    stat_fields = ['player_health', 'player_food', 'player_drink', 'player_energy', 'player_mana']
    fallback_fields = ['health', 'food', 'drink', 'energy', 'mana'] 
    
    stats_output = []
    
    # Сбор статов
    all_fields_to_check = stat_fields + fallback_fields
    checked_fields = set()

    for field in all_fields_to_check:
        if hasattr(state, field) and field not in checked_fields:
            val = get_val(getattr(state, field))
            display_name = field.replace("player_", "").title()
            stats_output.append(f"{display_name}: {int(val)}")
            checked_fields.add(field)

    if stats_output:
        ascii_output += "[STATS] " + " | ".join(stats_output) + "\n"

    # --- 5. Инвентарь (Items) ---
    inv_output = []
    
    for field in state.inventory.__class__.__dataclass_fields__:
        # Игнорируем то, что уже в статах
        if field in checked_fields:
            continue
            
        val = get_val(getattr(state.inventory, field))
        if val > 0:
            name = field.replace("_", " ").title()
            inv_output.append(f"{name}: {int(val)}")

    if inv_output:
        ascii_output += "[ITEMS]\n"
        chunk_size = 4
        for i in range(0, len(inv_output), chunk_size):
            ascii_output += "  " + ", ".join(inv_output[i:i+chunk_size]) + "\n"
    else:
        ascii_output += "[ITEMS] (Empty)\n"

    # --- 6. Легенда ---
    legend_items = []
    
    # Игрок и Стрела
    legend_items.append(f"'{current_player_char}': You")
    
    # Добавляем в легенду мобов, если они есть на экране
    if 3 in mob_map: # Arrow
        legend_items.append(f"'{MOB_SYMBOLS[3]}': Arrow")
    if 0 in mob_map: legend_items.append("Z: Zombie")
    if 1 in mob_map: legend_items.append("C: Cow")
    if 2 in mob_map: legend_items.append("S: Skeleton")

    for val in sorted(list(visible_blocks)):
        char, name = get_block_info(val)
        if name != "out_of_bounds":
            legend_items.append(f"'{char}': {name}")
    
    ascii_output += "-" * 20 + "\n"
    ascii_output += "Legend: " + ", ".join(legend_items)
    
    return ascii_output



def add_grid_overlay(image_array, block_pixel_size, grid_color=(255, 255, 255, 200), line_width=1):
    """
    Добавляет сетку на изображение для улучшения понимания пространственной информации.
    
    Args:
        image_array: numpy array изображения (H, W, 3) или (H, W, 4)
        block_pixel_size: размер блока в пикселях (для выравнивания сетки)
        grid_color: цвет сетки в формате RGBA (по умолчанию полупрозрачный белый)
        line_width: толщина линий сетки
    
    Returns:
        numpy array изображения с наложенной сеткой
    """
    import numpy as np
    
    # Конвертируем в numpy array если нужно
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)
    
    # Убеждаемся, что изображение в формате uint8
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Создаем копию изображения
    h, w = image_array.shape[:2]
    result = image_array.copy().astype(np.float32)
    
    # Извлекаем цвет и альфа из grid_color
    if len(grid_color) == 4:
        grid_rgb = np.array(grid_color[:3], dtype=np.float32)
        alpha = grid_color[3] / 255.0
    else:
        grid_rgb = np.array(grid_color[:3], dtype=np.float32)
        alpha = 0.5  # По умолчанию 50% прозрачность
    
    # Создаем маску для линий сетки
    grid_mask = np.zeros((h, w), dtype=bool)
    
    # Отмечаем вертикальные линии
    for x in range(0, w, block_pixel_size):
        x_start = max(0, x - line_width // 2)
        x_end = min(w, x + line_width // 2 + 1)
        grid_mask[:-128, x_start:x_end] = True
    
    # Отмечаем горизонтальные линии
    for y in range(0, h - 128, block_pixel_size):
        y_start = max(0, y - line_width // 2)
        y_end = min(h, y + line_width // 2 + 1)
        grid_mask[y_start:y_end, :] = True
    
    # Применяем альфа-блендинг только к пикселям на линиях сетки
    for c in range(3):  # Для каждого RGB канала
        result[:, :, c] = np.where(
            grid_mask,
            (1 - alpha) * result[:, :, c] + alpha * grid_rgb[c],
            result[:, :, c]
        )
    
    return result.astype(np.uint8)