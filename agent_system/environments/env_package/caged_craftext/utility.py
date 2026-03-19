import jax
import jax.numpy as jnp
from craftax.craftax.constants import MAX_OBS_DIM, BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax_classic.constants import OBS_DIM, BlockType
# from craftax.craftax.renderer import render
from craftax.craftax_classic.renderer import render_craftax_pixels as render_classic
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from .projection import ACTION_TO_TEXT

ACTION_TO_DIRECTION = {"1": "Left", "2": "Right", "3": "Up", "4": "Down"}


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
# В старом ascii-рендере оставляем нейтральный символ игрока для обратной совместимости.
PLAYER_SYMBOLS = ["@", "@", "@", "@"]
PLAYER_SYMBOLS_V2 = ["<", ">", "^", "v"]

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


def render_craftax_ascii_v2(state) -> str:
    """
    Новый ASCII-шаблон наблюдения:
    - игрок отображается направлением (с fallback до '@')
    - стрелы отображаются как 'a', чтобы не путать со стрелкой игрока
    - формат карты/статов/инвентаря/легенды совпадает с текущим ascii
    """
    H, W = OBS_DIM
    ascii_output = ""

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

    def get_block_info(val):
        try:
            name = BlockType(int(val)).name.lower()
            char = ASCII_MAPPING.get(name, name[0].upper())
            return char, name
        except Exception:
            return "?", "unknown"

    mob_map = np.zeros((H, W), dtype=np.int32) - 1
    mob_symbols = ["Z", "C", "S", "a"]

    def add_mobs(mobs, idx):
        if not hasattr(mobs, "position"):
            return
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

    ascii_output += "+" + "-" * (W * 2 + 1) + "+\n"

    visible_blocks = set()
    current_player_char = "@"

    for x in range(H):
        row_str = "| "
        for y in range(W):
            char_to_draw = " "

            if x == H // 2 and y == W // 2:
                try:
                    p_dir_raw = None
                    if hasattr(state, "player_direction"):
                        p_dir_raw = state.player_direction
                    elif hasattr(state, "variables") and hasattr(state.variables, "player_direction"):
                        p_dir_raw = state.variables.player_direction

                    if p_dir_raw is not None:
                        if hasattr(p_dir_raw, "item"):
                            p_dir = int(p_dir_raw.item())
                        elif hasattr(p_dir_raw, "__array__"):
                            p_dir = int(np.asarray(p_dir_raw).item())
                        else:
                            p_dir = int(p_dir_raw)

                        if 0 <= p_dir - 1 < len(PLAYER_SYMBOLS_V2):
                            char_to_draw = PLAYER_SYMBOLS_V2[p_dir - 1]
                        elif 0 <= p_dir < len(PLAYER_SYMBOLS_V2):
                            char_to_draw = PLAYER_SYMBOLS_V2[p_dir]
                        else:
                            char_to_draw = "@"
                    else:
                        char_to_draw = "@"
                except Exception:
                    char_to_draw = "@"
                current_player_char = char_to_draw

            elif mob_map[x, y] != -1:
                char_to_draw = mob_symbols[mob_map[x, y]]

            else:
                val = int(map_view[x, y])
                visible_blocks.add(val)
                char_to_draw, _ = get_block_info(val)

            row_str += char_to_draw + " "
        ascii_output += row_str + "|\n"

    ascii_output += "+" + "-" * (W * 2 + 1) + "+\n"

    def get_val(x):
        return x.item() if hasattr(x, "item") else x

    stat_fields = ["player_health", "player_food", "player_drink", "player_energy", "player_mana"]
    fallback_fields = ["health", "food", "drink", "energy", "mana"]

    stats_output = []
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

    inv_output = []
    for field in state.inventory.__class__.__dataclass_fields__:
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
            ascii_output += "  " + ", ".join(inv_output[i : i + chunk_size]) + "\n"
    else:
        ascii_output += "[ITEMS] (Empty)\n"

    legend_items = [f"'{current_player_char}': You"]

    if 3 in mob_map:
        legend_items.append(f"'{mob_symbols[3]}': Arrow")
    if 0 in mob_map:
        legend_items.append("Z: Zombie")
    if 1 in mob_map:
        legend_items.append("C: Cow")
    if 2 in mob_map:
        legend_items.append("S: Skeleton")

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




class VisualizerWithLLM:
    """
    Класс для визуализации эпизода Craftax с LLM агентом.
    Показывает промпт/вывод модели, верхнюю информационную панель и нижнюю панель распределения политики.
    """

    def __init__(
        self,
        env,
        env_params,
        pixel_render_size=6,  # Увеличено чтобы игра занимала минимум 1/3
        llm_panel_width=800,  # Ширина панели с текстом
        banner_height=80,
        font_size=18,  # Увеличен размер шрифта
        policy_panel_height=250,
    ):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.llm_panel_width = llm_panel_width
        self.banner_height = banner_height
        self.policy_panel_height = policy_panel_height
        self.frames = []

        # env_render_func = render if self.env.environment_key == 1 else render_classic
        env_render_func = render_classic
        self._render_game_fn = jax.jit(env_render_func, static_argnums=(1,))

        self.font_size = font_size
        # Пробуем загрузить шрифт разными способами
        self.font = None
        
        # Список возможных путей к шрифтам (в порядке приоритета)
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",  # Fedora/RHEL
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/Windows/Fonts/arial.ttf",  # Windows
            "arial.ttf",
        ]
        
        # Пробуем найти доступный шрифт
        for font_path in font_paths:
            try:
                self.font = ImageFont.truetype(font_path, font_size)
                print(f"Loaded font from: {font_path} (size: {font_size})")
                break
            except (IOError, OSError, TypeError):
                continue
        
        # Если не нашли системный шрифт, пытаемся найти через fontconfig (Linux)
        if self.font is None:
            try:
                import subprocess
                # Пробуем найти DejaVu через fc-list
                result = subprocess.run(
                    ['fc-list', ':', 'family', 'DejaVu'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout:
                    # Пробуем найти файл шрифта
                    font_file_result = subprocess.run(
                        ['fc-list', 'DejaVu Sans', 'file'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if font_file_result.returncode == 0:
                        font_file = font_file_result.stdout.strip().split('\n')[0].split(':')[-1].strip()
                        try:
                            self.font = ImageFont.truetype(font_file, font_size)
                            print(f"Loaded font via fontconfig: {font_file} (size: {font_size})")
                        except:
                            pass
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass
        
        # Если все еще нет шрифта, используем default (но размер не будет применен)
        if self.font is None:
            try:
                self.font = ImageFont.load_default()
                # Для default font пытаемся создать шрифт с размером через другой способ
                # Пробуем создать временный шрифт для проверки размера
                try:
                    # Пробуем создать шрифт с использованием ImageFont.FreeTypeFont
                    # Но для этого нужен файл шрифта, поэтому просто используем default
                    pass
                except:
                    pass
                print(f"Warning: Using default font (size {font_size} may not apply). "
                      f"Try installing DejaVu Sans: sudo apt-get install fonts-dejavu")
            except Exception as e:
                print(f"Error: Could not load any font! {e}")
                self.font = None
        
        # Проверяем, что шрифт имеет размер
        if self.font is not None:
            try:
                # Проверяем, имеет ли шрифт атрибут size
                if hasattr(self.font, 'size'):
                    actual_size = self.font.size
                    if actual_size != font_size:
                        print(f"Warning: Font size mismatch. Requested: {font_size}, Actual: {actual_size}")
                else:
                    print(f"Warning: Font does not support size attribute. Requested size: {font_size}")
            except:
                pass
        
        # Загружаем моноширинный шрифт для структурированных данных
        self.mono_font = None
        mono_font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansMono.ttf",  # Fedora/RHEL
            "/System/Library/Fonts/Menlo.ttc",  # macOS
            "/Windows/Fonts/cour.ttf",  # Windows Courier
            "/Windows/Fonts/consola.ttf",  # Windows Consolas
        ]
        
        for mono_font_path in mono_font_paths:
            try:
                self.mono_font = ImageFont.truetype(mono_font_path, font_size)
                print(f"Loaded mono font from: {mono_font_path} (size: {font_size})")
                break
            except (IOError, OSError, TypeError):
                continue
        
        # Если не нашли моноширинный шрифт, используем обычный
        if self.mono_font is None:
            self.mono_font = self.font
            print(f"Warning: Mono font not found, using regular font for structured data")

    def _render_game_view(self, env_state):
        """Рендерит игровое поле и возвращает его как PIL Image."""
        pixels = self._render_game_fn(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        return Image.fromarray(np.array(pixels).astype(np.uint8))

    def _create_llm_panel(self, prompt: str, model_output: str, height: int):
        """Создает изображение панели с промптом и выводом модели."""
        panel = Image.new("RGB", (self.llm_panel_width, height), "white")
        draw = ImageDraw.Draw(panel)
        
        y_pos = 15
        padding = 15
        max_width = self.llm_panel_width - 2 * padding
        
        # Заголовок для промпта
        try:
            text_bbox = self.font.getbbox("PROMPT:")
            header_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            header_height = self.font.size
        
        draw.text((padding, y_pos), "PROMPT:", font=self.font, fill=(100, 100, 200))
        y_pos += header_height + 10
        
        # Промпт - обрабатываем многострочный текст с сохранением структуры
        # Сначала разбиваем по переносам строк, затем обрабатываем каждую строку отдельно
        prompt_paragraphs = prompt.split('\n')
        
        # Вычисляем ширину для переноса текста (в символах приблизительно)
        # Для моноширинного шрифта (структурированные данные)
        try:
            if self.mono_font:
                mono_char_width = self.mono_font.getbbox("M")[2] - self.mono_font.getbbox("M")[0]
            else:
                mono_char_width = self.font.getbbox("M")[2] - self.font.getbbox("M")[0]
            if mono_char_width == 0:
                mono_char_width = self.font_size // 2
            wrap_chars_mono = int(max_width / mono_char_width)
        except:
            wrap_chars_mono = int(max_width / (self.font_size // 2))  # fallback
        
        # Для пропорционального шрифта (обычный текст)
        # Используем среднюю ширину символов для более точного расчета
        try:
            # Вычисляем среднюю ширину символов в обычном шрифте
            sample_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"
            total_width = 0
            char_count = 0
            for char in sample_chars:
                try:
                    bbox = self.font.getbbox(char)
                    char_width = bbox[2] - bbox[0]
                    if char_width > 0:
                        total_width += char_width
                        char_count += 1
                except:
                    pass
            if char_count > 0:
                avg_char_width = total_width / char_count
            else:
                avg_char_width = self.font_size // 2
            # Учитываем, что пробелы обычно уже, добавляем небольшой коэффициент
            wrap_chars_text = int(max_width / avg_char_width * 0.9)  # 0.9 для более консервативного переноса
        except:
            wrap_chars_text = int(max_width / (self.font_size // 2))  # fallback
        
        # Обрабатываем каждую строку отдельно, сохраняя информацию о том, является ли она структурированной
        processed_lines = []  # (line, is_structured)
        
        # Определяем, находимся ли мы в блоке карты (для группировки структурированных строк)
        in_map_block = False
        
        for paragraph in prompt_paragraphs:
            if paragraph.strip() == "":
                # Пустая строка - сохраняем как отдельную строку
                # Если мы были в блоке карты, продолжаем считать следующие строки картой
                processed_lines.append(("", False))
                # Сбрасываем флаг блока карты при пустой строке (но можно продолжать если следующая строка структурированная)
            else:
                # Определяем, является ли строка структурированной
                # Структурированные данные обычно содержат:
                # - Множественные пробелы подряд
                # - Начинаются с ключевых слов карты/таблицы
                # - Содержат символы карты: . @ : ~ % c T w и т.д.
                # - Содержат разделители типа +---+ или |---|
                # - Содержат символы типа ":", "|", или форматирование типа таблицы
                
                # Символы, характерные для карты Craftax
                map_chars = set('.@:~%cTwT#?*LPFBrgs')
                has_map_chars = any(char in paragraph for char in map_chars)
                has_separator = '+-' in paragraph or '---' in paragraph  # Разделители типа +---+
                
                # Проверяем паттерн строки карты: начинается с |, содержит пробелы и символы карты
                looks_like_map_row = (
                    paragraph.strip().startswith('|') and paragraph.strip().endswith('|') and
                    (has_map_chars or paragraph.count(' ') > 5)
                )
                
                is_structured = (
                    in_map_block or  # Продолжаем блок карты
                    looks_like_map_row or  # Строка выглядит как строка карты
                    '  ' in paragraph or  # Множественные пробелы
                    paragraph.strip().startswith(('Map:', 'Inventory:', 'Player Direction:', 'Legend:')) or
                    any(keyword in paragraph.lower() for keyword in ['map:', 'inventory:', 'legend:', 'grass', 'stone', 'tree', 'agent', 'wood', 'sapling']) or
                    '|' in paragraph or  # Табличное форматирование
                    has_separator or  # Разделители
                    (has_map_chars and ('|' in paragraph or paragraph.count(' ') > 3)) or  # Строки с символами карты и форматированием
                    paragraph.count(':') > 0 and len(paragraph.split(':')) > 1  # Ключ-значение пары
                )
                
                # Если строка начинается с "Map:", отмечаем что входим в блок карты
                if 'map:' in paragraph.lower():
                    in_map_block = True
                
                if is_structured:
                    # Если мы в структурированном блоке, следующая строка тоже структурированная
                    # (до следующей пустой строки или изменения типа контента)
                    in_map_block = True
                    
                    # Для структурированных данных не делаем перенос по словам - сохраняем структуру
                    # Если строка слишком длинная, разбиваем на части по wrap_chars_mono, но без добавления "..."
                    if len(paragraph) > wrap_chars_mono:
                        # Разбиваем длинную строку на несколько строк по wrap_chars_mono символов
                        # Это сохранит структуру, но позволит отобразить весь контент
                        start = 0
                        while start < len(paragraph):
                            end = start + wrap_chars_mono
                            chunk = paragraph[start:end]
                            processed_lines.append((chunk, True))
                            start = end
                    else:
                        processed_lines.append((paragraph, True))
                else:
                    # Если строка не структурированная, сбрасываем флаг блока карты
                    in_map_block = False
                    # Для обычного текста используем textwrap с правильной шириной
                    # Используем break_long_words=False чтобы не разрывать длинные слова в середине
                    # и break_on_hyphens=True для переноса по дефисам
                    wrapped = textwrap.wrap(
                        paragraph, 
                        width=wrap_chars_text,
                        break_long_words=False,  # Не разрываем длинные слова
                        break_on_hyphens=True,   # Разрешаем перенос по дефисам
                        expand_tabs=False,       # Сохраняем табуляции как есть
                        replace_whitespace=False # Сохраняем оригинальные пробелы где возможно
                    )
                    for wline in wrapped:
                        processed_lines.append((wline, False))
        
        # Ограничиваем количество строк промпта
        max_prompt_lines = min(len(processed_lines), height // (header_height + 5) // 3)
        
        # Рисуем строки, используя соответствующий шрифт
        for line, is_structured in processed_lines[:max_prompt_lines]:
            # Используем моноширинный шрифт для структурированных данных
            font_to_use = self.mono_font if (is_structured and self.mono_font) else self.font
            
            draw.text((padding, y_pos), line, font=font_to_use, fill=(50, 50, 50))
            try:
                text_bbox = font_to_use.getbbox(line)
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_height = font_to_use.size if hasattr(font_to_use, 'size') else self.font_size
            y_pos += text_height + 3
        
        if len(processed_lines) > max_prompt_lines:
            draw.text((padding, y_pos), "...", font=self.font, fill=(100, 100, 100))
            y_pos += header_height + 5
        
        # Разделитель
        y_pos += 10
        draw.line([padding, y_pos, self.llm_panel_width - padding, y_pos], fill=(200, 200, 200))
        y_pos += 15
        
        # Заголовок для вывода модели
        draw.text((padding, y_pos), "MODEL OUTPUT:", font=self.font, fill=(200, 100, 100))
        y_pos += header_height + 10
        
        # Вывод модели
        # Используем тот же расчет ширины для обычного текста (wrap_chars_text из промпта)
        # Если wrap_chars_text не определен, вычисляем его здесь
        try:
            # Используем среднюю ширину символов для расчета
            sample_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"
            total_width = 0
            char_count = 0
            for char in sample_chars:
                try:
                    bbox = self.font.getbbox(char)
                    char_width = bbox[2] - bbox[0]
                    if char_width > 0:
                        total_width += char_width
                        char_count += 1
                except:
                    pass
            if char_count > 0:
                avg_char_width = total_width / char_count
                output_wrap_chars = int(max_width / avg_char_width * 0.9)
            else:
                output_wrap_chars = int(max_width / (self.font_size // 2))
        except:
            output_wrap_chars = int(max_width / (self.font_size // 2))
        
        output_lines = textwrap.wrap(
            model_output, 
            width=output_wrap_chars,
            break_long_words=False,  # Не разрываем длинные слова
            break_on_hyphens=True,   # Разрешаем перенос по дефисам
            expand_tabs=False,       # Сохраняем табуляции как есть
            replace_whitespace=False # Сохраняем оригинальные пробелы где возможно
        )
        # Ограничиваем количество строк вывода
        max_output_lines = min(len(output_lines), (height - y_pos - 20) // (header_height + 5))
        for line in output_lines[:max_output_lines]:
            draw.text((padding, y_pos), line, font=self.font, fill=(50, 50, 50))
            try:
                text_bbox = self.font.getbbox(line)
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_height = self.font.size
            y_pos += text_height + 3
        if len(output_lines) > max_output_lines:
            draw.text((padding, y_pos), "...", font=self.font, fill=(100, 100, 100))
        
        return panel

    def _add_info_banner(self, image: Image.Image, text: str) -> Image.Image:
        """Добавляет верхнюю информационную панель к изображению."""
        img_with_banner = Image.new(
            "RGB", (image.width, image.height + self.banner_height), "white"
        )
        img_with_banner.paste(image, (0, self.banner_height))

        draw = ImageDraw.Draw(img_with_banner)

        # Логика для поддержки ручных (\n) и автоматических переносов
        try:
            char_width = self.font.getbbox("A")[2] - self.font.getbbox("A")[0]
            if char_width == 0: 
                raise AttributeError
            wrap_width = image.width // char_width
        except (AttributeError, TypeError):
            wrap_width = image.width // (self.font.size // 2)

        # Разделяем текст по ручным переносам \n
        manual_lines = text.split('\n')
        
        all_lines_to_render = []
        # Каждую ручную строку дополнительно переносим автоматически
        for manual_line in manual_lines:
            wrapped_lines = textwrap.wrap(manual_line.strip(), width=wrap_width)
            all_lines_to_render.extend(wrapped_lines)

        # Отрисовываем все получившиеся строки
        y_pos = 5
        for line in all_lines_to_render:
            draw.text((10, y_pos), line, font=self.font, fill=(0, 0, 0))
            try:
                text_bbox = self.font.getbbox(line)
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_height = self.font.size
            y_pos += text_height + 3
            
        return img_with_banner
    
    def _create_policy_panel(self, policy_distribution, width):
        """Создает изображение панели с гистограммой распределения действий."""
        panel = Image.new("RGB", (width, self.policy_panel_height), "white")
        draw = ImageDraw.Draw(panel)
        policy_probs = np.asarray(policy_distribution).flatten()
        num_actions = len(policy_probs)
        if num_actions == 0: 
            return panel

        padding = 20
        top_padding = 30
        bottom_padding = 130
        chart_area_height = self.policy_panel_height - top_padding - bottom_padding
        if chart_area_height <= 0: 
            chart_area_height = 1

        bar_spacing = 2
        bar_width = (width - padding * 2 - bar_spacing * (num_actions - 1)) / num_actions
        
        for i, prob in enumerate(policy_probs):
            bar_height = prob * chart_area_height
            x0, y0 = padding + i * (bar_width + bar_spacing), self.policy_panel_height - bottom_padding - bar_height
            x1, y1 = x0 + bar_width, self.policy_panel_height - bottom_padding
            draw.rectangle([x0, y0, x1, y1], fill="cornflowerblue")
            
            # Надпись сверху (вероятность)
            prob_text = f"{prob:.2f}"
            try:
                text_bbox = self.font.getbbox(prob_text)
                text_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_width = self.font.getlength(prob_text)
            
            draw.text((x0 + (bar_width - text_width) / 2, y0 - 15), prob_text, font=self.font, fill="black")

            # Надпись снизу (название действия)
            action_text = ACTION_TO_TEXT[i] if i < len(ACTION_TO_TEXT) else f"Action{i}"
            
            try:
                bbox = self.font.getbbox(action_text)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                size = self.font.getsize(action_text)
                bbox = (0, 0, size[0], size[1])
                text_width, text_height = size
            
            text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            # Рисуем текст со смещением, чтобы он не обрезался
            text_draw.text((-bbox[0], -bbox[1]), action_text, font=self.font, fill="black")
            
            rotated_text = text_img.rotate(60, expand=True, resample=Image.BICUBIC)
            
            bar_center_x = x0 + bar_width / 2
            paste_x = int(bar_center_x - rotated_text.width / 2)
            paste_y = int(y1 + 5)

            panel.paste(rotated_text, (paste_x, paste_y), rotated_text)

        draw.line([padding, y1, width - padding, y1], fill="black")
        return panel

    def render_step(
        self,
        env_state,
        action,
        prompt: str,
        model_output: str,
        step_num: int,
        reward: float,
        instruction: str,
        success_rate: float = 0.0,
        instruction_done: float = 0.0,
        policy_distribution=None,
        value: float = None,
        action_text: str = None,
        is_action_valid: bool = None,
        cost: float = None,  # Для Caged Craftext - стоимость шага/эпизода
        constraint: str = None,  # Для Caged Craftext - текстовое ограничение
    ):
        """
        Рендерит полный кадр (игра + панель LLM + инфо-баннер + гистограмма).
        
        Args:
            env_state: Состояние среды
            action: ID действия (int)
            prompt: Промпт, который был отправлен модели
            model_output: Полный вывод модели
            step_num: Номер шага
            reward: Награда за шаг
            instruction: Инструкция/задача
            success_rate: Процент успешности (опционально)
            instruction_done: Прогресс выполнения инструкции (опционально)
            policy_distribution: Распределение вероятностей действий (опционально)
            value: Значение value функции (опционально)
            action_text: Текстовое представление действия (если None, берется из ACTION_TO_TEXT)
            is_action_valid: Валидность действия (опционально)
        """
        # 1. Рендерим игру и панель LLM
        game_img = self._render_game_view(env_state)
        llm_panel = self._create_llm_panel(prompt, model_output, game_img.height)

        # 2. Убеждаемся, что игра занимает минимум 1/3 от общей ширины
        # Вычисляем минимальную ширину игры (1/3 от общей ширины с учетом llm_panel)
        # Если llm_panel_width = X, то игра должна быть минимум X/2 (чтобы игра была 1/3 от total = X + X/2 = 1.5X)
        min_game_width = llm_panel.width / 2
        
        # Если игра меньше минимальной ширины, масштабируем её
        # Но используем NEAREST для пиксельной графики (без размытия)
        if game_img.width < min_game_width:
            scale_factor = min_game_width / game_img.width
            new_game_width = int(game_img.width * scale_factor)
            new_game_height = int(game_img.height * scale_factor)
            # Используем NEAREST для пиксельной графики - без размытия
            game_img = game_img.resize((new_game_width, new_game_height), Image.Resampling.NEAREST)
            # Обновляем высоту llm_panel под новую высоту игры
            llm_panel = self._create_llm_panel(prompt, model_output, game_img.height)

        # 3. Объединяем их горизонтально
        combined_width = game_img.width + llm_panel.width
        combined_img = Image.new("RGB", (combined_width, game_img.height))
        combined_img.paste(game_img, (0, 0))
        combined_img.paste(llm_panel, (game_img.width, 0))
        
        # 3. Формируем текст для баннера
        action_display = action_text if action_text else (
            ACTION_TO_TEXT[action] if isinstance(action, int) and action < len(ACTION_TO_TEXT) 
            else str(action)
        )
        
        valid_text = ""
        if is_action_valid is not None:
            valid_text = f" | Action Valid: {is_action_valid}"
        
        value_text = ""
        if value is not None:
            value_text = f" | Value: {value:.4f}"
        
        # Добавляем информацию о cost и constraint для Caged Craftext
        cost_text = ""
        if cost is not None:
            cost_text = f" | Cost: {cost:.4f}"
        
        constraint_text = ""
        if constraint:
            constraint_text = f"\nConstraint: {constraint[:150]}"
        
        raw_info_text = (
            f"Step: {step_num} | Reward: {reward:.4f}{cost_text} | "
            f"Instruction Done: {instruction_done:.2f} | Success Rate: {success_rate:.2f}\n"
            f"Instruction: {instruction}{constraint_text}\n"
            f"Action: {action_display}{valid_text}{value_text}"
        )

        info_text = raw_info_text.replace("'", "'")
        
        # 4. Добавляем баннер к объединенному изображению
        img_with_banner = self._add_info_banner(combined_img, info_text)

        # 5. Создаем панель с гистограммой (если есть распределение политики)
        if policy_distribution is not None:
            policy_panel = self._create_policy_panel(policy_distribution, img_with_banner.width)
            
            # 6. Создаем финальный холст и объединяем все части
            final_frame = Image.new(
                "RGB",
                (img_with_banner.width, img_with_banner.height + self.policy_panel_height),
                "white"
            )
            final_frame.paste(img_with_banner, (0, 0))
            final_frame.paste(policy_panel, (0, img_with_banner.height))
        else:
            final_frame = img_with_banner

        self.frames.append(final_frame)

    def save_gif(self, filename="episode.gif", duration=300):
        """Сохраняет накопленные кадры в GIF файл."""
        if not self.frames:
            print("Нет кадров для сохранения.")
            return
        self.frames[0].save(
            filename, save_all=True, append_images=self.frames[1:], duration=duration, loop=0
        )
        print(f"GIF сохранен в {filename}")

    def clear_frames(self):
        """Очищает список кадров для новой визуализации."""
        self.frames = []