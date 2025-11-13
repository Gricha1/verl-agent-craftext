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
# --- Вспомогательные константы и функции (без изменений) ---

BORING_BLOCKS = {
    BlockType.GRASS.value,
    BlockType.WATER.value,
    BlockType.PATH.value,
    BlockType.SAND.value,
    BlockType.LAVA.value,
}
ACTION_TO_DIRECTION = {"1": "Left", "2": "Right", "3": "Up", "4": "Down"}


def get_detailed_direction_text(dx, dy, threshold=2.0):
    if dx == 0 and dy == 0:
        return "at your feet"
    v_str = ""
    if dy > 0:
        v_str = "in front"
    elif dy < 0:
        v_str = "behind"
    h_str = ""
    if dx > 0:
        h_str = "to the right"
    elif dx < 0:
        h_str = "to the left"
    if not v_str:
        return h_str
    if not h_str:
        return v_str
    if abs(dy) > abs(dx) * threshold:
        return v_str
    elif abs(dx) > abs(dy) * threshold:
        return h_str
    else:
        return f"{v_str} and {h_str}"


# --- Основная функция с исправлениями ---


def render_craftax_text_relative(state, max_targets=4) -> str:
    """
    Создает самый удачный, структурированный и "исполняемый" промпт для агента.
    С ИСПРАВЛЕНИЯМИ для совместимости с JAX.
    """
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    padded_grid = jnp.pad(state.map, (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2), constant_values=BlockType.OUT_OF_BOUNDS.value)
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    points_of_interest = []
    px, py = OBS_DIM[0] // 2, OBS_DIM[1] // 2

    for x in range(OBS_DIM[0]):
        for y in range(OBS_DIM[1]):
            if x == px and y == py:
                continue

            block_val = map_view[x, y]
            # ИСПРАВЛЕНИЕ 1: Преобразуем JAX массив в int перед проверкой
            if int(block_val) not in BORING_BLOCKS:
                points_of_interest.append(
                    {
                        "name": BlockType(int(block_val)).name.title().replace("_", " "),
                        "distance": abs(x - px) + abs(y - py),
                        "rel_x": x - px,
                        "rel_y": y - py,
                    }
                )

    points_of_interest.sort(key=lambda p: p["distance"])

    obs_parts = []
    # ИСПРАВЛЕНИЕ 2: Преобразуем JAX массив в int, а затем в str для ключа словаря
    direction = int(state.player_direction)
    direction_name = ACTION_TO_DIRECTION[str(direction)]

    player_block = BlockType(int(map_view[px, py])).name.lower()
    obs_parts.append(f"CURRENT STATUS:\nYou are standing on {player_block}. You are facing {direction_name}.")

    front_x, front_y = px, py
    if direction == 3:
        front_x -= 1
    elif direction == 4:
        front_x += 1
    elif direction == 1:
        front_y -= 1
    elif direction == 2:
        front_y += 1

    block_in_front_val = map_view[front_x, front_y]
    # ИСПРАВЛЕНИЕ 3: Та же проблема, что и в первом исправлении
    if int(block_in_front_val) not in BORING_BLOCKS:
        block_name = BlockType(int(block_in_front_val)).name.title()
        obs_parts.append(f"DIRECTLY IN FRONT OF YOU:\nThere is a **{block_name}** you can interact with.")

    if points_of_interest:
        obs_parts.append("NEARBY OBJECTS (Closest First):")
        for target in points_of_interest[:max_targets]:
            rx, ry = target["rel_x"], target["rel_y"]
            if direction == 4:
                rx, ry = -rx, -ry
            elif direction == 1:
                rx, ry = ry, -rx
            elif direction == 2:
                rx, ry = -ry, rx

            direction_text = get_detailed_direction_text(rx, -ry)
            obs_parts.append(f"- **{target['name']}**: is {direction_text} ({target['distance']} steps away).")

    obs_parts.append("ACTION HELPER:")
    if direction == 3:
        obs_parts.append("To move FORWARD use UP, to move RIGHT use RIGHT, to move LEFT use LEFT.")
    elif direction == 2:
        obs_parts.append("To move FORWARD use RIGHT, to move RIGHT use DOWN, to move LEFT use UP.")
    elif direction == 4:
        obs_parts.append("To move FORWARD use DOWN, to move RIGHT use LEFT, to move LEFT use RIGHT.")
    elif direction == 1:
        obs_parts.append("To move FORWARD use LEFT, to move RIGHT use UP, to move LEFT use DOWN.")

    inventory_items = []
    for field in state.inventory.__class__.__dataclass_fields__:
        value = getattr(state.inventory, field)
        # ИСПРАВЛЕНИЕ 4: Преобразуем JAX массив в int перед сравнением и форматированием
        if int(value) > 0:
            inventory_items.append(f"{field.replace('_', ' ').title()}: {int(value)}")
    inventory_str = "; ".join(inventory_items) if inventory_items else "Empty"
    obs_parts.append(f"INVENTORY:\n{inventory_str}")

    return "\n\n".join(obs_parts)
