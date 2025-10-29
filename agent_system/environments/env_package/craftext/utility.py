import jax
import jax.numpy as jnp
from craftax.craftax.constants import MAX_OBS_DIM
from craftax.craftax_classic.constants import OBS_DIM, BlockType

ACTION_TO_DIRECTION = {
    "1": "Left",
    "2": "Right",
    "3": "Up",
    "4": "Down"
}


def render_craftax_text(state) -> str:
    text_obs = ""
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    def block_name(val):
        try:
            return BlockType(int(val)).name.lower()
        except Exception:
            return "unknown"

    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_type_index = carry

        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]

        mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(
            on_screen.astype(jnp.uint8)
        )

        return (mob_map, mobs, mob_type_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )

    def mob_id_to_name(id):
        if id == 0:
            return "zombie"
        elif id == 1:
            return "cow"
        elif id == 2:
            return "skeleton"
        elif id == 3:
            return "arrow"

    text_obs += "Map: \n"

    table = ""
    for x in range(OBS_DIM[0]):
        row = ""
        for y in range(OBS_DIM[1]):
            tx, ty = x - OBS_DIM[0] // 2, y - OBS_DIM[1] // 2

            if tx == 0 and ty == 0:
                description = (
                    f"agent {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
                )
            elif mob_map[x, y].max() > 0.5:
                mob_name = mob_id_to_name(mob_map[x, y].argmax())
                description = f"{mob_name} {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
            else:
                block = block_name(map_view[x, y])
                description = (
                    f"{block} {(y - OBS_DIM[1] // 2)}, {-1 * (x - OBS_DIM[0] // 2)}"
                )

            row += description + " "
        table += row
        table += "\n"

    # text_obs += "\n" + tabulate(table, tablefmt="github") + "\n\n"
    # text_obs += str(table) + "\n"
    text_obs += "To gather a sapling execute 'do' action" + "\n"

    text_obs += "Inventory: "
    for field in state.inventory.__class__.__dataclass_fields__:
        value = getattr(state.inventory, field)
        formatted_name = field.replace("_", " ").title()

        text_obs += f"{formatted_name}: {value}; "

    text_obs += "\n \n" + "Player Direction: "

    direction = ACTION_TO_DIRECTION[str(state.player_direction)]
    text_obs += direction

    return text_obs