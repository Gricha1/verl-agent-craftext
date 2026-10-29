"""Text goals for every classic Craftax achievement, including diamond."""

from craftext.environment.craftext_constants import Achievement, AchievementState, Scenarios
from craftext.environment.scenarious.checkers.target_state import Achievements, TargetState


def target(achievement):
    mask = [AchievementState.NOT_MATTER for _ in range(Achievement.MAKE_IRON_SWORD + 1)]
    mask[achievement] = AchievementState.NEED_TO_ACHIEVE
    return TargetState(achievements=Achievements(achievement_mask=tuple(mask)))


TASKS = (
    ("COLLECT_WOOD", Achievement.COLLECT_WOOD, "Collect wood."),
    ("PLACE_TABLE", Achievement.PLACE_TABLE, "Place a crafting table."),
    ("EAT_COW", Achievement.EAT_COW, "Eat a cow."),
    ("COLLECT_SAPLING", Achievement.COLLECT_SAPLING, "Collect a sapling."),
    ("COLLECT_DRINK", Achievement.COLLECT_DRINK, "Collect a drink."),
    ("MAKE_WOOD_PICKAXE", Achievement.MAKE_WOOD_PICKAXE, "Craft a wooden pickaxe."),
    ("MAKE_WOOD_SWORD", Achievement.MAKE_WOOD_SWORD, "Craft a wooden sword."),
    ("PLACE_PLANT", Achievement.PLACE_PLANT, "Place a plant."),
    ("DEFEAT_ZOMBIE", Achievement.DEFEAT_ZOMBIE, "Defeat a zombie."),
    ("COLLECT_STONE", Achievement.COLLECT_STONE, "Collect stone."),
    ("PLACE_STONE", Achievement.PLACE_STONE, "Place stone."),
    ("EAT_PLANT", Achievement.EAT_PLANT, "Eat a plant."),
    ("DEFEAT_SKELETON", Achievement.DEFEAT_SKELETON, "Defeat a skeleton."),
    ("MAKE_STONE_PICKAXE", Achievement.MAKE_STONE_PICKAXE, "Craft a stone pickaxe."),
    ("MAKE_STONE_SWORD", Achievement.MAKE_STONE_SWORD, "Craft a stone sword."),
    ("WAKE_UP", Achievement.WAKE_UP, "Wake up."),
    ("PLACE_FURNACE", Achievement.PLACE_FURNACE, "Place a furnace."),
    ("COLLECT_COAL", Achievement.COLLECT_COAL, "Collect coal."),
    ("COLLECT_IRON", Achievement.COLLECT_IRON, "Collect iron."),
    ("COLLECT_DIAMOND", Achievement.COLLECT_DIAMOND, "Collect a diamond."),
    ("MAKE_IRON_PICKAXE", Achievement.MAKE_IRON_PICKAXE, "Craft an iron pickaxe."),
    ("MAKE_IRON_SWORD", Achievement.MAKE_IRON_SWORD, "Craft an iron sword."),
)


easy = {
    name: {
        "instruction": instruction,
        "instruction_paraphrases": [],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": target(achievement),
    }
    for name, achievement, instruction in TASKS
}
