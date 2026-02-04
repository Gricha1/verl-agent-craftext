
from craftext.environment.craftext_constants import BlockType, Scenarios, TimeState#, create_target_state_by_contrained_class,
from craftext.environment.scenarious.checkers.target_state import TargetState

from craftext.environment.scenarious.checkers.target_state import TimeCosntrainedPlacmentState as AchievmentClass

def create_target_state_by_contrained_class(block_type: int, time_state: int):
    target_achievements = AchievmentClass(block_type=block_type, time_state=time_state, radius=10)
    tg = TargetState(time_placement=target_achievements)
    # tg.time_placement = target_achievements
    return tg

easy = {
    "INSTRUCTION_FURNACE_DAY": {
        "instruction": "Place a furnace during the day",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "In the daytime, put down a furnace",
            "When it's bright out, install a heating device",
            "Set up an oven in the surroundings illuminated by sunlight",
            "Position a stovetop when daylight is present",
            "During the period of sunshine, establish a heating apparatus"
        ],
        "arguments": create_target_state_by_contrained_class(block_type=BlockType.FURNACE, time_state=TimeState.DAY),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    },

    "INSTRUCTION_STONE_DAY": {
        "instruction": "Place a stone during the day",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Put down a rock when it's daytime",
            "During the sunlight hours, set up a stone",
            "Position a pebble when the sun is up",
            "As the sun shines, lay a stone",
            "At sunrise, insert a boulder in your setting"
        ],
        "arguments": create_target_state_by_contrained_class(BlockType.STONE, TimeState.DAY),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    },

    "INSTRUCTION_FURNACE_NIGHT": {
        "instruction": "Place a furnace during the night",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Set up the furnace at nightfall",
            "Establish the kiln during the evening",
            "Situate the heating unit when it's dark",
            "At dusk, put the stove",
            "Position the heat producer under the moonlight"
        ],
        "arguments": create_target_state_by_contrained_class(BlockType.FURNACE, TimeState.NIGHT),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    },

    "INSTRUCTION_PLANT_DAY": {
        "instruction": "Place a plant during the day.",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "During daytime, set a plant.",
            "When it's light out, put down a flora.",
            "Position a flora in the sunlight.",
            "Situate a greenery while it's daytime.",
            "At sunrise, position some vegetation."
        ],
        "arguments": create_target_state_by_contrained_class(BlockType.PLANT, TimeState.DAY),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    },

    "INSTRUCTION_PLANT_NIGHT": {
        "instruction": "Please place a plant when it's dark outside.",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Could you position the herb during the night?",
            "At nightfall, I'd like you to set down a flora.",
            "When evening comes, make sure you've put a plant in place.",
            "I need you to utilize the cover of darkness to locate a shrub.",
            "Under the cloak of night, please ensure you've established a greenery."
        ],
        "arguments": create_target_state_by_contrained_class(BlockType.PLANT, TimeState.NIGHT),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    },

    "INSTRUCTION_CRAFTING_TABLE_DAY": {
        "instruction": "Place a crafting table during the daytime",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Set a workbench when it's daytime",
            "Put a carpenter's bench out in the daylight",
            "Position a crafting station during the day",
            "During the daylight hours, place the crafting desk",
            "Set the work table in place at sunrise"
        ],
        "arguments": create_target_state_by_contrained_class(BlockType.CRAFTING_TABLE, TimeState.DAY),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, target_state)"
    }
}
    

medium = {
    **easy,
    "INSTRUCTION_FURNACE_MORNING": {
        "instruction": "Place a furnace in the morning",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "During the early hours, set a heater",
            "At sunrise, position a stove",
            "Put a kiln when it's sunrise",
            "Install a heating device as day breaks",
            "Mount a heater during the dawn"
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.FURNACE, TimeState.MORNING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
,
    "INSTRUCTION_FURNACE_EVENING": {
        "instruction": "Place a furnace during the evening",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Set the furnace when it's nighttime.",
            "Position the heating appliance as the sun is setting.",
            "During the evening, ensure the furnace is placed.",
            "As the day turns into night, fix the smelting device in place.",
            "When the twilight sets in, manufacture and position the smelter.",
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.FURNACE, TimeState.EVENING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
,
    "INSTRUCTION_STONE_MORNING": {
        "instruction": "Place a stone during the morning.",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "Set a rock at sunrise.",
            "Position the pebble when it's daytime.",
            "In the morning, lay down the stone.",
            "At daybreak, put a stone in its place.",
            "When morning arrives, place your piece of rock."
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.STONE, TimeState.EVENING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
,
    "INSTRUCTION_STONE_EVENING": {
        "instruction": "Place a stone in the evening",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "During dusk, put a rock down.",
            "Please set a cobble in the nightfall.",
            "As the night begins, position a pebble.",
            "Insert a boulder as the sun is setting.",
            "Position a chunk of stone in position when it's twilight."
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.STONE, TimeState.EVENING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
,
    "INSTRUCTION_ENCHANTMENT_TABLE_ICE_EVENING": {
        "instruction": "Place a enchantment table of ice during the evening",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "At dusk, set down an ice enchantment table",
            "Position a frosty enchanting bench when the sun goes down",
            "Put an icy magician's table in place at twilight",
            "When it's getting dark, install an icy magical workbench",
            "During the time of sundown, establish a frosty charm desk",
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.ENCHANTMENT_TABLE_ICE, TimeState.EVENING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
,
    "INSTRUCTION_ENCHANTMENT_TABLE_FIRE_EVENING": {
        "instruction": "Place a Enchantment_Table_Fire during the evening",
        "scenario_checker": Scenarios.TIME_CONSTRAINED_PLACEMENT ,
        "instruction_paraphrases": [
            "During the evening, set an Enchantment Table of Fire.",
            "When it gets dark, position a Fire Enchantment Table.",
            "Put down an Enchantment_Table_Fire as the evening falls.",
            "At dusk, establish an Enchantment Table with Fire attributes.",
            "In the later part of the day, positioned a fire imbued enchantment table into the game.",
        ],
        "arguments":  create_target_state_by_contrained_class(BlockType.ENCHANTMENT_TABLE_FIRE, TimeState.EVENING),
        "str_check_lambda": "lambda gd, ix: at_time_block_placed(gd, ix)"
    }
}

