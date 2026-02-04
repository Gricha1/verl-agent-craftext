import json
import argparse
import re
from enum import Enum

class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21


class InventoryItems(Enum):
    WOOD = 0
    STONE = 1
    COAL = 2
    IRON = 3
    DIAMOND = 4
    SAPLING = 5
    WOOD_PICKAXE = 6
    STONE_PICKAXE = 7
    IRON_PICKAXE = 8
    WOOD_SWORD = 9
    STONE_SWORD = 10
    IRON_SWORD = 11
 

class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36
    
enums = [Achievement, InventoryItems, BlockType]
def validate_instruction_string(instruction_string):
    """
    Validates the instruction dictionary string.
    :param instruction_string: A dictionary-like string representation.
    :return: True if all checks pass, otherwise False and a dictionary of errors.
    """
    import re
    errors = {}

    # Check if all required keys are present
    required_keys = ['instruction', 'instruction_paraphrases', 'check_lambda', 'str_check_lambda']

    try:
        # Safely parse the string using `eval`-free methods
        instruction_dict = eval(instruction_string, {"__builtins__": {}})
        
        if 'INSTRUCTION' not in instruction_dict:
            errors['missing_key'] = 'INSTRUCTION key is missing.'
            return False, errors

        instruction_content = instruction_dict['INSTRUCTION']

        # Check for required keys in INSTRUCTION
        for key in required_keys:
            if key not in instruction_content:
                errors[f'missing_key_{key}'] = f'{key} is missing in INSTRUCTION.'

        # Check `check_lambda` and `check_lambda_str` for improper `.value` usage
        if 'check_lambda_str' in instruction_content:
            lambda_str = instruction_content['check_lambda_str']
            if re.search(r'\.value\b', lambda_str):
                errors['invalid_enum'] = '`check_lambda_str` contains .value usage.'

        # Check for other potential issues
        if 'instruction_paraphrases' in instruction_content:
            if not isinstance(instruction_content['instruction_paraphrases'], list):
                errors['invalid_paraphrases'] = '`instruction_paraphrases` should be a list.'

    except Exception as e:
        errors['parsing_error'] = str(e)
        return False, errors

    return (True, {}) if not errors else (False, errors)


def validate(file):
    """
    Validates a file containing multiple instructions separated by "----".

    Args:
        file (str): The path to the file with instructions.

    Returns:
        str: The name of the saved errors file.
    """
    # Read and split instructions
    with open(file, 'r') as f:
        instructions = f.read()
    instructions_list = instructions.split("----")

    # Validate each instruction
    errors_per_instruction = []
    valid_indices = []
    correct_instructions = []
    for i, instruction in enumerate(instructions_list, start=1):
        correct, errors = validate_instruction_string(instruction)
        if correct:
            correct_instructions.append(instruction)
        errors_per_instruction.append({"instruction_index": i, "validation": errors})
        if correct:
            valid_indices.append(i)

    # Save errors and valid indices to a file
    output_data = {
        "errors": errors_per_instruction,
        "valid_indices": valid_indices
    }
    errors_file = file.replace(".txt", "") + "_validation_results.json"
    with open(errors_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    with open(file.replace(".txt", "")+"_correct.txt", 'w') as f:
        f.write("----".join(correct_instructions))

    return errors_file


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate a file of instructions.")
    parser.add_argument("file", type=str, help="Path to the file containing instructions to validate.")
    args = parser.parse_args()

    # Run validation
    errors_file = validate(args.file)
    print(f"Validation complete. Results saved to: {errors_file}")