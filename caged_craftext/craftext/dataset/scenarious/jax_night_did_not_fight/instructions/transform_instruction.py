import os
import json
from craftext.scenarios.constants import base_path, BlockType
from craftext.checkers.building_line import is_line_formed
from craftext.scenarios.parce_dataset import update_previous_dict

def transform_instruction(instruction: json):
    """
    Преобразует одну инструкцию:
    - Удаляет поле 'check_lambda', так как его нельзя сохранить в JSON.
    - Оставляет 'str_check_lambda' для последующего восстановления логики.
    """
    """
    'INSTRUCTION_SWORD_ZOMBIE_ONLY': {
        'instruction': "Craft a wooden sword and defeat a zombie, but do not attack a skeleton.",
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS.value, 
        'instruction_paraphrases': [
            "Forge a wooden blade and slay a zombie, but avoid fighting skeletons.",
            "Your task is to craft a wooden sword and take down a zombie, not a skeleton.",
            "Make a wooden sword and hunt zombies, but leave skeletons alone.",
            "Only zombies are your target—skeletons must remain untouched.",
            "Defeat a zombie using a wooden sword, but do not engage a skeleton.",
            "Make a sword from wood and fight zombies, but do not challenge a skeleton.",
            "Slay the undead, but only the ones that rot—skeletons are not to be harmed.",
            "Craft a wooden weapon and battle zombies, not bone warriors.",
            "The wooden sword is for zombies only, not for skeletons.",
            "Destroy the walking dead, but ignore the ones without flesh."
        ],
        "arguments": create_target_state([Achievement.MAKE_WOOD_SWORD.value, Achievement.DEFEAT_ZOMBIE.value], [Achievement.DEFEAT_SKELETON.value]),
        'str_check_lambda': 'conditional_achivments(gd, jnp.array([1 if a == Achievement.MAKE_WOOD_SWORD.value else 1 if a == Achievement.DEFEAT_ZOMBIE.value else -1 if a == Achievement.DEFEAT_SKELETON.value else 0 for a in range(Achievement.MAKE_IRON_SWORD.value+1)]))'
    }
    
    {
    "INSTRUCTION": {
        "instruction": "Check for a line of plants with a length of three",
        "instruction_paraphrases": [
            "Verify if there's a row of three flora in the game",
            "Inspect the game map for a straight line formation of vegetation that consists of three blocks ",
            "Ascertain if there's a series of three plant blocks in a row on the playing field",
            "Investigate if a linear arrangement of three flora blocks is present in the current game state",
            "Can you confirm the existence of a straight sequence of three vegetation units in line on the game?"
        ],
         'check_lambda': lambda game_data, ix: is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=False),
        "str_check_lambda": "is_line_formed(game_data, ix, BlockType.PLANT, 3, check_diagonal=False)"
    }
}
"""
    
    # При необходимости можно добавить другие преобразования
    return instruction

def process_instructions_file(input_filepath, output_filepath):
    """
    Обрабатывает один файл с инструкциями:
      - Читает данные из JSON.
      - Преобразует каждую инструкцию.
      - Сохраняет результат в новый JSON-файл.
    """
    with open(input_filepath, "r", encoding="utf-8") as infile:
        data = infile.read()
    data = list(filter(lambda x: x != '', data.split("----\n")))
    print(len(data))
    for i, instruction in enumerate(data[:]):
        # instruction = instruction.replace("\'", "\"")
        print(instruction)
        instruction = instruction.split("\n")
        instruction[-5:] = instruction[-4:]
        del instruction[-1]
        print(*instruction, sep="\n")
        instruction = json.loads("\n".join(instruction))
        instruction = transform_instruction(instruction)
        data[i] = json.dumps(instruction, ensure_ascii=False, indent=4)
    # print(data[-1])
    # print(data[-1])
    # Предполагаем, что данные – это словарь, где ключи – идентификаторы инструкций.
    # transformed_data = {}
    # for key, instr in data.items():
    #     transformed_data[key] = transform_instruction(instr)
    print(data[-1])
    transformed_data = "\n----\n".join(data)
    # with open(output_filepath, "w", encoding="utf-8") as outfile:
    #     json.dump(transformed_data, outfile, ensure_ascii=False, indent=4)
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        outfile.write(transformed_data)
         
    print(f"Файл обработан и сохранён: {output_filepath}")

def process_directory(input_dir, output_dir):
    """
    Обрабатывает все JSON-файлы из входной директории и сохраняет их в выходной директории.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)
            process_instructions_file(input_filepath, output_filepath)

if __name__ == "__main__":
    # Пусть старые файлы для построения лежат в следующих директориях:
    input_easy_dir = os.path.join(".", "instructions/train/easy/")
    input_medium_dir = os.path.join(".", "instructions/train/medium/")
    
    # Новая структура файлов будет сохранена в новых директориях:
    output_easy_dir = os.path.join(".", "instructions/train/easy_transformed/")
    output_medium_dir = os.path.join(".", "instructions/train/medium_transformed/")
    
    process_directory(input_easy_dir, output_easy_dir)
    process_directory(input_medium_dir, output_medium_dir)
