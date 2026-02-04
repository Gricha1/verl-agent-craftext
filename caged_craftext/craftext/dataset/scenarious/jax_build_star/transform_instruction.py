import os
import json
from craftext.scenarios.constants import base_path, BlockType, Scenarios, Achievement, AchievementState#, create_target_state
from craftext.checkers.target_state import Achievements, TargetState, BuildStarState
import re
from craftext.checkers.building_star import is_cross_formed
from craftext.scenarios.parce_dataset import update_previous_dict
import jax
from jax import numpy as jnp

from jax.tree_util import Partial  # Используем jax.tree_util.Partial
class JSONEncoderEx(json.JSONEncoder):

    def __init__(self, *, skipkeys, ensure_ascii, check_circular, allow_nan, sort_keys, indent, separators, default):
        super().__init__(skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular,
                         allow_nan=allow_nan, sort_keys=sort_keys, indent=indent, separators=separators,
                         default=default)
        self.item_separator = ""


json.encoder.encode_basestring = lambda x: json.encoder.py_encode_basestring(x)[1:-1]
json.encoder.encode_basestring_ascii = lambda x: json.encoder.py_encode_basestring_ascii(x)[1:-1]

@jax.jit
def _check_func(gd, ix, block_type, size, radius):
    return is_cross_formed(gd, ix, block_type, size, radius)

def create_check_lambda(gd, ix):
    return Partial(_check_func, gd=gd, ix=ix)

def create_target_state(block_type:int, size:int, radius:int, cross_type:int):
    target_achievements = BuildStarState(block_type, size, radius, cross_type)
    return TargetState(building_star=target_achievements)


def transform_instruction(instruction, func, args):
    block_type, size, radius = args
    if instruction == {}:
        raise ValueError("Instruction is empty")
    if instruction.keys() is None:
        raise ValueError("Instruction is empty")
    print(block_type)
    instruction_name =  f'{block_type.split('.')[1].upper()}_{size}'
    template_instruction = {
        f'INSTRUCTION_{instruction_name}':{
            'instruction': f"{instruction["INSTRUCTION"]['instruction']}", 
            "scenario_checker": Scenarios.BUILD_STAR.value, 
            'instruction_paraphrases': instruction["INSTRUCTION"]['instruction_paraphrases'],
            "arguments": f'create_target_state({block_type}, {size}, {radius}, crossType.STRAIGHT)',
            'str_check_lambda': f'{func}(gd, ix)))'
        }
    }
    
    # При необходимости можно добавить другие преобразования
    return template_instruction

def extract_function_call(call_str):
    """
    Извлекает имя функции и список аргументов из строки вызова.
    
    Пример:
      call_str = "is_line_formed(game_data, ix, BlockType.STONE, 4, check_diagonal=False)"
    Вернет:
      ("is_line_formed", ['game_data', 'ix', 'BlockType.STONE', '4', 'check_diagonal=False'])
    """
    open_index = call_str.find("(")
    close_index = call_str.rfind(")")
    
    if open_index == -1 or close_index == -1 or close_index <= open_index:
        return None, []
    
    func_name = call_str[:open_index].strip()
    params_str = call_str[open_index + 1:close_index].strip()
    # Делим строку по запятым и обрезаем пробелы
    params = [p.strip() for p in params_str.split(",")]
    print(params)
    return func_name, params[1:]



def process_instructions_file(input_filepath, output_filepath):
    
    with open(input_filepath, "r", encoding="utf-8") as infile:
        data = infile.read()
    data = list(filter(lambda x: x != '', data.split("----\n")))
    print(len(data))
    for i, instruction in enumerate(data[:]):
        instruction = instruction.split("\n")
        # print(instruction[-4])
        func, args = extract_function_call(instruction[-4])
        instruction[-5] = ''
        
        # print(*instruction, sep="\n")
        instruction = json.loads("\n".join(instruction))
        # print(instruction)
        
        instruction = transform_instruction(instruction, func, args)
        data[i] = json.dumps(instruction, ensure_ascii=False, indent=4)

    transformed_data = "\n----\n".join(data)
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
    # input_easy_dir = os.path.join(".", "instructions/train/easy/")
    input_medium_dir = os.path.join(".", "instructions/train/medium/")
    
    output_easy_dir = os.path.join(".", "instructions/train/easy_transformed/")
    output_medium_dir = os.path.join(".", "instructions/train/medium_transformed/")
    
    # process_directory(input_easy_dir, output_easy_dir)
    # process_directory(input_medium_dir, output_medium_dir)
    
    input_test_easy_other_dir = os.path.join(".", "instructions/test/easy/other_params/")
    input_test_easy_parapshare_dir = os.path.join(".", "instructions/test/easy/paraphrases/")
    input_test_medium_other_dir = os.path.join(".", "instructions/test/medium/other_params/")
    # input_test_medium_parapshare_dir = os.path.join(".", "instructions/test/medium/paraphrases/")
    

    output_test_easy_other_dir = os.path.join(".", "instructions/test/easy_transformed/other_params/")
    output_test_easy_parapshare_dir = os.path.join(".", "instructions/test/easy_transformed/paraphrases/")
    output_test_medium_other_dir = os.path.join(".", "instructions/test/medium_transformed/other_params/")
    # output_test_medium_parapshare_dir = os.path.join(".", "instructions/test/medium_transformed/paraphrases/")
    
    process_directory(input_test_easy_other_dir, output_test_easy_other_dir)
    # process_directory(input_test_easy_parapshare_dir, output_test_easy_parapshare_dir)
    process_directory(input_test_medium_other_dir, output_test_medium_other_dir)
    # process_directory(input_test_medium_parapshare_dir, output_test_medium_parapshare_dir)
