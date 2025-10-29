import os
from typing import List, Tuple
### Imports for correct lambda-function parsing, dont remove! 
import jax.numpy as jnp

# from craftext.checkers_jax.building import is_line_formed, is_square_formed
# from craftext.checkers_jax.achivments import conditional_achivments
# from craftext.checkers_jax.conditional import conditional_placing
# from craftext.checkers_jax.relevant import place_object_relevant_to
# from craftext.scenarios.constants import Achievement, MediumInventoryItems,InventoryItems,BlockType

def parse_instructions(file_name_txt: str) -> Tuple[List[str], List[str]]:
    with open(file_name_txt, 'r') as file:
        content = file.read()
    # Split the content into separate dictionaries by '----'
    chunks = content.split('----')
    # Parse each chunk into a dictionary using eval

    instructions_list = []
    correct_chunks = []
    for chunk in chunks:
        try:
            if chunk.strip():
                c = eval(chunk.strip()) 
                instructions_list.append(c)
                correct_chunks.append(chunk)
        except Exception as e:
            pass

    return instructions_list, correct_chunks

def parse_instructions_from_folder(folder_path, return_chunks=False) -> List[str] | Tuple[List[str], List[str]]:
    instructions_list = []
    correct_chunks = []
    print(os.listdir(folder_path))
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            instructions_list_, correct_chunk = parse_instructions(file_path)
            instructions_list.extend(instructions_list_)
            correct_chunks.extend(correct_chunk)
    if return_chunks:
        return instructions_list, correct_chunks
    return instructions_list

def update_previous_dict(previous_dict, folder_path, TASK_NAME='relevant_placement') -> dict[str, List[str] | Tuple[List[str], List[str]]]:
    instructions_list = parse_instructions_from_folder(folder_path)

    for i, parsed_dict in enumerate(instructions_list):
        if 'INSTRUCTION' in parsed_dict:
            previous_dict[f"{TASK_NAME}_INSTRUCTION_{i}"] = parsed_dict['INSTRUCTION']
    

    print()
    print(len(previous_dict.keys()), folder_path)
    print()
    print("---"*30)
    return previous_dict
