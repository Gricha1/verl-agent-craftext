import os
from openai import OpenAI

from tqdm import tqdm
from craftext.scenarios.parce_dataset import parse_instructions_from_folder


parafrase_prompt = "prompts/parafrase_generation.txt"
def get_instructions(path):
    instructions_dict, chunks = parse_instructions_from_folder(path, return_chunks=True)
    instructions = [id['INSTRUCTION']['instruction'] for id in instructions_dict if 'INSTRUCTION' in id]
    instructions_dict = [id for id in instructions_dict if 'INSTRUCTION' in id]
    chunks = [ch for ch in chunks if 'INSTRUCTION' in ch]
    return instructions_dict, instructions, chunks

def generate_prompt(template, **kwargs):
    """Generates a single prompt using a template and keyword arguments."""
    return template.format(**kwargs)

def generate_parafrases(path_for_parafrasing, clinet):
    """Generates a list of prompts based on the instruction class and count."""
    # Read base prompt
    with open(parafrase_prompt, 'r') as f:
        base_prompt = f.read()

    instructions_dict, instructions, chunks = get_instructions(path_for_parafrasing)
    print(instructions)
    prompts = [generate_prompt(base_prompt,instruction=i) for i in instructions]

    instructions_parafrases = []
    for prompt in tqdm(prompts, desc='Generating Instructions'):
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        instructions_parafrases.append(content)

    updated_dicts_list = []
    for i, id in enumerate(instructions_dict):


        first = chunks[i].index("[")
        second = chunks[i].index("]")
        chunks[i] = chunks[i].replace(chunks[i][first-1: second+1], instructions_parafrases[i])


        # print("------ REPLACE -------")
        # print( chunks[i])
        # print()
        # print(str(id["INSTRUCTION"]['instruction_paraphrases']))
        # print()
        # print( instructions_parafrases[i])
        # print()

        # print( chunks[i])
        # print("-------"*3)


       # updated_dicts_list.append(id)

   # print(chunks)
    return chunks

if __name__=="__main__":

    # Fetch API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key.strip())

    tasks = ['jax_build_star']
    complecity_lvl = ['easy', 'medium']
    for task in tasks:
        for lvl in complecity_lvl:
            file = f"../craftext/scenarios/{task}/instructions/train/{lvl}"
            chunks = generate_parafrases(file, client)

            file_output = file.replace("/train/", "/test/") + "/paraphrases/test.txt"
            print(len(chunks))
            with open(file_output, "w") as f:
                f.write("")
            for chunck in chunks:
                with open(file_output, "a") as f:
                        f.write(chunck + "\n----\n")
                        
                        
                        
# "craftext/scenarios/jax_time_cosntrained_placment/instructions/test/easy/paraphrases/test.txt"
# "craftext/scenarios/jax_time_cosntested_placment/instructions/test/easy/paraphrases/test.txt"