prompt_template = """
    Craftax is a virtual environment designed for exploration, crafting, and task completion. The procedurally generated world includes resources (trees, stones, coal, iron), interactive objects (crafting tables, furnaces, chests), and diverse terrains (water, grass, sand). The agent operates in this dynamic environment, performing actions such as moving, collecting resources, crafting, placing objects, and interacting with surroundings. Completing tasks often involves gathering resources, crafting items, and strategically placing objects.
    The agent has a fixed set of discrete actions:
    Movement: Navigate the map (up, down, left, right).
    Resource Collection: Chop trees, mine stones, or gather coal.
    Crafting: Create tools (e.g., pickaxes) or structures (e.g., furnaces).
    Object Interaction: Use objects (e.g., cook food in a furnace).
    Placement: Place crafted objects in specific locations.
    Task: Create a step-by-step action plan (maximum 5 steps) for the agent in Craftax to achieve the following instruction:
    "{instruction}"
    Response Format:
    1. using only object names exists in Craftax environment provide the plan as a numbered list . 
    2. Each step should outline a specific action or logical task for the agent, such as resource collection, crafting, or object placement.
    3. Keep steps clear, concise, and implementable in Craftax, with a maximum of 5 words per step.

    Check yourself! 
    !ATTENTION! Ensure the plan uses only objects names and actions existing in Craftax. Replace any incorrect terms in the instruction with their correct Craftax equivalents .
    """
 
import json
from tqdm import tqdm
import os
import openai
import argparse

def load_instructions(instructions_path):
    instructions=[]
    with open(instructions_path, "r", encoding="utf-8") as f:
        inst = json.load(f)
        instructions += inst

    return instructions


def main(instructions_path, plans_path):
    """Generates instructions based on prompts."""
    plans = []
    action_plans = {}

    instructions = load_instructions(instructions_path)#[:2]
   # print(prompts[0])
    
    for instruction in tqdm(instructions, desc='Generating Instructions'):
        prompt = prompt_template.format(instruction=instruction)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["choices"][0]["message"]["content"]
        action_plan = response["choices"][0]["message"]["content"]

        action_plans[instruction] = action_plan
        
    with open(plans_path, "w") as f:
        json.dump(action_plans, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate Plans from Prompts')
    parser.add_argument('--instructions_path', type=str, required=True)
    parser.add_argument('--plans_path', type=str, required=True)
    
    args = parser.parse_args()
    
    # Fetch API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    openai.api_key = api_key.strip()
    
    main(args.instructions_path, args.plans_path)
