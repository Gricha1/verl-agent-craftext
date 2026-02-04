class PlanFormatError(Exception):
    """Raised when the model answer format is invalid or cannot be parsed."""
    pass

class PlanExtractor():
    @staticmethod
    def extract(model_answer, prompt_template):
        if prompt_template == 0: #base prompt
            return BasePlanExtractor.extract(model_answer)
        else:
            return FunctionPlanExtractor.extract(model_answer)
    @staticmethod    
    def extract_prompt(prompt_with_answer, prompt_template):
        if prompt_template == 0: #base prompt
            return BasePlanExtractor.extract_prompt(prompt_with_answer)
        else:
            return FunctionPlanExtractor.extract_prompt(prompt_with_answer)
        
        
class BasePlanExtractor():
    @staticmethod
    def extract(model_answer):
        try:
            formeted_r = ""
            if "Plan:" in model_answer:
                formeted_r = model_answer.split("Plan:")[2]
            elif "1" in model_answer:
                formeted_r = model_answer.split("1")[2]
            else:
                formeted_r = model_answer
            if "Finish!" in model_answer:
                formeted_r = formeted_r.split("Finish!")[0] + "Finish!"
            return formeted_r
        except:
            raise PlanFormatError(f"Incorrect plan format, model answer was {model_answer}")
    @staticmethod
    def extract_prompt(prompt_with_answer):
        return "Plan: ".join(prompt_with_answer.split("Plan: ")[:2])+"Plan: "

import re
class FunctionPlanExtractor():
    @staticmethod
    def clean_subtask(text: str) -> str:
        text_no_quotes = re.sub(r'[\'"“”‘’«»]', '', text)
        cleaned = ' '.join(text_no_quotes.split())
        cleaned = cleaned.replace("),", ")")
        return cleaned
    
    @staticmethod
    def check_subtask(text: str) -> bool:
        cleaned = text
        count_arguments = len(cleaned.split(","))
        count_arguments_eq = cleaned.count("=")

        count_squares_1 = cleaned.count("(")
        count_squares_2 = cleaned.count(")")

        first_check = count_arguments == count_arguments_eq
        second_check = count_squares_1 == count_squares_2
        third_check = count_squares_2 > 0 and count_arguments_eq > 0
        four_check = cleaned.endswith(")")
        

        return first_check and second_check and third_check and four_check


    @staticmethod
    def extract(model_answer):
        try:
            list_starts = model_answer.split("[",)[2]
            list_end = list_starts.split("]",)[0]
            if list_end.count("\n")<2:
                list_end = list_end.replace("\",", "\",\n")
                list_end = list_end.replace("\n \n,", "\n")
                list_end = list_end.replace("\n\n,", "\n")
            subtasks = list_end.split("\n")
            subtasks_clear = [FunctionPlanExtractor.clean_subtask(s) for s in subtasks]
            remove_bad_subtasks = [s for s in subtasks_clear if FunctionPlanExtractor.check_subtask(s)]
            
            if len(remove_bad_subtasks)==0:
                raise PlanFormatError(f"Incorrect plan format: {e}")
                remove_bad_subtasks = ["[MASK]"]
           # print("cleared subtasks: ")
           # print(remove_bad_subtasks)
           # print("- - - - - - - - - - ")
            upd_plan = "\n".join(remove_bad_subtasks)
            return upd_plan
        except Exception as e:
            raise PlanFormatError(f"Incorrect plan format: {e}")
    @staticmethod
    def extract_prompt(prompt_with_answer):
        return "Answer: ".join(prompt_with_answer.split("Answer: ")[:2])+"Answer: "
        
    
BASE_PROMPT = """
        Craftax is a virtual environment designed for exploration, crafting, and task completion. 
        The agent can move, collect resources, craft items, place objects, and interact with its surroundings. 
        Tasks often require gathering resources and crafting items before placing or utilizing them.

        Task: Create a step-by-step action plan (minimum 2, maximum 3 steps) for the agent in Craftax to achieve the instruction.

        Response Format:  
        - Provide the plan as a numbered list.  
        - Each step must describe a specific action or logical task for the agent, such as gathering resources, crafting items, or placing objects.  
        - Keep each step concise (maximum 5 words), clear, and focused on efficiently fulfilling the instruction.  
        - End the list with "6. Finish!"

        For instruction: "Collect wood and Place Table"  
        Plan:  
        1. Find tree  
        2. Collect some wood  
        3. Craft Table  
        4. Place Table  
        5. Repeat if needed  
        6. Finish!  

        For Instruction: $INSTRUCTION$  
        Plan:
"""


PROMPT_WITH_FUNCTIONS = f"""
        You are controlling an agent in a 2D game set within a simplified Minecraft-like environment. 
        The agent starts from scratch with an empty inventory and no gathered resources. 
        Your task is to generate a step-by-step plan that enables the agent to follow a given user instruction.

        What you must do:
        - Break down the instruction into atomic actions the agent needs to perform.
        - Include all necessary preliminary steps, such as gathering or crafting resources.
        - Assume the agent has nothing at the beginning — you must plan from the ground up.
        - Output your answer as a Python list of strings.
        - Each string must represent one atomic skill invocation, written on a separate line.

        Format for each step:
        "skill_name(arg1 = value1, arg2 = value2, ...)"
        - skill_name: the name of the primitive skill or action the agent will execute.
        - Inside the parentheses, list all required arguments with their names and corresponding values.

        Example:
        gather_resource(resource_type = wood)
        make_figure(block_type=stone, figure=square, side_size=2)
        
        Each of the step agents will be implemented without knowledge of what it did before, so it can only rely on observation and the current step. Therefore, each step must be self-sufficient and not require knowledge of past steps.
       
        Instruction: Make a pickaxe from wood
        Answer: 
        ["gather_resource(resource_type = wood)",
        "gather_resource(resource_type = wood)",
        "create_item(item_type = table)",
        "gather_resource(resource_type = wood)",
        "create_item(item_type = wooden pickaxe)"]

        Send your answer as a python list.
        Instruction: $INSTRUCTION$  
        Answer: 
        """
        
# PROMPT_WITH_FUCNTIONS_HINTS = f"""
#         You control an agent in a 2D game with siplified Minecraft environment. You will need to provide a detailed step-by-step plan for following the user's instructions. 
#         You must include all the preliminary steps that it needs to complete.
        
#         You are controlling an agent in a 2D game set within a simplified Minecraft-like environment. 
#         The agent starts from scratch with an empty inventory and no gathered resources. 
#         Your task is to generate a step-by-step plan that enables the agent to follow a given user instruction.

#         What you must do:
#         - Break down the instruction into atomic actions the agent needs to perform.
#         - Include all necessary preliminary steps, such as gathering or crafting resources.
#         - Assume the agent has nothing at the beginning — you must plan from the ground up.
#         - Output your answer as a Python list of strings.
#         - Each string must represent one atomic skill invocation, written on a separate line.

#         Format for each step:
#         "skill_name(arg1 = value1, arg2 = value2, ...)"
#         - skill_name: the name of the primitive skill or action the agent will execute.
#         - Inside the parentheses, list all required arguments with their names and corresponding values.

#         Example:
#         gather_resource(resource_type = wood)
        
#         Each of the step agents will be implemented without knowledge of what it did before, so it can only rely on observation and the current step. Therefore, each step must be self-sufficient and not require knowledge of past steps.

#         Existed skills:
#         {{'gather_resource': "'resource_type'"}}, 
#         {{'place_item': "'item_type'}},
#         {{construct_figure: 'block_name', 'figure_type', 'side_size'}},
#         {{'create_item': "'item_type'"}},
#         {{'defeat_enemy': "'enemy_type'"}},
#         {{'eat': "'food_type'"}},
#         {{'place_item_relative_to_another': "'item_type_to_place', 'item_type_reference', 'direction', 'distance'"}},

#         Existed arguments:
#         resource_type: wood, stone, coal, diamond, iron, plant, water
#         item_type: stone, table, plant, furnace
#         block_name: stone, table, plant, furnace
#         figure_type: line, diagonal_line, square
#         item_type: table, wooden sword, wooden pickaxe, stone sword, stone pickaxe, iron sword, iron pickaxe
#         enemy_type: zombie, skeleton, cow
#         food_type: cow, plant
#         item_type_to_place: stone, table, plant, furnace
#         item_type_reference: stone, table, plant, furnace, water, tree, stone, coal, diamond, iron, plant
#         direction: left, right, bottom, top

#         Send your answer as a python list.
#         Instruction: Make a pickaxe from wood
#         Answer: 
#         ["gather_resource(resource_type = wood)",
#         "gather_resource(resource_type = wood)",
#         "create_item(item_type = table)", 
#         "gather_resource(resource_type = wood", 
#         "gather_resource(resource_type = wood", 
#         "create_item(item_type = wooden pickaxe)"]

#         Send your answer as a python list.
#         Instruction: $INSTRUCTION$  
#         Answer: 
#         """


       
PROMPT_WITH_FUCNTIONS_HINTS = f"""
        You control an agent in a 2D game with siplified Minecraft environment. You will need to provide a detailed step-by-step plan for following the user's instructions. 
        You must include all the preliminary steps that it needs to complete.
        
        You are controlling an agent in a 2D game set within a simplified Minecraft-like environment. 
        The agent starts from scratch with an empty inventory and no gathered resources. 
        Your task is to generate a step-by-step plan that enables the agent to follow a given user instruction.

        What you must do:
        - Break down the instruction into atomic actions the agent needs to perform.
        - Include all necessary preliminary steps, such as gathering or crafting resources.
        - Assume the agent has nothing at the beginning — you must plan from the ground up.
        - Output your answer as a Python list of strings.
        - Each string must represent one atomic skill invocation, written on a separate line.

        Format for each step:
        "skill_name(arg1 = value1, arg2 = value2, ...)"
        - skill_name: the name of the primitive skill or action the agent will execute.
        - Inside the parentheses, list all required arguments with their names and corresponding values.

        Example:
        gather_resource(resource_type = wood)
        
        Each of the step agents will be implemented without knowledge of what it did before, so it can only rely on observation and the current step. Therefore, each step must be self-sufficient and not require knowledge of past steps.

        Existed skills:
        {{'gather_resource': "'resource_type'"}}, 
        {{'place_item': "'item_type'}},
        {{'create_item': "'item_type_to_craft'"}},
        {{'defeat_enemy': "'enemy_type'"}},
        {{'eat': "'food_type'"}},


        Existed arguments:
        resource_type: wood, stone, coal, diamond, iron, plant, water
        item_type: stone, table, plant, furnace
        item_type_to_craft: table, wooden sword, wooden_pickaxe, stone_sword, stone_pickaxe, iron_sword, iron_pickaxe
        enemy_type: zombie, skeleton, cow
        food_type: cow, plant, water

        Send your answer as a python list.
        Instruction: Make a pickaxe from wood
        Answer: 
        ["gather_resource(resource_type = wood)",
        "gather_resource(resource_type = wood)",
        "create_item(item_type = table)", 
        "gather_resource(resource_type = wood", 
        "gather_resource(resource_type = wood", 
        "create_item(item_type = wooden_pickaxe)"]

        Send your answer as a python list.
        Instruction: $INSTRUCTION$  
        Answer: 
        """

INTERACTIVE_PROMPT_WITH_FUNCTIONS = f"""
        You are controlling an agent in a 2D game set within a simplified Minecraft-like environment. 
        The agent starts from scratch with an empty inventory and no gathered resources. 
        Your task is to generate a step-by-step plan that enables the agent to follow a given user instruction.

        What you must do:
        - Break down the instruction into atomic actions the agent needs to perform.
        - Include all necessary preliminary steps, such as gathering or crafting resources.
        - Assume the agent has nothing at the beginning — you must plan from the ground up.
        - Output your answer as a Python list of strings.
        - Each string must represent one atomic skill invocation, written on a separate line.

        Format for each step:
        "skill_name(arg1 = value1, arg2 = value2, ...)"
        - skill_name: the name of the primitive skill or action the agent will execute.
        - Inside the parentheses, list all required arguments with their names and corresponding values.

        Example:
        gather_resource(resource_type = wood)
        make_figure(block_type=stone, figure=square, side_size=2)
        
        Each of the step agents will be implemented without knowledge of what it did before, so it can only rely on observation and the current step. Therefore, each step must be self-sufficient and not require knowledge of past steps.
        
        Existed skills:
        $SKILLS$  
        
        Existed arguments:
        $ARGS$ 
        
        Instruction: Make a pickaxe from wood
        Reflections: 
        - Wood pickaxe making needed a resorce gathering, spicifically wood - ADD NEW SKILL 'gather_resource'
        - Wood pickaxe making needed a table for craft - ADD NEW SKILL'create_item'
        'create_item' also might be used for  wood pickaxe crafting.
        Answer: 
        ["gather_resource(resource_type = wood)",
        "gather_resource(resource_type = wood)",
        "create_item(item_type = table)",
        "gather_resource(resource_type = wood)",
        "create_item(item_type = wooden pickaxe)"]

        Send your answer as a python list.
        Instruction: $INSTRUCTION$ 
        Reflections: 
        """

PROMPTS = [BASE_PROMPT, PROMPT_WITH_FUNCTIONS, PROMPT_WITH_FUCNTIONS_HINTS, INTERACTIVE_PROMPT_WITH_FUNCTIONS]
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def render(self, kwargs) -> str:
        rendered_prompt = self.template
        for key, value in kwargs.items():
            placeholder = f"${key.upper()}$"
            rendered_prompt = rendered_prompt.replace(placeholder, value)
        return rendered_prompt

def promt_instruction(instruction, prompt=BASE_PROMPT):
    prompt_template = PromptTemplate(prompt)
    instruction = instruction
    full_prompt = prompt_template.render({'INSTRUCTION':instruction})
    return full_prompt


def adwanced_promt_instruction(keys_values, prompt=BASE_PROMPT):
    prompt_template = PromptTemplate(prompt)
    full_prompt = prompt_template.render(keys_values)
    return full_prompt