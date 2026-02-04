
def generate_prompt(template, **kwargs):
    """Generates a single prompt using a template and keyword arguments."""
    return template.format(**kwargs)


def load_resources(instructions_class):
    """Loads resources based on the instruction class."""
    resource_paths = {
        "achivments"                : "prompts/achivments",
        "building_line"             : "prompts/building_line",
        "building_star"             : "prompts/building_star",
        "building_square"           : "prompts/building_square",
        "conditonal_placing"        : "prompts/conditonal_placing",
        "localization_placing"      : "prompts/localization_placing",
        "night_at_shelter"          : "prompts/night_at_shelter",
        "night_did_not_fight"       : "prompts/night_did_not_fight",
        "night_base"                : "prompts/night_base",
        "time_cosntrained_placment" : "prompts/time_cosntrained_placment",
    }
    
    if instructions_class not in resource_paths:
        raise ValueError(f"Unknown instruction class: {instructions_class}")
    
    base_path = resource_paths[instructions_class]
    
    module = __import__(
        f"prompts.{instructions_class}.goal_generator",
        fromlist=['generate_example_goals']
    )
    
    with open(f"{base_path}/template.txt", 'r') as f:
        code = f.read()
    with open(f"{base_path}/example.txt", 'r') as f:
        example = f.read()
    
    return module.generate_example_goals, code, example


def generate_prompts(count_goals, instructions_class, difficulty='EASY'):
    """Generates a list of prompts based on the instruction class and count."""
    # Read base prompt
    print("load promt")
    with open("prompts/instruction_generation.txt", 'r') as f:
        base_prompt = f.read()
    
    print("load resources")
    # Load resources
    generate_example_goals, code, example = load_resources(instructions_class)
 
    print("generate goals")
   # Generate goals
    goals, synonyms = generate_example_goals(num_goals=count_goals, difficulty=difficulty)

    print("generate prompts")
    # Generate prompts
    prompts = [
        generate_prompt(base_prompt, code=code, instruction=goal, example=example, synonym=synonym)
        for goal, synonym in zip(goals, synonyms)
    ]
    return prompts
