# Instruction Generator

## Overview
This script generates instructions based on user-defined goals, instruction classes, and difficulty levels using the OpenAI GPT-4 API. The generated instructions are written to an output file.

## Requirements
- Python 3.8+
- OpenAI API key
- Required libraries: `openai`, `tqdm`, `argparse`

## Installation
 Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your_api_key_here'
   ```

## Usage
Run the script with the following arguments:

```bash
python script.py --count_goals <number_of_goals> \
                 --instructions_class <class_of_instructions> \
                 --output_file <output_file> \
                 [--difficulty EASY|MEDIUM|HARD]
```

### Arguments:
- `--count_goals` *(int, required)*: Number of goals to generate instructions for.
- `--instructions_class` *(str, required)*: Class of instructions to generate.
  - Options: `achivments`, `building_line`, `building_squere`, `conditonal_placing`, `localization_placing`, `building_star`
- `--output_file` *(str, required)*: File to save the generated instructions.
- `--difficulty` *(str, optional, default='EASY')*: Difficulty level (`EASY`, `MEDIUM`, `HARD`).

## Example
```bash
python script.py --count_goals 5 \
                 --instructions_class building_line \
                 --output_file instructions.txt \
                 --difficulty MEDIUM
```

## Script Workflow
1. Prompts are generated based on the given parameters (`count_goals`, `instructions_class`, `difficulty`).
2. For each prompt, a request is sent to the OpenAI GPT-4 API.
3. The content from the response is extracted and formatted.
4. The formatted content is written to the specified output file.

## Environment Variables
- `OPENAI_API_KEY`: Required for authenticating with the OpenAI API.

## Output example
```
 \
{
    'INSTRUCTION': 
    {
        'instruction': "Collect five units of coal and then set down two crafting tables.",
        'instruction_paraphrases': 
        [
            "Acquire five pieces of coal and place two workbenches.",
            "Mine five chunks of coal and establish two construction platforms.",
            "Grab five coal and deploy two building tables.",
            "Obtain five nuggets of coal and lay down two fabrication tables.",
            "Gather five lumps of coal and put two manufacturing counters."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.COAL.value, BlockType.CRAFTING_TABLE, 5, 2),
        'check_lambda_str':"conditional_placing(gd, InventoryItems.COAL.value, BlockType.CRAFTING_TABLE, 5, 2)"
    }
}
----
 \
    {'INSTRUCTION': \
        {'instruction': "Gather 2 diamonds and lay down a stone block.",
         'instruction_paraphrases': [
             "Collect 2 diamond resources and place a rock blockade.",
             "Obtain 2 precious gems and construct a stone partition.",
             "Scavenge 2 pieces of diamond precious metal and build a cobble barricade.",
             "Procure 2 shiny diamond deposits and erect a stone wall.",
             "Secure 2 diamond stones and set down a block made of pebble."
        ],
        'check_lambda': lambda gd, ix: conditional_placing(gd, InventoryItems.DIAMOND.value, BlockType.STONE.value, 2, 1),
        'check_lambda_str':"conditional_placing(gd, InventoryItems.DIAMOND.value, BlockType.STONE.value, 2, 1)"
    }\
\
----
```

