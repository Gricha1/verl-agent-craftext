# CrafText

CrafText is an extension of the Craftex environment (<https://github.com/MichaelTMatthews/Craftax>). This extension modifies the environment to be goal-oriented, where the agent's objectives are defined by natural language instructions. The extension includes:

- A set of scenarios: These represent the possible goals an agent might have.
- A set of instructions: These are descriptions of the goals. Each goal can have multiple descriptive variants.
- A set of scenario completion checks: Code corresponding to a specific scenario that takes the agent's state as input and returns a boolean value indicating whether the agent has successfully achieved the goal.

![Place Crafting Table Near Tree](./imgs/tree_cropp.gif) ![Place Crafting Table Near Water](./imgs/water_cropp.gif) ![Make Squere of Stone](./imgs/stone.gif)

## Installation

1. Clone the repository.
2. Create a virtual environment and install the dependencies from `requirements.txt`:

   ```bash
   conda create --name craftext python=3.9
   conda activate craftext
   pip install -r requirements.txt
   ```

3. Navigate to the repository and install the dataset:

   ```bash
   cd CrafText
   pip install -e .
   ```

## Run the PPO Baseline

1. Navigate to the `baselines` directory:

   ```bash
   cd baselines
   ```

2. Run the `ppo_with_instruction.py` script:

   ```bash
   python ppo_with_instruction.py
   ```

3. You can configure the settings for the CrafText dataset (i.e., which instructions to use for training) by setting the `--craftext_settings` flag. You can specify your own configuration or choose one from the `./craftext/configs` directory.

   ```bash
   python ppo_with_instruction.py --craftext_settings simple_build
   ```

4. **Important**: Make sure to specify the same environment for training that is defined in your dataset configuration. For example, if your configuration file specifies `base_environment: Craftax-Classic-Pixels-v1-Text`, you need to include the `--env_name` argument when running the script to match the environment:

   ```bash
   python ppo_with_instruction.py --craftext_settings simple_build --env_name "Craftax-Classic-Pixels-v1-Text"
   ```

This ensures that the correct environment is used during training, matching the one defined in your dataset configuration.

## CrafText dataset configuration file

You can configure a subset of the CrafText dataset for training by specifying different scenario settings. Examples of predefined configurations can be found in the `craftext/configs` folder. To create your own custom configuration, you need to define 4 fields in a YAML file:

- `dataset_key`: Specifies the scenario name. This can be the full name of a specific scenario, such as `build_square`, which will load all tasks involving square structures. Alternatively, you can use broader names like `build` to load all tasks where the agent is required to build something.
- `subset_key`: Defines the complexity of the instructions. Available options include:
  - `ONE`: Simple one-step tasks.
  - `EASY`: Relatively simple instructions.
  - `MEDIUM`: Tasks with moderate complexity.
  
  Choose the appropriate subset based on the training difficulty you want.
  
- `base_environment`: The environment to use during training. For example:
  - `Craftax-Classic-Pixels-v1-Text` – this defines the classic Craftax environment with pixel-based visuals and text instructions.
  
  Ensure that this matches the environment your task is designed for.
  
- `use_paraphrases`: A boolean field (`True` or `False`). Set this to `True` if you want to include paraphrased instructions in your training process, or `False` if you prefer using only the original instructions.

### Example YAML configuration

```yaml
dataset_key: build_square
subset_key: EASY
base_environment: Craftax-Classic-Pixels-v1-Text
use_paraphrases: True
```

### Alternative Configuration Using Environment Variables

Instead of specifying the configuration in a YAML file, you can use the `CRAFTEXT_SETTINGS` environment variable for simpler setups. The format for this variable is as follows:

`
<scenario> && <instruction_type> && <subset>
`

Where:

- `<scenario>`: The specific scenario or task type to use (e.g., `build_line`, `collect_items`).
- `<instruction_type>`: Choose between `pure_instruction` for the original set of instructions or `instruction_with_paraphrases` for instructions with variations.
- `<subset>`: Select the subset to train on, such as `small_train` for simpler instructions, or another custom subset.

#### Example usage

```bash
#!/bin/bash
export CRAFTEXT_SETTINGS="build_line&&pure_instruction&&small_train"
export CRAFTEXT_SETTINGS="build_square&&instruction_with_paraphrases&&medium"
```

In these examples:

- The first setting configures training to use tasks related to building lines with original instructions from the `small_train` subset.
- The second setting loads square building tasks, using paraphrased instructions from the `medium` subset.

This method provides flexible control over the dataset, allowing you to adjust scenarios, instruction types, and subsets on the fly without needing to modify YAML files.

<!-- ## Existed Scenarios 

| Name                                       | Class Name   | Supports JAX      |
|--------------------------------------------|--------------|-------------------|
| build_line                                 | build        | ✅                 |
| squere                                     | build        | ✅                 |
| is_item_in_closed_contour                  | build        | ❌                 |
| cross                                      | build        | ✅                 |
| did_placing_item_increase_variable         | base         | ❌                 |
| was_item_after_increase                    | combo        | ❌                 |
| nerar_increase                             | combo        | ❌                 |
| item_after_another_contour                 | combo        | ❌                 |
| coutour_placing_item_increase_var          | combo        | ❌                 |
| after_another_near_objects                 | combo        | ❌                 |
| was_item_placed_near_another               | localization | ❌                 |
| place                                      | localization | ✅                 |
| water_sources                              | localization | ❌                 |
| old_place_near_game_block                  | localization | ❌                 |
| was_item_collected_after_another_object    | conditional  | ❌                 | -->

## Dataset Generation Details

### Instruction and Checker Generation Pipeline

1. Come up with the scenario.
2. Use the standard checker functions and scenario format to write the code for verifying the scenario. Look at the examples (<https://github.com/ZoyaV/CrafText/blob/main/checkers/scenarius.py>)
3. Use the Instruction Generation Prompt and AskTheCode(ChatGPT4o) to create examples of scenario instructions.

### Instruction Generation Prompt

The code for verifying played scenarios can be found at the following repository link:

<https://github.com/ZoyaV/CrafText/blob/main/checkers/scenarius.py>

A scenario consists of instructions provided by Player 1 to Player 2. Player 2 follows these instructions, which are then validated by a corresponding function. For the `scenario.py` function, please provide realistic examples of instructions that Player 1 might give, along with 5 paraphrases for each.

**Requirements:**

1. When specifying target objects (objects with which the player will interact), use different synonyms in the paraphrases to assess Player 2's vocabulary range.
2. Present the target objects in varying orders to evaluate how well Player 2 understands different language structures.
3. Sort the paraphrases for each instruction from simplest to most complex language.
4. Ensure the instructions are as varied as possible, utilizing a broad vocabulary.

**Format your answer as a Python dictionary with the following structure:**

```python
instructions = {
    'instruction_id': {
        'instruction': "Example instruction here",
        'instruction_paraphrases': [
            "Paraphrase 1 here",
            "Paraphrase 2 here",
            "Paraphrase 3 here",
            "Paraphrase 4 here",
            "Paraphrase 5 here"
        ],
        'check_lambda': lambda ...: scenario_function(...): ...  # Example usage of the function
    }
}
```

Replace `instruction_id` with a unique identifier for each instruction, and complete the `check_lambda` to demonstrate how you would verify the given instruction using the function.
