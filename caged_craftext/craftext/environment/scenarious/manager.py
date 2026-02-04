import json
import numpy as np
import jax
import jax.numpy as jnp
import logging
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from craftext.environment.scenarious.loader import ScenariosConfigLoader, load_scenarios
from craftext.environment.craftext_constants import plans_path
from craftext.environment.scenarious.checkers.target_state import TargetState

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🔹 Define an Enum for scenario field processing types
class ScenarioFieldType(Enum):
    SINGLE_VALUE = "single_value"  # The base instruction (not copied)
    PARAPHRASE_LIST = "paraphrase_list"  # A list of paraphrases (added to the base instruction)
    REPEAT_WITH_PARAPHRASES = "repeat_with_paraphrases"  # Repeated for each instruction and its paraphrases

# 🔹 Define the schema for scenario processing
SCENARIO_SCHEMA = {
    "instruction": ScenarioFieldType.SINGLE_VALUE,  
    "instruction_paraphrases": ScenarioFieldType.PARAPHRASE_LIST,  
    "scenario_checker": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
    "arguments": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
    "str_check_lambda": ScenarioFieldType.REPEAT_WITH_PARAPHRASES  
}

@dataclass
class ScenarioData:
    instructions_list: list
    scenario_checker: int
    arguments: TargetState
    str_check_lambda_list: list
    indices_list: list
    scenario_names: list
    embeddings_list: list

@dataclass
class ScenarioDataO:
    instructions_list: list
    scenario_checker: list
    arguments: list
    str_check_lambda_list: list
    indices_list: list
    scenario_names: list
    embeddings_list: list
    original_instructions: list

@dataclass
class ScenarioDataJAX:
    embeddings_list: jax.Array
    scenario_checker: int
    arguments: List[TargetState]

class ScenariosNoLambda:
    def __init__(self, encode_model, config_name=None, use_plans=False):
        """
        Initializes the CrafTextScenarios with an EncodeModel and scenario configuration.
        """
        self.encode_model = encode_model
        self.config = ScenariosConfigLoader().load_config(config_name)
        self.use_paraphrases = self.config.use_parafrases
        self.environment_key = 0 if "Classic" in self.config.base_environment else 1
        self.n_instructions = 0
        self.use_plans = use_plans
        self.instruction_to_update_file = plans_path
        self.all_scenario = self._load_scenarios(self.config)
        self.scenario_data = self._prepare_scenarios()
        self.scenario_data_jax = self.scenarios_to_jax()
        # print(f'scenario_data_jax: {self.scenario_data_jax}')

    @property
    def initial_instruction(self):
        """Generates the default encoded instruction for initializing network parameters."""
        return self.encode_model.encode(["None"])[:1]
    

    def castom_initial_instruction(self, instruction):
        """Generates the default encoded instruction for initializing network parameters."""
        return self.encode_model.encode([instruction])[:1]

    def _load_scenarios(self, config):
        """Loads scenarios from a specified configuration file."""
        return load_scenarios(config)

    def get_scenarios(self):
        """Retrieves the processed scenario data."""
        return self.scenario_data

    def encode(self, instruction):
        """Encodes an instruction using the provided encoding model."""
        return Tuple(self.encode_model.encode(instruction))

    
    def _prepare_scenarios(self, add_original_instructions=False):
        """
        Prepares and encodes scenarios while considering paraphrases.
        """
        instructions_list, indices_list, checkers_data_dict = self.pairwise_instructions_and_checkers()

        checkers_data_f = {key: [] for key in checkers_data_dict.keys()}
        batch_size = 2
        print(f"Initial number of instructions: {len(instructions_list)}")
        logger.info(f"Initial number of instructions: {len(instructions_list)}")

        instructions_f, indices_f, embeddings_f, o_instruction_f = [], [], [], []

        for i in tqdm(range(0, len(instructions_list), batch_size)):
            batch_instructions = instructions_list[i:i + batch_size]
            batch_indices = indices_list[i:i + batch_size]

            batch_results = self._pairwise_with_embeddings(batch_instructions, batch_indices, checkers_data_dict, i)
            instructions_f.extend(batch_results["instructions"])
            indices_f.extend(batch_results["indices"])
            embeddings_f.extend(batch_results["embeddings"])
            o_instruction_f.extend(batch_results["o_instructions"])

            for key in checkers_data_f.keys():
                checkers_data_f[key].extend(batch_results["checkers_data"][key])
                
        # with open("instructions_new_obj.json", 'w', encoding='utf-8') as f:
        #          json.dump(instructions_f, f, ensure_ascii=False, indent=4)
        # exit()
        if add_original_instructions:
            self.scenario_data = ScenarioDataO(
            instructions_list=instructions_f,
            scenario_checker=checkers_data_f["scenario_checker"],
            arguments=checkers_data_f["arguments"],
            str_check_lambda_list=checkers_data_f["str_check_lambda"],
            scenario_names=[str(i) for i in indices_f],
            indices_list=np.array(indices_f).reshape(-1, 1),
            embeddings_list=np.array(embeddings_f) if embeddings_f else None,
            original_instructions=o_instruction_f
            )
        else:
            self.scenario_data = ScenarioData(
            instructions_list=instructions_f,
            scenario_checker=checkers_data_f["scenario_checker"],
            arguments=checkers_data_f["arguments"],
            str_check_lambda_list=checkers_data_f["str_check_lambda"],
            scenario_names=[str(i) for i in indices_f],
            indices_list=np.array(indices_f).reshape(-1, 1),
            embeddings_list=np.array(embeddings_f) if embeddings_f else None
        )
        # self.scenario_data = ScenarioData(
        #     instructions_list=instructions_f,
        #     scenario_checker=checkers_data_f["scenario_checker"],
        #     arguments=checkers_data_f["arguments"],
        #     str_check_lambda_list=checkers_data_f["str_check_lambda"],
        #     scenario_names=[str(i) for i in indices_f],
        #     indices_list=np.array(indices_f).reshape(-1, 1),
        #     embeddings_list=np.array(embeddings_f) if embeddings_f else None
        # )
        return self.scenario_data


    def encode_instructions(self, instructions):
        encoded_instructions = self.encode_model.encode(instructions)
        # There is possible, than self.encode_model retunrn different version of instructions-plans and related embeddings
        num_variants = len(encoded_instructions) // len(instructions)
        assert len(encoded_instructions) == len(instructions) * num_variants, \
            f"Unexpected size of encoded instructions ({len(encoded_instructions)} vs {len(instructions)}). Ensure encode_model is consistent."

        return encoded_instructions, instructions, num_variants
        
    def _pairwise_with_embeddings(self, batch_instructions, batch_indices, checkers_data_dict, base_idx):
        """
        Encodes a batch of instructions and processes extracted data.
        """
        old_instructions = batch_instructions
        encoded_instructions, batch_instructions, num_variants = self.encode_instructions(batch_instructions)
        # print("encode instr", encoded_instructions)
        batch_results = {
            "instructions": [],
            "indices": [],
            "embeddings": [],
            "o_instructions": [],
            "checkers_data": {key: [] for key in checkers_data_dict.keys()}
        }

        for j, instruction in enumerate(old_instructions):
            for k in range(num_variants):
                variant_index = j * num_variants + k
                batch_results["indices"].append(batch_indices[j])
                batch_results["o_instructions"].append(instruction)
                batch_results["instructions"].append(batch_instructions[variant_index])
                batch_results["embeddings"].append(encoded_instructions[variant_index])

                for field in checkers_data_dict.keys():
                    batch_results["checkers_data"][field].append(checkers_data_dict[field][base_idx + j])

        return batch_results

    
    def pairwise_instructions_and_checkers(self):
        """
        Loads and processes scenarios based on the SCENARIO_SCHEMA.
        Return 3 lists - instructions_list, checkers_data_dict, indices_list
        instructions_list - all instruction, including parafrased vesions
        checkers_data_dict - all variable connected with checker to each instruction, len(checkers_data_dict[key]) == len(instructions_list)
        indices_list - indices of instrictions
        
        """
        instructions_list, indices_list = [], []
        checkers_data_dict = {key: [] for key in SCENARIO_SCHEMA.keys() if key != "instruction_paraphrases" and key != "instruction"}
        # print(f'all_scenario: {self.all_scenario}')
        # Run throw all goal dicts
        for idx, (key, scenario) in tqdm(enumerate(self.all_scenario.items())):
            instructions, indices, checkers_data = self._pairwise_goal_parafrases_and_checkers(scenario, idx)

            instructions_list.extend(instructions)
            indices_list.extend(indices)

            for field in checkers_data_dict.keys():
                checkers_data_dict[field].extend(checkers_data[field])
                
        # Change instructions to plans if necessary
        if self.use_plans:
            instructions_list = self._load_action_plans(instructions_list)
        return instructions_list, indices_list, checkers_data_dict
    

    def _pairwise_goal_parafrases_and_checkers(self, scenario, scenario_id):
        """
        Processes a single scenario based on the SCENARIO_SCHEMA.
        """
        instructions = [scenario.get("instruction", "Unknown instruction")]
        if self.use_paraphrases:
            instructions += scenario.get("instruction_paraphrases", [])

        indices = [scenario_id] * len(instructions)
        checkers_data = {key: [] for key in SCENARIO_SCHEMA.keys() if key != "instruction_paraphrases" and key != "instruction"}

        for key, field_type in SCENARIO_SCHEMA.items():
            if field_type == ScenarioFieldType.REPEAT_WITH_PARAPHRASES:
                checkers_data[key] = [scenario.get(key, None)] * len(instructions)
        return instructions, indices, checkers_data

    def _load_action_plans(self, instructions_list):
        """
        Loads action plans from a predefined file and updates instructions if applicable.
        """
        with open(self.instruction_to_update_file, 'r', encoding='utf-8') as f:
            action_plans = json.load(f)
        # print("Action_plans: \n",action_plans)
        # print("_______--")
        updated_instructions = [action_plans.get(instr, "none") for instr in instructions_list]
        logger.info("Using preloaded plans in craftext_scenarios.py")
        logger.info("Encoding instructions...")        
        return updated_instructions


    def scenarios_to_jax(self):
        """
        Converts scenario data to JAX-compatible structures.
        """
        embeddings_jax = jnp.array(self.scenario_data.embeddings_list) if self.scenario_data.embeddings_list is not None else None
        scenario_checker_jax = self._prepare_jax_checkers(self.scenario_data.scenario_checker)
        # print("scen fata",self.scenario_data)
        print(f"Final number of instructions: {len(self.scenario_data.embeddings_list)}")
        logger.info(f"Final number of instructions: {len(self.scenario_data.embeddings_list)}")

        return ScenarioDataJAX(
            embeddings_list=embeddings_jax,
            scenario_checker=scenario_checker_jax,
            arguments=self.scenario_data.arguments
            
        )

    def _prepare_jax_checkers(self, checkers_list):
        """
        Prepares the scenario checkers list for JAX.
        """
        # print(checkers_list)
        return jnp.array(checkers_list) if checkers_list else None

def create_scenarios_with_dataset(use_plans_gpt):
    class CustomCrafTextScenariosWithPlans(ScenariosNoLambda):
        def __init__(self, encode_model, config_name):
            super().__init__(encode_model, config_name=config_name, use_plans=use_plans_gpt)
    return CustomCrafTextScenariosWithPlans