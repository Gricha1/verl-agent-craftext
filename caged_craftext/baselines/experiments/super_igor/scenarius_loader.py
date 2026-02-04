from tqdm import tqdm
import time
import numpy as np

from craftext.instruction.wrappers.craftext_wrapper import InstructionWrapper
from baselines.experiments.super_igor.encoder import QwenEncodeModel, EncodeForm
from baselines.experiments.super_igor.super_dataset import SuperDataset

from craftax.craftax_env import make_craftax_env_from_name
from craftext.craftext_scenarious import CrafTextScenarios, ScenarioData

class CrafTextScenariosWithSuperDataset(CrafTextScenarios):
    def __init__(self, encode_model, config_name=None, super_dataset_name = "None", use_plans=False):
        self.use_plans = use_plans
        self.super_dataset_name = super_dataset_name
        if  self.super_dataset_name != "None":
            self.super_dataset = SuperDataset.load_from_json(super_dataset_name)
        else:
            self.super_dataset = SuperDataset()
        super().__init__(encode_model, config_name, use_plans)
    
    def is_instruction_trainable(self, mean_instruction_value):
        lower_treshold = 0
        upper_treshold = 0.95
        return mean_instruction_value > lower_treshold and mean_instruction_value < upper_treshold
    
    def filter_instructions_by_reward(self, instructions_list):
        indices_for_train = []
        if self.super_dataset_name is not None and self.super_dataset_name!="None":
            for i, instruction in enumerate(instructions_list):
                super_instruction = self.super_dataset.instructions[instruction] 
                if self.is_instruction_trainable(super_instruction.mean_reward):
                    indices_for_train.append(i)
        else:
            return list(range(len(instructions_list)))
        return indices_for_train
   
    def _prepare_scenarios(self):
        """
        Prepares and encodes the scenarios, deciding whether to use embeddings or tokens,
        while handling multiple embeddings per instruction.
        """
        instructions_list, checkers_list, indices_list, scenario_names_list = self._load_original_scenarios()
        # Process instructions in batches
        before_filtering = len(instructions_list)
        indeces_for_filtering = self.filter_instructions_by_reward(instructions_list) 
        if self.super_dataset_name is not None:
            instructions_list = [instructions_list[i] for i in indeces_for_filtering]
            checkers_list = [checkers_list[i] for i in indeces_for_filtering]
            indices_list = [indices_list[i] for i in indeces_for_filtering]
            scenario_names_list = [scenario_names_list[i] for i in indeces_for_filtering]
        after_filtering = len(instructions_list)
        
        print("="*60)
        print()
        print(f"Saved {after_filtering} / {before_filtering} instructions ")
        print()
        print("="*60)

        instructions_list_f, checkers_list_f, indices_list_f, embeddings_list_f, scenario_names_f = [], [], [], [], []
        batch_size = 2
        for i in tqdm(range(0, len(instructions_list), batch_size)):
            batch_instructions = instructions_list[i:i+batch_size]
            batch_checkers = checkers_list[i:i+batch_size]
            batch_indices = indices_list[i:i+batch_size]
            batch_names = scenario_names_list[i:i+batch_size]

            # Encode instructions in the batch
            encoded_instructions, responses = self.encode_model.encode(batch_instructions, return_responses=True)

            # Determine the number of variants per instruction
            num_variants = len(encoded_instructions) // len(batch_instructions)
            assert len(encoded_instructions) == len(batch_instructions) * num_variants, \
                f"Unexpected size of encoded instructions(instructions len - {len(encoded_instructions)} and batch len - {len(batch_instructions)}). Ensure encode_model is consistent."

            # Process each instruction's embeddings and replicate corresponding metadata
            for j, instruction in enumerate(batch_instructions):
                for k in range(num_variants):
                    variant_index = j * num_variants + k
                    instructions_list_f.append(responses[variant_index])
                    checkers_list_f.append(batch_checkers[j])
                    indices_list_f.append(batch_indices[j])
                    embeddings_list_f.append(encoded_instructions[variant_index])
                    scenario_names_f.append(batch_names[j])

                    self.super_dataset.add_instruction(instruction=instruction, plans=[responses[variant_index]])

        #self.super_dataset.save_to_json(filepath='super_igor.json')

        scenario_data = ScenarioData(
            instructions_list=instructions_list_f,
            checkers_list=checkers_list_f,
            str_check_lambda_list = scenario_names_f,
            scenario_names = scenario_names_f,
            indices_list=np.array(indices_list_f).reshape(-1, 1),
            embeddings_list=np.array(embeddings_list_f).reshape(len(embeddings_list_f), -1) if embeddings_list_f else None
        )
        # Return the prepared ScenarioData
        return scenario_data


def create_scenarios_with_super_dataset(super_dataset_name=None):
    class CustomCrafTextScenariosWithSuperDataset(CrafTextScenariosWithSuperDataset):
        def __init__(self, encode_model, config_name):
            super().__init__(encode_model, super_dataset_name=super_dataset_name, config_name=config_name)
    return CustomCrafTextScenariosWithSuperDataset

if __name__ == "__main__":
    env = make_craftax_env_from_name(
        "Craftax-Classic-Pixels-v1", False
    )
    env_params = env.default_params
    env = InstructionWrapper(env, 'build', scenario_handler_class=CrafTextScenariosWithSuperDataset,
                                  encode_model_class=QwenEncodeModel,
                                  encode_form=EncodeForm.WEIGHTED_MEAN)