from baselines.experiments.super_igor.super_dataset import SuperDataset
from craftext.instructions.scenarios.manager import ScenariosNoLambda

class CrafTextScenariosWithSuperDataset(ScenariosNoLambda):
    def __init__(self, encode_model, config_name=None, super_dataset_name = "None", load_preinited=False, use_plans=False, update_sd = False):
        self.use_plans = use_plans
        self.update_sd = update_sd
        self.super_dataset_name = super_dataset_name
        self.load_preinited = load_preinited
        if  self.super_dataset_name != "None" and load_preinited:
            self.super_dataset = SuperDataset.load_from_json(super_dataset_name)
        else:
            self.super_dataset = SuperDataset()
        super().__init__(encode_model, config_name, use_plans)
    
    
    def pairwise_instructions_and_checkers(self):
        instructions_list, indices_list, checkers_data_dict = super().pairwise_instructions_and_checkers()
        print(" oooo "*5)
        print(checkers_data_dict)
        print(" oooo "*5)
        if not self.update_sd:
            return instructions_list, indices_list, checkers_data_dict
        
        # Pairvise SD plan and checkerts
        instructions_list_upd, indices_list_upd, checkers_data_dict_upd = [], [], dict()
        sd_instructions = self.super_dataset.instructions.keys()
        
        for i in range(len(instructions_list)):
            if instructions_list[i] in sd_instructions:
                instructions_list_upd.append(instructions_list[i])
                indices_list_upd.append(indices_list[i])
                for key in checkers_data_dict:
                    if key not in checkers_data_dict_upd:
                        checkers_data_dict_upd[key] = []
                    checkers_data_dict_upd[key].append(checkers_data_dict[key][i])
                    
                    print("oooo"*5)
                    print(checkers_data_dict_upd)
                    print("oooo"*5)
                    
        #instructions_list_upd - instead of list of str - list of list of str, each with the same size
        return instructions_list_upd, indices_list_upd, checkers_data_dict_upd 

    def encode_instructions(self, instructions):
        encoded_instructions, responses = self.encode_model.encode(instructions, return_responses=True)
        # Determine the number of variants per instruction
        num_variants = len(encoded_instructions) // len(instructions)
        print(len(responses), len(instructions))
        assert len(encoded_instructions) == len(instructions) * num_variants, \
            f"Unexpected size of encoded instructions(instructions len - {len(encoded_instructions)} and batch len - {len(instructions)} and num_vars {num_variants}). Ensure encode_model is consistent."
        if self.update_sd or not self.load_preinited:
        # Extend SuperDataset with generated plans
            for j, instruction in enumerate(instructions):
                for k in range(num_variants):
                    variant_index = j * num_variants + k
                    self.super_dataset.add_instruction(instruction=instruction, plans=[responses[variant_index]])
            if self.update_sd:
                name = self.super_dataset_name.replace(".json", "_upd.json")
            else:
                name = self.super_dataset_name
            self.super_dataset.save_to_json(filepath=name)

        return encoded_instructions, responses, num_variants
    
    def _prepare_scenarios(self):
        return super()._prepare_scenarios(add_original_instructions=True)
        
def create_scenarios_with_super_dataset(super_dataset_name=None, load_preinited=False, update_sd=False):
    class CustomCrafTextScenariosWithSuperDataset(CrafTextScenariosWithSuperDataset):
        def __init__(self, encode_model, config_name, load_preinited=load_preinited, update_sd=update_sd):
            super().__init__(encode_model, super_dataset_name=super_dataset_name,
                             config_name=config_name, load_preinited=load_preinited, update_sd=update_sd)
    return CustomCrafTextScenariosWithSuperDataset

