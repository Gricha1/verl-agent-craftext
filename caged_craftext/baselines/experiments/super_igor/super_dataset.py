import json
import numpy as np
import pandas as pd
from datasets import Dataset
from baselines.experiments.super_igor.prompts import promt_instruction, PROMPTS
from collections import defaultdict
import time
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import os
import warnings

plan_config = {
    'max_plan_steps': 12,
    'max_step_words_count': 5,
    'end_command': "Finish!", 
    'remove_symbols': [',', ".", "-", ":", "_", ";", "Step"]
}
def check_plan(plan, plan_config):
    steps = plan.split("\n")
    if len(steps) > plan_config['max_plan_steps']:
          return False
    return True

def prepare_plan(plan, plan_config):
    return f"[{plan}]"


def generate_sd_from_subtasks(path, new_path):
    df = pd.read_csv(path)
    count_new_plans = [2, 5, 10]
    bad_step = "Deafault_plans"
    directory = os.path.dirname(path)
    optimized_sd_path = os.path.join(directory, new_path)

    sd_optimized = SuperDataset()
    instructions = list(set(df["instruction"].values))

    for i_ix, instruction in enumerate(instructions):
        mask = df["instruction"] == instruction

        # Get all plans for the instruction
        plans_for_instructions = df[mask]
        clear_plans = list(
            set(plans_for_instructions["plan"].tolist())
        )

        for i, plan in enumerate(clear_plans):
            mask = plans_for_instructions["plan"] == plan
            instruction = plans_for_instructions["instruction"][mask].values[0]

            
            mask_per_instruction = plans_for_instructions["instruction"] == instruction

            join_mask = mask & mask_per_instruction

            
            mask = join_mask
            subtasks_for_plan = plans_for_instructions[mask][
                ["subtask", "per_step_score"]
            ]
            count_subtaks_in_plan = len(plan.split("\n"))
            subtasks = subtasks_for_plan["subtask"].values[:count_subtaks_in_plan]
            scores = subtasks_for_plan["per_step_score"].values[:count_subtaks_in_plan]
            
            max_scores = defaultdict(float)
            for subtask, score in zip(subtasks, scores):
                max_scores[subtask] = max(max_scores[subtask], score)

            scores = np.array([max_scores[subtask] for subtask in subtasks])

           # min_score = scores.min()
         #   max_score = scores.max()
            norm_scores = scores / scores.sum()
          #  norm_scores = norm_scores

            old_plan_sr = plans_for_instructions[mask]["sr"].values[0]

            if old_plan_sr < 0.2:
                new_plans = count_new_plans[0]
            elif 0.2 <= old_plan_sr < 0.7:
                new_plans = count_new_plans[1]
            else:
                new_plans = count_new_plans[2]

            optimized_plans = []
            print("Subtasks: ", subtasks)
            print("Scores: ", norm_scores)
            print("- - - - - - - - - - - - - -  -")
            
            for j in range(new_plans):
                optimized_plan = None
  
                for _ in range(10):
                    
                    # Sample subtasks
                    random_values = np.random.rand(len(norm_scores))
                    random_mask = random_values < norm_scores
                    filtered_subtasks = subtasks[random_mask].tolist()
                    optimized_plan = "\n".join(filtered_subtasks)
                    
                    print("F-Subtasks: ", filtered_subtasks)
                    
                    if len(filtered_subtasks)==0:
                        continue
                    if optimized_plan not in optimized_plans:
                        optimized_plans.append(optimized_plan)
                        break


            if optimized_plans:
                sd_optimized.add_instruction(
                    instruction=instruction, plans=optimized_plans
                )
            else:
                sd_optimized.add_instruction(
                    instruction=instruction, plans=[plan]
                )
                

    sd_optimized.save_to_json(optimized_sd_path)
  
    

achievement_dict = {
    0: "COLLECT_WOOD",
    1: "PLACE_TABLE",
    2: "EAT_COW",
    3: "COLLECT_SAPLING",
    4: "COLLECT_DRINK",
    5: "MAKE_WOOD_PICKAXE",
    6: "MAKE_WOOD_SWORD",
    7: "PLACE_PLANT",
    8: "DEFEAT_ZOMBIE",
    9: "COLLECT_STONE",
    10: "PLACE_STONE",
    11: "EAT_PLANT",
    12: "DEFEAT_SKELETON",
    13: "MAKE_STONE_PICKAXE",
    14: "MAKE_STONE_SWORD",
    15: "WAKE_UP",
    16: "PLACE_FURNACE",
    17: "COLLECT_COAL",
    18: "COLLECT_IRON",
    19: "COLLECT_DIAMOND",
    20: "MAKE_IRON_PICKAXE",
    21: "MAKE_IRON_SWORD",
}

def achievments_vector_to_dict(vector, devision=1):
    acievments_dict_v= dict()
    devision = 1 if devision == 0 else devision.item()
    for i in range(len(vector)):
        acievments_dict_v[achievement_dict[i]] = vector[i].item()/devision
    
    cleaned_achievments = dict()
    for achievment in acievments_dict_v:
        if achievment == 'WAKE_UP': continue
        if acievments_dict_v[achievment] > 0.1:
            cleaned_achievments[achievment] = acievments_dict_v[achievment] 
    str_result = str(acievments_dict_v).replace(",", "\n")
    str_result_clean =  str(cleaned_achievments).replace(",", "\n")
    return str_result, str_result_clean
            
class Instruction:
    def __init__(self, instruction, plan_options, rewards=None):
        """
        Initialize an Instruction object.

        :param instruction: A string representing the instruction.
        :param plan_options: A list of strings, each representing a potential plan to achieve the instruction.
        :param rewards: A list of rewards corresponding to each plan option.
        """
        self.PLANS_STORE_SIZE = 50
        self.NOT_MEAURED = -1
        self.bad_list = ["Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans",
                         ]
        self.instruction = instruction
        self.plan_options = plan_options
        self.rewards = rewards if rewards is not None else [self.NOT_MEAURED for _ in range(len(plan_options))]
        if -1 in self.rewards:
            warnings.warn("For some reason, during the initialization of instructions and plans\
                          in init, some plans were not validated. \
                          They were added to the dataset with a value of -1.", UserWarning)

        self.mean_reward = 0
        self.update_mean_reward()
       # self.sort_plans_by_reward()
    
    def __str__(self):
        """Return a formatted string representation of the instruction."""
        result = [f"Instruction: {self.instruction}"]
        for i, (plan, reward) in enumerate(zip(self.plan_options, self.rewards)):
            reward_str = f"{reward}" if reward != self.NOT_MEAURED else "Not Measured"
            result.append(f"  ----- Plan_{i + 1}: {plan} | Reward: {reward_str}")
        return "\n".join(result)
    
    def clear_scores(self):
        self.rewards = [-1 for _ in range(len(self.rewards))]
    
    def return_best_plans(self):
        t = self.kde_treshold()
        plans = []
        for rewards, option in zip(self.rewards, self.plan_options):
            if rewards > t:
                plans.append((option, rewards))
        return plans
                
    def kde_treshold(self, return_peaks=False):
        rewards = np.array(self.rewards)
        
        good_idx = [idx for idx in range(len(self.plan_options)) if self.plan_options[idx] not in self.bad_list]
        rewards = rewards[good_idx]
        mask = ~np.isnan(rewards)
        rewards = rewards[mask]
        data_reshaped = np.array(rewards).reshape(-1, 1)
        # Построим KDE (ядровая оценка плотности)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(data_reshaped)
        x_vals = np.linspace(min(rewards), max(rewards), 500).reshape(-1, 1)
        log_density = kde.score_samples(x_vals)

        # Находим пики (локальные максимумы) и минимумы
        peaks, _ = find_peaks(log_density)
        minima, _ = find_peaks(-log_density)
        
        boundaries = x_vals[peaks].flatten()
        boundaries_v2 = x_vals[minima].flatten()
        if len(boundaries) == 0:
            boundaries = [1.0,1.1]
        if len(boundaries_v2) == 0:
            boundaries_v2 = [1.0,1.1]
        
        max_peaks = np.max(boundaries)
        max_minima = np.max(boundaries_v2)
        
        treshold = np.max([max_peaks,max_minima]) 
        if return_peaks:
            return treshold, peaks, minima

        return treshold
    
    def update_mean_reward(self):
        np_rewards = np.array(self.rewards)
        mean_reward = np.mean(np_rewards[np_rewards>=0])
        self.mean_reward = mean_reward
        return
    
    def sort_plans_by_reward(self):
        paired_options = list(zip(self.rewards, self.plan_options))
        paired_options.sort(key=lambda x: x[0], reverse=True)
        self.rewards, self.plan_options = zip(*paired_options)
        self.rewards = list(self.rewards)[:self.PLANS_STORE_SIZE]
        self.plan_options = list(self.plan_options)[:self.PLANS_STORE_SIZE]
        return

    def map_instruction_to_plan(self):
        """
        Returns a dictionary where the instruction is the key,
        and the value is a list of tuples containing the plan options and their corresponding rewards.

        :return: dict {instruction: [(plan_option_1, reward_1), (plan_option_2, reward_2), ...]}
        """
        return {self.instruction: list(zip(self.plan_options, self.rewards))}
    
    def map_plan_to_instruction(self):
        """
        Returns a dictionary where each plan option is a key,
        and the value is a tuple of the corresponding instruction and reward.

        :return: dict {plan_option_1: (instruction, reward_1), ...}
        """
        return {plan: (self.instruction, reward) for plan, reward in zip(self.plan_options, self.rewards)}


    def update(self, plan_options, rewards):
       # print(rewards)
        updated = False
        for i, plan_option in enumerate(plan_options):
            
           # Fetermine what reward to use
            if rewards is None or rewards[0] is None:
                 # if ther no reward and no options yet
                reward_to_add = self.NOT_MEAURED
                warnings.warn("For some reason, the added plan during the 'update()' \
                              operation was not validated. It was added to the dataset \
                              with a value of -1", UserWarning)
            else:
                reward_to_add = rewards[i]

            # CASE - 1: WE ALREADY HAVE PLAN 
            if plan_option in self.plan_options:
                # Look for all same plans
                indices = [i for i, option in enumerate(self.plan_options) if option == plan_option]

                 # CASE - 1.1: WE ALREADY HAVE PLAN BUT IT IS NOT MEASURED
                for i in indices:
                    if self.rewards[i]==self.NOT_MEAURED or reward_to_add>self.rewards[i]:
                        new_reward = reward_to_add
                        self.rewards[i] = new_reward 
                        updated = True
                        break
                    
                # IF ALL SAME PLANS ALREADY MESURED THIS INSTRUCTION WILL SKIPED
            
             # CASE - 2: WE HAVENT THIS PLAN YET
            else:
                self.plan_options.append(plan_option)
                self.rewards.append(reward_to_add)
                updated = True
                
        self.update_mean_reward()
        return updated

 
    
    def _update_reward(self, old_reward, new_reward):
        if old_reward==self.NOT_MEAURED:
            return new_reward
        
        print(self)
        print("SR pairs to updates: ", old_reward, new_reward)
        
        return new_reward #np.max([old_reward, new_reward])

    def to_dict(self):
        return {
            "instruction": self.instruction,
            "plan_options": self.plan_options,
            "rewards": self.rewards,
            "mean_reward": self.mean_reward
        }
    
    @staticmethod
    def from_dict(data):
        rewards = []
        for reward in (data["rewards"]):
            if np.isnan(reward):
                rewards.append(0.1)
            else:
                rewards.append(reward)
        instruction = Instruction(
            instruction=data["instruction"],
            plan_options=data["plan_options"],
            rewards=rewards
        )
        
        print(data)
        print("- - - - - ")
        #time.sleep(1)
        
        return instruction
        

class SuperDataset:
    def __init__(self):
        self.NOT_MEAURED = -1
        self.instructions = dict()  # List of Instruction objects
        self.PLANS_STORE_SIZE = 50
        self.bad_list = ["Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans",
                         ]
        self.mapping_plan_to_instruction = dict()
        
    def rebuild_mapping(self):
        instructions = self.instructions
        mapping_plan_to_instruction = {}
        for instruction in instructions:
            for plan in instructions[instruction].plan_options:
                if plan not in mapping_plan_to_instruction:           
                    mapping_plan_to_instruction[plan] = [instruction]
                else:
                    mapping_plan_to_instruction[plan].append(instruction)
        self.mapping_plan_to_instruction = mapping_plan_to_instruction
    
    def is_plan_correct_rule(self, plan):
        steps = plan.split("\n")
        if len(steps)<2:
            return 0
        for step in steps:
            if step in self.bad_list:
                return 0
        return 1

    def add_instruction(self, instruction, plans, rewards = None):
        if instruction in self.instructions:
            self.instructions[instruction].update(plans, rewards)
        else:
            self.instructions[instruction] = Instruction(instruction=instruction, 
                                                         plan_options=plans, 
                                                         rewards=[self.NOT_MEAURED for _ in range(len(plans))])
        for plan in plans:
            self.mapping_plan_to_instruction[plan] = instruction
    
    def clear_scores(self):
        for instruction in self.instructions:
             self.instructions[instruction].clear_scores()
             
    def update(self, plan, reward):
        if plan == "?\n\nFinish!":
            print(self.mapping_plan_to_instruction[plan])
        if plan not in list(self.mapping_plan_to_instruction.keys()):
            return
        instructions = self.mapping_plan_to_instruction[plan]
        for instruction in instructions:
            updated = self.instructions[instruction].update([plan], [reward])
            if plan == "?\n\nFinish!":
                print("- - - - - - - - - - - -")
                print()
                print(reward)
                print()
                print("- - - - - - - - - - - -")
            if updated:
                if plan == "?\n\nFinish!":
                    print("Update for :") 
                    print(instruction)
                break
   
    def super_print(self):
        for instruction in self.instructions:
            print( self.instructions[instruction])
        return 
    
    def batch_update(self, plans, rewards):
        for plan, reward in zip(plans, rewards):
            #print(reward)
            self.update(plan, reward)
    
    def batch_update_instruction(self,instructions, plans, rewards):
        for i, instruction in enumerate(instructions):
            print("instruction -- plan -- reward")
            print(instruction, [plans[i]], [rewards[i]])
            self.instructions[instruction].update([plans[i]], [rewards[i]])

    
    def per_subtask_table(self, instructions,
                                plans,
                                rewards,
                                matrix_count, 
                                matrix_reward, 
                                per_achivment_sum,
                                output_path):
        
        matrix = matrix_reward/matrix_count
        
        counts = []
        sum_reward = []
        instructions_ = []
        subtasks_ = []
        sr = []
        full_plan = []
        ps_rewards = []
        achievments_vector = [] 
        clean_achievments_vector = [] 
        for i in range(len(plans)):
            plan, reward = plans[i], rewards[i]
            instruction = instructions[i] #self.mapping_plan_to_instruction[plan][0] #This wrong, need to fix
            subtask_values = matrix[i]
            counts_values = matrix_count[i]
            reward_values = matrix_reward[i]
            per_plan_achievments = per_achivment_sum[i]
            
            subtasks = plan.split("\n")
            for j,subtask in enumerate(subtasks):
                instructions_.append(instruction)
                subtasks_.append(subtask)
                counts.append(counts_values[j])
                sum_reward.append(reward_values[j])
                ps_rewards.append(subtask_values[j])
                _achievments_vector, _cleaned_achievments_vector = achievments_vector_to_dict(per_plan_achievments[j],counts_values[j] )
                achievments_vector.append(_achievments_vector)
                clean_achievments_vector.append(_cleaned_achievments_vector)
                full_plan.append(plan)
                sr.append(reward)
        
        df = pd.DataFrame({"instruction": instructions_,
                           "plan": full_plan,
                           "subtask":subtasks_,
                           "run_count": counts,
                           "sum_reward":sum_reward,
                           "per_step_score": ps_rewards,
                           "clean_achievments": clean_achievments_vector,
                           "achievments":achievments_vector,
                           "sr": sr,
                           
                           })
        df.to_csv(f"{output_path}/per_step_evaluation.csv")
                

    def save_to_json(self, filepath):
        data = {
            "instructions": {key: instr.to_dict() for key, instr in self.instructions.items()},
            "mapping_plan_to_instruction": self.mapping_plan_to_instruction
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_from_json(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        dataset = SuperDataset()
        dataset.instructions = {key: Instruction.from_dict(instr) for key, instr in data["instructions"].items()}
       # print(dataset.instructions)
        #exit()
        dataset.mapping_plan_to_instruction = data["mapping_plan_to_instruction"]
        dataset.remove_unecessary_keys()
       # print(dataset.mapping_plan_to_instruction)
       # exit()
        return dataset
    
    def get_rewards(self, return_full=False):
        rewards = []
        for instruction in self.instructions.values():
            for i in range(len(instruction.rewards)):
                if instruction.rewards[i]!=self.NOT_MEAURED:
                    rewards.append(instruction.rewards[i])
        array = np.array(rewards)
        cleaned_array = array[~np.isnan(array)]
        sorted_array = np.sort(cleaned_array)  # Сортируем массив по возрастанию
        top_50_percent = sorted_array[int(len(sorted_array) / 2):]  # Берем 50% лучших значений
        if return_full:
            return top_50_percent, sorted_array
        return top_50_percent


    def _treshold(self):
        rewards = []
        for instruction in self.instructions.values():
            for i in range(len(instruction.rewards)):
                if instruction.rewards[i] > 0:  
                    rewards.append(instruction.rewards[i])
        
        array = np.array(rewards)
        cleaned_array = array[~np.isnan(array)]
        median = np.percentile(cleaned_array, 70) 
        return median

    @staticmethod
    def formatting(prompts, responses, eos_token):
        if eos_token:
            formatted_data = [f"{prompt}{response}\n {eos_token}" for prompt, response in zip(prompts, responses)]
        else:
            formatted_data = [f"{prompt}{response}\n" for prompt, response in zip(prompts, responses)]
        return formatted_data

    def to_dataset(self, fields=None):
        if fields is None:
            fields = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_")]

        data_dict = {field: getattr(self, field) for field in fields}

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict(data_dict)
        return dataset

    def merge_and_optimize(self, other_dataset):
        for instruction, instr_obj in other_dataset.instructions.items():
            for i, plan in enumerate(other_dataset.instructions[instruction].plan_options):
                self.instructions[instruction].update([plan], [other_dataset.instructions[instruction].rewards[i]])
                    
            # if instruction in self.instructions:
            #     current_instr = self.instructions[instruction]
            #     combined_plans = current_instr.plan_options + instr_obj.plan_options
            #     combined_rewards = current_instr.rewards + instr_obj.rewards
            # else:
            #     combined_plans = instr_obj.plan_options
            #     combined_rewards = instr_obj.rewards
            # if combined_rewards is not None:
            #     sorted_data = sorted(zip(combined_plans, combined_rewards), key=lambda x: x[1], reverse=True)
            #     top_5_plans, top_5_rewards = zip(*sorted_data[:self.PLANS_STORE_SIZE])
            # else:
            #     top_5_plans = combined_plans[:self.PLANS_STORE_SIZE]
            #     top_5_rewards = [None] * len(top_5_plans)
            # self.instructions[instruction] = Instruction(instruction=instruction, 
            #                                              plan_options=list(top_5_plans),
            #                                              rewards=list(top_5_rewards))
            #self.add_instruction(instruction, list(top_5_plans), list(top_5_rewards))

    def remove_unecessary_keys(self):
        current_keys = self.instructions.keys()
       # print(current_keys)
        plans = self.mapping_plan_to_instruction.keys()
       # print(plans)
        new_mapping = dict()
        for plan in plans:
            instruction_lst = self.mapping_plan_to_instruction[plan]
            #print(instruction)
            for instruction in instruction_lst:
                if instruction in list(current_keys):
                # print("YYY")
                    if plan not in new_mapping:
                        new_mapping[plan] = []
                    new_mapping[plan].append(instruction)
        self.mapping_plan_to_instruction = new_mapping
        
        
    def llm_dataset(self, plans_type, eos_token=None, use_kde=True,  return_poor_plans=False):
        prompts = []
        answers = []
        treshold =  self._treshold()
        
        bad_examples_prompts = []
        bad_example_answer = []
        for instruction in self.instructions.keys():
            instruction_obj = self.instructions[instruction]
            if use_kde:
                try:
                    treshold = instruction_obj.kde_treshold()
                except Exception as e:
                    print(e)
                    treshold = 2
            for i in range(len(instruction_obj.rewards)):
                reward = instruction_obj.rewards[i]
                if self.is_plan_correct_rule(instruction_obj.plan_options[i]):
                    if reward >= treshold and reward is not np.nan:
                        if check_plan(instruction_obj.plan_options[i], plan_config=plan_config):
                            prompts.append(promt_instruction(instruction, PROMPTS[plans_type]))
                            answers.append(prepare_plan(instruction_obj.plan_options[i], plan_config=plan_config))

                    else:
                        bad_examples_prompts.append(promt_instruction(instruction, PROMPTS[plans_type]))
                        bad_example_answer.append(instruction_obj.plan_options[i])
            
        text = SuperDataset.formatting(prompts, answers, eos_token)
        bad_text =  SuperDataset.formatting(bad_examples_prompts, bad_example_answer, eos_token)
        data_dict = {
        "text": text,
            }
        bad_data_dict = {
            "text":bad_text
            }
        full_dataset = Dataset.from_dict(data_dict)
        full_bad_dataset =  Dataset.from_dict(bad_data_dict)
        
        if return_poor_plans:
            return full_dataset, data_dict, bad_data_dict, answers
    
        return full_dataset, data_dict, bad_data_dict
        


        
        