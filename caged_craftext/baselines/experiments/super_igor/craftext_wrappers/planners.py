from typing import  List
from dataclasses import dataclass
from itertools import combinations

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def sigmoid(x, center=0.8, sharpness=10):
    return 1 / (1 + np.exp(-sharpness * (x - center)))

def normalize_to_one(arr):
    arr = np.array(arr, dtype=np.float64)
    arr /= arr.sum()

    n = 10**6
    counts = np.round(arr * n).astype(int)

    diff = counts.sum() - n

    if diff != 0:
        # Если diff > 0, уменьшаем элемент, если < 0, увеличиваем элемент
        if diff > 0:
            # Ищем элемент > 0, чтобы безопасно уменьшить
            idx = np.where(counts > 0)[0][0]
            counts[idx] -= diff
        else:
            # Увеличиваем любой элемент
            idx = 0
            counts[idx] -= diff  # diff < 0, вычитание отрицательного = прибавление

    probs = counts / n
    return probs


def softmax_with_power(rewards, power=2):
    rewards = np.array(rewards)
    mask_nonzero = rewards > 0

    weights = np.zeros_like(rewards, dtype=np.float32)
    weights[mask_nonzero] = rewards[mask_nonzero] ** power
    adjusted = sigmoid(weights)
    
    adjusted = np.round(adjusted, 2)
    
    if np.sum(adjusted) == 0:
        probs = [1/len(adjusted)]*len(adjusted)
    else:
    # Then normalize to sum to 1
      
       probs = normalize_to_one(adjusted)# / np.sum(adjusted)
       probs[probs<0] = 0
      # probs[-1] += 1.0 - np.sum(probs)
       probs = probs.tolist()

    print("- - - ")
    print(adjusted)
    print(probs)
    print(np.sum(probs))
    return probs

from baselines.experiments.super_igor.prompts import (
    promt_instruction,
    PlanExtractor,
    PROMPTS
)

DEFAULT_PLAN = "Deafault_plans Deafault_plans Deafault_plans Deafault_plans Deafault_plans"
DEFAULT_STEP = "[MASK]"
    

def plans_batch_fill(plans, samples_needed, sr=None, full_sampled=False):
    """
    Extend the list of plans by sampling from it (with replacement),
    using a distribution based on sr (if provided) or uniform otherwise.
    """
    
    print(plans)
    if full_sampled:
        samples_needed = samples_needed + len(plans)
    num_plans = len(plans)
    if num_plans == 0:
        num_plans = 1
    if sr is not None:
        sr = np.asarray(sr)
        if len(sr) != num_plans:
            raise ValueError("Length of sr must match length of plans")
        sr[sr==-1] = 0
        sum_v = sr.sum() 
       
        if sum_v == 0:
            probabilities = [1/len(sr)]*len(sr)  # Uniform if all rewards is zero
        else:
            probabilities = softmax_with_power(sr) # / sum_v
    else:
        probabilities = np.full(num_plans, 1 / num_plans)

    print("- - - PROBS - - -")
    print(probabilities)
    sampled_indices = np.random.choice(num_plans, size=samples_needed, p=probabilities)
    sampled_plans = [plans[i] for i in sampled_indices]
    
    print("Full sampled ?", full_sampled, "Count sampled",  len(sampled_plans), "Count plans was", len(plans))
    print(probabilities)
    
    
   # exit()
    if full_sampled:
        return sampled_plans
    return plans + sampled_plans

    
    
def plan_augmentations(plan, use_self_fill=False, sr=None):
    steps = plan.split("\n")
    combinations_result = []
    constraint = 3

    for i in range(2, 6):
        combo = combinations(steps, i)
        combo_strs = ["\n".join(c) for c in combo]
        combinations_result.extend(combo_strs)

    random.shuffle(combinations_result)

    constrained_combinations = combinations_result[:constraint] + [plan]

    samples_needed = (constraint + 1) - len(constrained_combinations)
    if samples_needed > 0:
        if use_self_fill:
            constrained_combinations = plans_batch_fill(constrained_combinations, samples_needed, sr=sr)
        else:
            constrained_combinations += [DEFAULT_PLAN] * samples_needed

    return constrained_combinations

def format_responces(responses):
        responses_new =  []
        for r in responses:
            formeted_r = ""
            if "Plan:" in r:
                formeted_r = r.split("Plan:")[2]
            elif "1" in r:
                formeted_r = r.split("1")[2]
            else:
                formeted_r = r
            if "Finish!" in r:
                formeted_r = formeted_r.split("Finish!")[0] + "Finish!"
            responses_new.append(formeted_r)
        return responses_new

@dataclass 
class PromptTemplate:
    base = 0
    functions = 1
    fucntions_hints = 2
    
@dataclass
class ModelConfig():
    original_model_path: str
    peft_weights_path: str
    
@dataclass 
class GenerationConfig():
    num_paraphrases: int
    beam_groups: int
    beams_count: int
    max_new_tokens: int
    prompt_template: int
    

class SDPlanner:
    def __init__(self, super_dataset, num_return_sequences: int = 5, augment: bool = False, full_sampled: bool = True):
        self.super_dataset = super_dataset
        self.num_return_sequences = num_return_sequences
        self.augment = augment
        self.default_plan = DEFAULT_PLAN
        self.use_self_fill = True
        self.full_sampled = full_sampled
    
    def return_plans(self, instructions: List[str]) -> List[str]:
        plans = self.extract_plans(instructions, self.use_self_fill)
        print("Did Plan")
        if self.augment:
            augmented = []
            for plan in plans:
                augmented.extend(plan_augmentations(plan, use_self_fill=self.use_self_fill))
            plans = augmented
            print("Augment Plan")
        return plans

    def extract_plans(self, instructions: List[str], 
                      use_self_fill: bool = False) -> List[str]:
        collected = []
    
        for instruction in instructions:
            if instruction in self.super_dataset.instructions:
                plans = self.super_dataset.instructions[instruction].plan_options
                sr = self.super_dataset.instructions[instruction].rewards
            else:
                print(instruction)
                raise Exception(f"{instruction} not in SD!")

            num_missing = self.num_return_sequences - len(plans)

            if num_missing > 0:
                print(instruction)
                print(f"Plans count: {len(plans)}, but should be {self.num_return_sequences}")
                if use_self_fill:
                    plans = plans_batch_fill(plans, num_missing, sr=sr, full_sampled=self.full_sampled)
                else:
                    plans += [self.default_plan] * num_missing
                
                print("Update plans count: ", len(plans))
                print("= = ="*30)

            plans = plans[:self.num_return_sequences]  # In case someone added too many
            collected.extend(plans)
        print("!!!--!!!"*5)
        print(len(collected))
        print("!!!--!!!"*5)
        return collected

import gc
class LLMPlanner:
    """
    Generate plans with LLM models
    """
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        self.generation_config = generation_config
        self.plan_model, self.plan_tokenizer = self._init_model_and_tokenizer(model_config)
        self.device = self.plan_model.device
        self.use_self_fill = True

    def _init_model_and_tokenizer(self, model_config: ModelConfig):

        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.original_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.original_model_path,
            trust_remote_code=True
        )
        tokenizer.add_eos_token = True

        if (model_config.peft_weights_path != model_config.original_model_path 
            and model_config.peft_weights_path is not None):
        
            try:
                base_model = PeftModel.from_pretrained(base_model, model_config.peft_weights_path).eval()
            except Exception as e:
                raise(f"Failed to load LoRA weights: {e}")

        return base_model, tokenizer
    
    def format_responses(self, raw_responses, sr=None):
        prompt_type = self.generation_config.prompt_template
        formatted = []
        k = self.generation_config.num_paraphrases

        assert len(raw_responses) % k == 0, f"raw_responses size {len(raw_responses)} not divisible by num_paraphrases={k}"

        # Split into chunks of size k (one chunk per instruction)
        for i in range(0, len(raw_responses), k):
            chunk = raw_responses[i:i + k]
            extracted_plans = []

            for response in chunk:
                try:
                    plan = PlanExtractor.extract(response, prompt_type)
                except:
                    plan = DEFAULT_PLAN
                extracted_plans.append(plan)

            # Identify missing (DEFAULT_PLAN) entries
            valid_plans = [p for p in extracted_plans if p != DEFAULT_PLAN]
            num_missing = k - len(valid_plans)

            if num_missing > 0:
                if self.use_self_fill:
                    filled = plans_batch_fill(valid_plans, num_missing, sr=sr)
                    extracted_plans = valid_plans + filled[len(valid_plans):]
                else:
                    extracted_plans = valid_plans + [DEFAULT_PLAN] * num_missing

            extracted_plans = extracted_plans[:k]  # trim in case of oversampling
            formatted.extend(extracted_plans)
        return formatted

    def return_plans(self, instructions: list[str]):
        promt_template = PROMPTS[self.generation_config.prompt_template]
        prompts = [f"{promt_instruction(instr, prompt=promt_template)}" for instr in instructions]

        inputs = self.plan_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        if self.generation_config.num_paraphrases != 1:
            num_beams = self.generation_config.beams_count
            num_beam_groups = self.generation_config.beam_groups
        else:
            num_beams = num_beam_groups = 1

        generation_params = {
            "max_new_tokens": self.generation_config.max_new_tokens,
            "eos_token_id": self.plan_tokenizer.eos_token_id,
            "num_beams": num_beams,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": 0.1 if self.generation_config.num_paraphrases > 1 else 0.0,
            "do_sample": False,
            "early_stopping": True,
            "num_return_sequences": self.generation_config.num_paraphrases,
            "return_dict_in_generate": False
        }

        with torch.no_grad():
            outputs = self.plan_model.generate(**inputs, **generation_params)
        raw_responses = self.plan_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        formated = self.format_responses(raw_responses)
        
        del inputs, outputs, raw_responses
        torch.cuda.empty_cache() 
        gc.collect() 

        return formated 


def make_planer(planer_type: str, config: dict, full_sampled: bool):
    
    if planer_type == "llm":
        model_cfg = ModelConfig(**config["model_config"])
        generation_cfg = GenerationConfig(**config["generation_config"])
        return LLMPlanner(model_config=model_cfg, generation_config=generation_cfg)

    elif planer_type == "sd":
        return SDPlanner(
            super_dataset=config["super_dataset"],
            num_return_sequences=config["generation_config"]["num_paraphrases"],
            augment=config.get("augment", False),
            full_sampled=full_sampled
        )

    else:
        raise ValueError(f"Unknown planner type: {planer_type}")
