import pickle

from dataclasses import dataclass

#from craftext.craftext_encoder import EncodeForm, DistilBertEncode

from craftext.encoders.craftext_distilbert_model_encoder import DistilBertEncode
from craftext.encoders.craftext_base_model_encoder import EncodeForm

from baselines.experiments.super_igor.craftext_wrappers.encoder_trainer import EncoderTrainer
from baselines.experiments.super_igor.craftext_wrappers.planners import SDPlanner, LLMPlanner, DEFAULT_STEP
from baselines.experiments.super_igor.craftext_wrappers.onehot_encoder import encode_plans
from baselines.experiments.super_igor.craftext_wrappers.planners import make_planer


import random

# --------------- Utils --------------- 
def split_plans(plan, max_steps_length=35):
    #print(plan)
    steps = plan.split("\n")
   
    if len(steps)<max_steps_length:
        steps += [DEFAULT_STEP]*(max_steps_length -  len(steps))
    else:
        steps = steps[:max_steps_length]
    return  steps+[DEFAULT_STEP]
# --------------- Encoder - Load plans from sd --------------- 

from enum import Enum
import numpy as np 
@dataclass
class EmbeddingSource(Enum):
    bert=0
    onehot=1

def encode_step_by_step(plans, encode_f, batch_encode=False):
    plans_per_instruction = [split_plans(plan) for plan in plans]
    print("Finish spliting")
    embeddings_list = []
    if not batch_encode:
        for plan in plans_per_instruction:
            embedings = []
            for step in plan:
                emb = encode_f([step])[0]
                embedings.append(emb)
            embeddings_list.append(embedings)
    else:
        print("Encoding...")
        all_steps = [step for plan in plans_per_instruction for step in plan]
        all_embeddings = encode_f(all_steps)  # assume returns list of embeddings

        # Reconstruct original structure
        idx = 0
        for plan in plans_per_instruction:
            step_count = len(plan)
            embeddings_list.append(all_embeddings[idx:idx + step_count])
            idx += step_count
        print("Finish encoding...")
    return embeddings_list

class EncoderWithPlanning(DistilBertEncode):
    def __init__(self, planer, encoder=EmbeddingSource.bert,
                 form_to_use=EncodeForm.EMBEDDING, step_by_step=True):
        super().__init__(form_to_use=form_to_use, n_splits=1)
        self.planer = planer
        self.embedding_source = encoder
        # Mode for giving plan step-by-step
        self.step_by_step = step_by_step
    
    def default_step_embedding(self):
        if self.embedding_source == EmbeddingSource.onehot.value:
            encode_f = encode_plans
        else:
            encode_f = super().encode
        return encode_f(DEFAULT_STEP)
        
    def encode(self, instruction, return_responses=False):
        plans = self.planer.return_plans(instruction)

        if self.embedding_source == EmbeddingSource.onehot.value:
            encode_f = encode_plans
            batch_encode = False
        else:
            encode_f = super().encode
            batch_encode = True

        print("Start split to step...")

        if self.step_by_step:
            embeddings_list = encode_step_by_step(plans, encode_f, batch_encode)
        else: 
            embeddings_list = encode_f(plans)

        print("Finish spliting...")

        if return_responses:
            return [embeddings_list, plans]
        return embeddings_list



def make_encoder_with_planning(
    planer_type: str,
    planer_config: dict,
    full_sampled: bool = True,
    embedding_source: EmbeddingSource = EmbeddingSource.bert,
    step_by_step: bool = True,
) -> type[EncoderWithPlanning]:
    """
    Factory that returns a subclass of EncoderWithPlanning
    preconfigured with planner and encoder setup.

    Returns:
        A class inheriting from EncoderWithPlanning.
    """
    # print(planer_config)
    # exit()
    planer = make_planer(planer_type, planer_config, full_sampled)

    class CustomEncoderWithPlanning(EncoderWithPlanning):
        def __init__(self,form_to_use,
                     planer=planer,
                     encoder=embedding_source,
                     step_by_step=step_by_step ):
            super().__init__(
                planer=planer,
                encoder=encoder,
                step_by_step=step_by_step,
                form_to_use=form_to_use
            )

    return CustomEncoderWithPlanning


if __name__ == "__main__":
    encoder = make_encoder_with_planning(
        planer_type="llm",
        planer_config={
            "model_config": {
                "original_model_path": "Qwen/Qwen2.5-3B-Instruct",
                "peft_weights_path": None
            },
            "generation_config": {
                "num_paraphrases": 5,
                "augment": True,
                "beam_groups": 2,
                "beams_count": 4,
                "max_new_tokens": 128,
                "prompt_template": 1
            },
            "super_dataset": None,
            "augment": False
        },
        embedding_source=EmbeddingSource.bert,
        step_by_step=True
    )
