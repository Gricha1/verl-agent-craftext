import sys
import os
import argparse
import numpy as np
import wandb
import yaml
import json

from transformers import AutoTokenizer 
from datasets import Dataset

from baselines.experiments.super_igor.craftext_wrappers.old_encode_code import QwenEncodeModel
from baselines.experiments.super_igor.craftext_wrappers.encoder_trainer import train
from baselines.experiments.super_igor.super_dataset import SuperDataset
from baselines.paths import set_os_environ
from baselines.experiments.super_igor.prompts import promt_instruction, PlanExtractor

def prepare_external_dataset(dataset_path, eos_token=None):
        prompts = []
        answers = []
        with open(dataset_path, "r") as f:
            data = json.load(f)
        for instruction in data.keys():
            plans_options = data[instruction]
            for i in range(len(plans_options)):
                    prompts.append(promt_instruction(instruction))
                    answers.append(plans_options[i]+"\n Finish!")

        text = SuperDataset.formatting(prompts, answers, eos_token)
    
        data_dict = {
        "text": text,
            }
       
        full_dataset = Dataset.from_dict(data_dict) 
        return full_dataset, data_dict, data_dict
    
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_latest_folder(path='./results'):
    folders = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    if not folders:
        print("В папке нет подкаталогов.")
        return None
    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder


def train_llm(dataset, 
              plans_type,
              base_model_name="Qwen/Qwen2.5-3B-Instruct",
              previos_checkpoint_dir="Qwen/Qwen2.5-3B-Instruct", 
              ouput_dir='./results', use_external=False):
    
    llm_train_config = load_config('llama_training.yaml')
    llm_train_config['training_args']['output_dir'] = ouput_dir
   
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                              trust_remote_code=True)
    eos_token = tokenizer.eos_token
    if use_external: 
        llm_dataset, data_dict, bad_data_dict = dataset
    else:
        llm_dataset, data_dict, bad_data_dict = dataset.llm_dataset(eos_token=eos_token,plans_type=plans_type)
    
    train(dataset=llm_dataset, data_for_inference=data_dict, plans_type=plans_type,
          train_config=llm_train_config,
          plan_model_name=previos_checkpoint_dir)
    
    return get_latest_folder(ouput_dir)


def main(dataset_name, base_model_name, external_dataset, llm_name, plans_type, output_dir):
    use_external = False    

    if external_dataset is None:
        dataset = SuperDataset.load_from_json(dataset_name)
    else:
        dataset = prepare_external_dataset(external_dataset)
        use_external = True
    new_llm_path = train_llm(dataset=dataset, 
                            base_model_name=base_model_name,
                            previos_checkpoint_dir=llm_name, 
                            ouput_dir=output_dir, use_external=use_external,
                            plans_type=plans_type)
    return new_llm_path

if __name__=="__main__":
    set_os_environ()
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--external_dataset", default=None, type=str)
    parser.add_argument("--llm_name", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-3B-Instruct", type=str)
    parser.add_argument("--plans_type", default=None, type=int)
    args, rest_args = parser.parse_known_args(sys.argv[1:])
    main(args.dataset_name, args.base_model_name,
         args.external_dataset, args.llm_name, 
         args.plans_type, args.output_dir)
