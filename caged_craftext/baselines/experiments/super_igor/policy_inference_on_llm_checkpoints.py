import subprocess
from datetime import datetime

import numpy as np
import wandb
import os
import glob
from baselines.experiments.super_igor.super_dataset import SuperDataset
from faker import Faker

def run_policy_train(craftext_settings, env_name, 
                     llm_path, super_dataset, num_envs,
                     start_checkpoint_path=None, experiment_name="experiment", 
                     encode_form_name="EMBEDDING", total_timesteps=250000000):
    args = [
        "python", "policy_train.py",
        "--craftext_settings", craftext_settings,
        "--env_name", env_name,
        "--llm_path", llm_path,
        "--super_dataset", super_dataset,
        "--num_envs", str(num_envs),
        "--experiment_name", experiment_name,
        "--encode_form_name", encode_form_name,
        "--start_checkpoint_path", start_checkpoint_path,
        "--total_timesteps", str(total_timesteps)
    ]

    try:
        process = subprocess.Popen(args)
        process.wait()  # Дождаться завершения процесса
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Attempting to terminate the process...")
        process.terminate()  # Завершить процесс
        process.wait()  # Подождать завершения
    except Exception as e:
        print(f"Error occurred: {e}")
    except subprocess.CalledProcessError as e:
        print(f"RL Training failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()
    finally:
        if process.poll() is None:
            print("Forcibly killing the process...")
            process.kill()
        print("Process terminated.")
    


def run_policy_inference(llm_name, dataset_name, save_dataset_name, experiment_name, plan_with_llm, 
                         craftext_settings,num_return_sequences='5', augment="False"):
    args = [
            "python", "policy_inference.py", 
            "--experiment_name", experiment_name,
            "--craftext_settings", craftext_settings,
            "--num_envs", "1024",  
            "--plan_with_llm", str(plan_with_llm),
            "--inference", "True",  
            "--llm_path", llm_name,
            "--augment", str(augment),
            "--dataset_path", dataset_name,
            "--save_dataset_path", save_dataset_name,
            "--num_return_sequences", num_return_sequences,
        ]

    print(args)
    #exit()
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()

def run_llm_train(dataset_name, llm_name, output_dir):
    args = ["python", "llm_train.py",
        "--dataset_name", dataset_name,
        "--llm_name", llm_name,
        "--output_dir", output_dir]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"LLM Training failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()



def log_validation(dataset_name, context=""):
    dataset = SuperDataset.load_from_json(dataset_name)
    top_50_percent, full =dataset.get_rewards(return_full=True)
    mean_rw = np.mean(top_50_percent)
    total_mean = np.mean(full)
    wandb.log({f"{context}_mean_top_50_sr":mean_rw})
    wandb.log({f"{context}_total_mean":total_mean})
    wandb.log({f"{context}_top_50_percent_sr": wandb.Histogram(top_50_percent)})

def merger_last_datasets(old_data_path, new_data_path):
    dataset = SuperDataset.load_from_json(old_data_path)
    dataset_new = SuperDataset.load_from_json(new_data_path)
    dataset.merge_and_optimize(dataset_new)
    dataset.save_to_json(new_data_path)

def get_rl_checkpoint_path(experiment_name):
    with open(f"{experiment_name}/path_to_last_checkpoint.txt", "r", encoding="utf-8") as file:
        content = file.read()
    return content

def get_rl_experiment_name(rl_experiment_path):
    p1 = rl_experiment_path.split("wandb/")[1]
    experiment_name = p1.split("/")[0]
    return experiment_name

if __name__=="__main__":
    wandb.init(project="super_igor_cycle_rest")
    os.makedirs("super_experiments", exist_ok=True)
    
    craftext_settings = "SI_simplified_set"
    
    experiment_name = "super_experiments/simple_achivements_one_test_llmt_True_Lawrence_Gibbs_20250311_082123"
    temp_path = f"{experiment_name}/temp_dataset"
    trained_llm_path = f"{experiment_name}/llm_checkpoints"
    
    start_checkpoint =  "./external_experiment_v3/3_" 
    all_checkpoints = [start_checkpoint]
    for path in glob.glob(trained_llm_path+"/*"):
        for ppath in glob.glob(path+"/*"):
            all_checkpoints.append(ppath)
    
    print(all_checkpoints)
    
    rl_experiment_path = get_rl_checkpoint_path(experiment_name)
    rl_experiment_name = get_rl_experiment_name(rl_experiment_path)
    for i, checkpoint in enumerate(all_checkpoints):
        save_dataset_path = f"{temp_path}/val_train_{i}.json"
        run_policy_inference(checkpoint, save_dataset_path, save_dataset_path,
                                        plan_with_llm=True,
                                        experiment_name=rl_experiment_name,
                                        craftext_settings=craftext_settings, 
                                        augment=0,
                                        num_return_sequences='1')
        log_validation(save_dataset_path, context="val_train_1")
        
    