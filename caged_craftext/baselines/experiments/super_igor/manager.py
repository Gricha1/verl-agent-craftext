import os 
os.environ["CRAFTEXT_SCENARIO_PATH"] = "../../../craftext/scenarios/"

import subprocess
from datetime import datetime

import numpy as np
import wandb
import os
from baselines.experiments.super_igor.super_dataset import SuperDataset,generate_sd_from_subtasks   
from faker import Faker

def run_policy_train(craftext_settings, env_name, 
                     llm_path, super_dataset, num_envs,
                     start_checkpoint_path=None, experiment_name="experiment", 
                     encode_form_name="EMBEDDING", total_timesteps=250000000,
                     additional_args=None):
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
    if additional_args:
        for key, value in additional_args.items():
            if value is not None:
                args.extend([key] + value.split() if isinstance(value, str) and " " in value else [key, str(value)])
        
    try:
        process = subprocess.Popen(args)
        process.wait()  
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Attempting to terminate the process...")
        process.terminate() 
        process.wait() 
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
                         craftext_settings,num_return_sequences='5', augment="False", additional_args=None, per_step_scoring=1):
    args = [
            "python", "policy_inference.py", 
            "--experiment_name", experiment_name,
            "--craftext_settings", craftext_settings,
            "--num_envs", "1024",  
            "--plan_with_llm", str(plan_with_llm),
            "--inference", "1",  
            "--llm_path", llm_name,
            "--augment", str(augment),
            "--dataset_path", dataset_name,
            "--save_dataset_path", save_dataset_name,
            "--num_return_sequences", num_return_sequences,
            "--per_step_scoring", str(per_step_scoring),
        ]
    if additional_args:
        for key, value in additional_args.items():
            if value is not None:
                args.extend([key] + value.split() if isinstance(value, str) and " " in value else [key, str(value)])

    print(args)
    #exit()
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with return code {e.returncode}")
        print(f"Error message: {e}")
        exit()

def run_llm_train(dataset_name, base_model_name, llm_name,plans_type, output_dir):
    args = ["python", "llm_train.py",
        "--base_model_name",base_model_name,
        "--dataset_name", dataset_name,
        "--llm_name", llm_name,
        "--plans_type", str(plans_type),
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

import yaml
def planer_config_to_args(config_path, start_from_checkpoint):
    
    if start_from_checkpoint:
        planer_type = "sd"
    else:
        planer_type = "llm"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    planer_config = config["planer_config"]

    model_config = planer_config.get("model_config", {})
    generation_config = planer_config.get("generation_config", {})

    print(model_config)
    params = {
        "--planer_type": planer_type,
        "--embedding_source": 0,
        "--step_by_step": True,
        "--original_model_path": model_config.get("original_model_path", ""),
        "--peft_weights_path": model_config.get("peft_weights_path"),
        "--num_paraphrases": generation_config.get("num_paraphrases", 15),
        "--beam_groups": generation_config.get("beam_groups", 15),
        "--beams_count": generation_config.get("beams_count", 15),
        "--max_new_tokens": generation_config.get("max_new_tokens", 128),
        "--prompt_template": generation_config.get("prompt_template", 1),
        "--super_dataset": planer_config.get("super_dataset"),
    }
    return params

def ac_config_to_args(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    ac_config = config["network"]
    args = {}
    for key, value in ac_config.items():
        flag = f"--ac_{key}"
        if isinstance(value, list):
            args[flag] = " ".join(map(str, value))
        elif value is not None:
            args[flag] = value
        else:
            continue 
    return args

    
if __name__=="__main__":
    # - checkpoint
    start_from_checkpoint= True 
    restart_checkpoint_name = "./super_experiments/SI_simplified_set_one_llmt_True_Heather_Campbell_20250423_112253"
    rl_skip = 1 if start_from_checkpoint else 0
    inference_skip = 1 if start_from_checkpoint else 0
    llm_skip = 0 if start_from_checkpoint else 0
    
    # ------- Experiment args
    craftext_settings = "SI_simplified_set_one"
    start_planer_config = planer_config_to_args("./configs/qwen_3b_function.yaml", start_from_checkpoint)
    start_ac_config = ac_config_to_args("./configs/policy_base.yaml")
    llm_name =  "./pretrained_plan_llm/3_" #"Qwen/Qwen2.5-3B-Instruct" 
    
    use_llm_tuning = True
    validate = False    
    
    # ----- Experiment 
    wandb.init(project="super_igor_cycle_rest")
    os.makedirs("super_experiments", exist_ok=True)
    
    rl_done = 0
    inference_done = 0
    llm_done = 0
    # Init experiment
    if start_from_checkpoint:
        experiment_name = restart_checkpoint_name
        temp_path = f"{experiment_name}/temp_dataset"
        rl_experiment_path = "None"
        dataset_name = f"{temp_path}/super_dataset{0}_{0}.json"
    else:
        
        # Come up with name for experiment
        fake = Faker()
        name = fake.first_name()
        surname = fake.last_name()
        
        # Make experiment paths
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_tag = craftext_settings
        experiment_name = f"super_experiments/{experiment_tag}_llmt_{str(use_llm_tuning)}_{name}_{surname}_{current_time}"
        temp_path = f"{experiment_name}/temp_dataset"
        rl_experiment_path = "None"
        super_dataset_name = "None"
        os.makedirs(temp_path, exist_ok=True)
    
    # Get arguments for planing and pociey from config 
    additional_args =  {**start_planer_config, **start_ac_config}
    base_model_name = start_planer_config["--original_model_path"]
    plans_type = start_planer_config["--prompt_template"]
    
    for j in range(5):
        
        if rl_skip<= rl_done:
            dataset_name = f"{temp_path}/super_dataset{j}_{0}.json"
            run_policy_train(
                craftext_settings=craftext_settings,
                env_name="Craftax-Classic-Pixels-v1-Text",
                llm_path=llm_name,
                super_dataset=dataset_name,
                num_envs=1024,
                start_checkpoint_path=rl_experiment_path,
                experiment_name=experiment_name,
                encode_form_name="EMBED_CLS_FOR_SPLITS",
                total_timesteps=25000000,
                additional_args=additional_args
            )
        else:
            print("SKIP RL TRAINING!")
       # exit()
        rl_experiment_path = get_rl_checkpoint_path(experiment_name)
        rl_experiment_name = get_rl_experiment_name(rl_experiment_path)
        rl_done += 1
 
        for i in range(0,4):
            dataset_name = f"{temp_path}/super_dataset{j}_{i}.json"   
            persubtasks_res = f"{temp_path}/per_step_evaluation.csv"   
            output_dir = f'./{experiment_name}/llm_checkpoints/mix_text_cycle_{j}_{i}'
            plan_with_llm =  i>0 #Generate new plans after LLM training
            llm_checkpoint = llm_name
            if inference_skip <= inference_done:
                augment = 0 if i<1 else 0
                save_dataset_path = dataset_name
                
                # Validation on train with new LLM and SuperDataset generation
                
                run_policy_inference(llm_checkpoint,
                                    dataset_name, 
                                    save_dataset_path,
                                    experiment_name=rl_experiment_name,
                                    plan_with_llm=plan_with_llm,
                                    craftext_settings=craftext_settings,
                                    augment=augment,
                                    num_return_sequences='10',
                                    per_step_scoring=1,
                                    additional_args=additional_args)
                log_validation(dataset_name, context="train_dataset")
                
                save_dataset_path = f"{temp_path}/train_{j}_{i}.json"
                run_policy_inference(llm_checkpoint, save_dataset_path, save_dataset_path,
                                        plan_with_llm=True,
                                        experiment_name=rl_experiment_name,
                                        craftext_settings=craftext_settings, 
                                        augment=0,
                                        num_return_sequences='1',
                                        per_step_scoring=0,
                                        additional_args=additional_args)
                log_validation(save_dataset_path, context="train_1")
            else:
                print("SKIP RL INFERECNE!")
        
            inference_done += 1
            
           # generate_sd_from_subtasks(path=persubtasks_res, new_path=f"optim_super_dataset{j}_{i}.json") 
            dataset_name = f"{temp_path}/optim_super_dataset{j}_{i}.json"  
            # save_dataset_path = f"{temp_path}/optim_super_dataset{j}_{i}.json"    
            # run_policy_inference(llm_checkpoint,
            #                         dataset_name, 
            #                         save_dataset_path,
            #                         experiment_name=rl_experiment_name,
            #                         plan_with_llm=False,
            #                         craftext_settings=craftext_settings,
            #                         augment=0,
            #                         num_return_sequences='20',
            #                         per_step_scoring=0,
            #                         additional_args=additional_args)
            # log_validation(dataset_name, context="optim_train_dataset")

            # generated_data =f"{temp_path}/super_dataset{j}_{i}.json"
            # merger_last_datasets(generated_data, dataset_name) 
            
            run_policy_train(
                craftext_settings=craftext_settings,
                env_name="Craftax-Classic-Pixels-v1-Text",
                llm_path=llm_name,
                super_dataset=dataset_name,
                num_envs=1024,
                start_checkpoint_path=rl_experiment_path,
                experiment_name=experiment_name,
                encode_form_name="EMBED_CLS_FOR_SPLITS",
                total_timesteps=25000000,
                additional_args=additional_args
            )
            exit()
           
            
            if i>0:
                # Merge datasets
                old_data_path = f"{temp_path}/super_dataset{j}_{i-1}.json"
                merger_last_datasets(old_data_path, dataset_name) 
                #
                # Validation with RL on parafrases and new goals
                if validate:
                    test_parafeases_results_path = f"{temp_path}/test_parafeases{j}_{i}.json"
                    run_policy_inference(llm_name, test_parafeases_results_path,
                                        experiment_name=rl_experiment_name,
                                        plan_with_llm=plan_with_llm,
                                        craftext_settings=craftext_settings+"_test_parafrases", 
                                        num_return_sequences='1')
                    log_validation(test_parafeases_results_path, context="test_parafeases")

                    test_new_obj_results_path = f"{temp_path}/test_new_obj{j}_{i}.json"
                    run_policy_inference(llm_name, test_new_obj_results_path, 
                                        experiment_name=rl_experiment_name,
                                        plan_with_llm=plan_with_llm,
                                        craftext_settings=craftext_settings+"_test_other_params", 
                                        num_return_sequences='1')
                    log_validation(test_new_obj_results_path, context="test_new_obj")

                # Train LLM
            if use_llm_tuning:
                if llm_skip <= llm_done:
                    llm_checkpoint = llm_name
                    run_llm_train(dataset_name, base_model_name, llm_checkpoint, plans_type, output_dir)
                else:
                    print("SKIP LLM TRAIN!")
                llm_name = output_dir+"/0_"
                llm_done += 1
        super_dataset_name = dataset_name
