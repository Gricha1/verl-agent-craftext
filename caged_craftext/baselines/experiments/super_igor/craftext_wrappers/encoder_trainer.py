from transformers import (AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          TrainingArguments,
                          AutoTokenizer, 
                          pipeline)
from peft import LoraConfig
from peft import PeftModel
import torch
from trl import SFTTrainer



class EncoderTrainer():
    def __init__(self, config, base_model_name="Qwen/Qwen2.5-3B-Instruct"):

        self.config = config 
        self.base_model_name=base_model_name
        self.training_config = self.config['training_args']
        self.lora_config = self.config['q_lora']
        self.bnb_config = self.configure_bnb_config(**self.config['bitsandbytes'])
    
    def configure_bnb_config(self, use_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, use_nested_quant):
        """
        Configure BitsAndBytes for model quantization.
        """
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

    def init_trainer(self, dataset, model_name):
        """
        Initialize the SFT Trainer with the given configuration.
        """
        base_model_name = self.base_model_name
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=self.bnb_config,
            device_map=self.config['device_map']
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            r= self.lora_config['lora_r'],
            bias= "none",
            task_type="CAUSAL_LM",
        )

        if not "Qwen" in model_name:
            model.enable_input_require_grads()
            model = PeftModel.from_pretrained(model, model_name, is_trainable=True).to("cuda")

        training_arguments = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=int(self.training_config['num_train_epochs']),
            per_device_train_batch_size=int(self.training_config['per_device_train_batch_size']),
            per_device_eval_batch_size=int(self.training_config['per_device_eval_batch_size']),
            gradient_accumulation_steps=int(self.training_config['gradient_accumulation_steps']),
            gradient_checkpointing=bool(self.training_config['gradient_checkpointing']),
            optim=str(self.training_config['optim']),
            save_steps=int(self.training_config['save_steps']),
            logging_steps=int(self.training_config['logging_steps']),
            learning_rate=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config['weight_decay']),
            fp16=bool(self.training_config['fp16']),
            bf16=bool(self.training_config['bf16']),
            max_grad_norm=float(self.training_config['max_grad_norm']),
            max_steps=int(self.training_config['max_steps']),
            warmup_ratio=float(self.training_config['warmup_ratio']),
            group_by_length=bool(self.training_config['group_by_length']),
            lr_scheduler_type=str(self.training_config['lr_scheduler_type']),
            report_to="wandb"
        )


        return SFTTrainer(
            model=model,
            train_dataset=dataset,#['train'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None, #TODO: How to load None from configs?
            tokenizer=tokenizer,
            args=training_arguments,
            packing=self.config['sft']['packing'],
        )
    
import random
import wandb

def generate(prompt, model, tokenizer):
    with torch.cuda.amp.autocast():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=250)
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result


from baselines.experiments.super_igor.prompts import PlanExtractor

def generate_examples(dataset, model, tokenizer, plans_type):
    plans_extractor = PlanExtractor()
    dataset_size = len(dataset['text'])
    examples_inices = [random.randint(1, dataset_size) for i in range(5)]
    promts_with_answers = [dataset['text'][i] for i in examples_inices]
    
    prompts = [plans_extractor.extract_prompt(pa,plans_type) for pa in promts_with_answers]
    previos_plans = [plans_extractor.extract(pa,plans_type) for pa in promts_with_answers]
    
    plans = []
    model.eval()
    for prompt in prompts:
        plan = generate(prompt, model, tokenizer)
        plan = plans_extractor.extract(plan,plans_type)
        plans.append(plan)
        
    predictions = {'prompts':prompts, 
                'previos_plans': previos_plans, 
                'current_plans': plans}
    return predictions

def extract_promt_and_answers(dataset, plans_type):
    plans_extractor = PlanExtractor()
    texts = dataset['text']
    prompts = [plans_extractor.extract_prompt(pa,plans_type) for pa in texts]
    previos_plans = [plans_extractor.extract(pa,plans_type) for pa in texts]
    return {"prompts": prompts, 'previos_plans':previos_plans}

import pandas as pd
import os

def log_train_dataset(dataset, output_dir,plans_type):
    table_train = wandb.Table(columns=["Prompt", "Previous Plans"])
    train_data_dict = extract_promt_and_answers(dataset, plans_type)

    csv_data = {
        "Prompt": [],
        "Previous Plans": []
    }

    for prompt, prev_plan in zip(
        train_data_dict["prompts"],
        train_data_dict["previos_plans"]
    ):
        table_train.add_data(prompt, prev_plan)
        csv_data["Prompt"].append(prompt)
        csv_data["Previous Plans"].append(prev_plan)

    wandb.log({"TrainSet": table_train})
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "train_dataset_table.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8')
    
def train(
    dataset,
    data_for_inference,
    train_config,
    plan_model_name,
    plans_type,
    runs=1,
    full_train_set=True
):
    trainer = (
        EncoderTrainer(config=train_config)
        .init_trainer(dataset, plan_model_name)
    )
    
    if full_train_set:
        output_dir = train_config['training_args']['output_dir']
        log_train_dataset(dataset, output_dir, plans_type)

    for i in range(runs):
        # Generate examples from inference data
        examples_bad = generate_examples(
            data_for_inference, trainer.model, trainer.tokenizer,plans_type
        )
        table_bad = wandb.Table(
            columns=["Prompt", "Previous Plans", "Current Plans"]
        )
        for prompt, prev_plan, curr_plan in zip(
            examples_bad["prompts"],
            examples_bad["previos_plans"],
            examples_bad["current_plans"]
        ):
            table_bad.add_data(prompt, prev_plan, curr_plan)

        # Generate examples from training data
        examples = generate_examples(dataset, trainer.model, trainer.tokenizer,plans_type)
        table = wandb.Table(
            columns=["Prompt", "Previous Plans", "Current Plans"]
        )
        for prompt, prev_plan, curr_plan in zip(
            examples["prompts"],
            examples["previos_plans"],
            examples["current_plans"]
        ):
            table.add_data(prompt, prev_plan, curr_plan)

        # Train model and log examples
        trainer.train()
        wandb.log({"Generated Examples Good": table})
        wandb.log({"Generated Examples Previously Bad": table_bad})

        # Save the model after each run
        trainer.model.save_pretrained(
            f"{train_config['training_args']['output_dir']}/{i}_"
        )
