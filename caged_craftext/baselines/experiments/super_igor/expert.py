
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
# def correctness_plan_prompt(plan):
#     return  f"Does the given text represent a structured step-by-step plan?  \n {plan} \n A valid plan consists of clearly numbered several steps describing specific sequential actions and must starts from \"1.\".Answer 'Yes' or 'No'. Your answer: "

template = """Does the given TEXT provide a sequence?
                TEXT:
                ----------------
                {}
                ----------------
Do not select 'Yes' if TEXT do not starts from word "Here".
Answer 'Yes' or 'No'. Your answer: {}"""


few_shot_examples = [
    ("Here's a revised plan: ?\\n\\nFinish!", "No"),
    ("Here's a revised plan following the provided format:1. Gather resources 2. Build a foundation 3. Construct walls 4.Finish!", "No"),
    ("1. Find logs 2. Make a table 3.Finish!", "Yes"),
    ("?\\nFinish!", "No"),
    ("1. Gather resources 2. Build a foundation 3. Construct walls 4.Finish!", "Yes")
]


few_shot = "\n\n".join(template.format(text, answer) for text, answer in few_shot_examples)

def correctness_plan_prompt(plan):
    return few_shot + "\n\n" + template.format(plan, "")

class PlansExpert():
    def __init__(self):
        self.plan_model  = AutoModelForCausalLM.from_pretrained(
                                    "Qwen/Qwen2.5-3B-Instruct",
                                ).cuda()
        self.plan_tokenizer = AutoTokenizer.from_pretrained( "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        self.plan_tokenizer.padding_side = "right"
        self.plan_model.eval()

    
    def check_plan_correctness(self, plans):
        prompts = [correctness_plan_prompt(plan) for plan in plans]
        inputs = self.plan_tokenizer(prompts, truncation=True, return_tensors="pt", padding="longest").to(self.plan_model.device)

        with torch.no_grad():
            outputs = self.plan_model(**inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            
        input_length = inputs["input_ids"].shape[1]
        first_token_logits = logits[:, input_length-1, :]  # (batch_size, vocab_size)
        yes_token_id = self.plan_tokenizer.convert_tokens_to_ids("Yes")
        no_token_id = self.plan_tokenizer.convert_tokens_to_ids("No")

        yes_probs = first_token_logits[:, yes_token_id].cpu().numpy()  # (batch_size,)
        no_probs = first_token_logits[:, no_token_id].cpu().numpy()  # (batch_size,)

        return [[round(float(yes), 3), round(float(no), 3)] for yes, no in zip(yes_probs, no_probs)]
    
    