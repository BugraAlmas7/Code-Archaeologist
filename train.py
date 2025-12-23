# train.py
import os
import sys
import subprocess
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset
import torch

# NOT: Hoca bu dosyayi calistirdiginda "DEEP" veya "DIVERSE" secebilsin diye
# burayi degisken yaptik. Varsayilan DEEP kalsin.
MOD_TIPI = "DEEP" 

print(f"🚀 EGITIM BASLIYOR: {MOD_TIPI} MODU")

# ==============================================================================
# 1. MODEL VE LORA AYARLARI
# ==============================================================================
max_seq_length = 1024 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==============================================================================
# 2. VERI SETI SECIMI
# ==============================================================================
if MOD_TIPI == "DEEP":
    dataset_id = "Naholav/CodeGen-Deep-5K"
elif MOD_TIPI == "DIVERSE":
    dataset_id = "Naholav/diverse-instruction"

dataset = load_dataset(dataset_id, split="train")

def formatting_prompts_func(examples):
    # Esnek Sütun Bulucu
    if "instruction" in examples: instructions = examples["instruction"]
    elif "question" in examples: instructions = examples["question"]
    elif "problem" in examples: instructions = examples["problem"]
    elif "content" in examples: instructions = examples["content"]
    else: instructions = examples[list(examples.keys())[0]]

    if "solution" in examples: solutions = examples["solution"]
    elif "output" in examples: solutions = examples["output"]
    elif "response" in examples: solutions = examples["response"]
    else: solutions = [""] * len(instructions)

    texts = []
    system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."
    
    for instruction, solution in zip(instructions, solutions):
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{solution}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# ==============================================================================
# 3. EGITIM PARAMETRELERI
# ==============================================================================
output_dir = f"./models/{MOD_TIPI.lower()}_instruction/checkpoints"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4, 
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,
        logging_steps = 20,            
        eval_strategy = "steps",       
        eval_steps = 20,               
        save_strategy = "steps",       
        save_steps = 100,              
        output_dir = output_dir,
        learning_rate = 2e-4,
        fp16 = False,  
        bf16 = True, 
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

trainer.train()
print("Egitim Tamamlandi.")