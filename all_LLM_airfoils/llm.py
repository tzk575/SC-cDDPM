

import torch
import json

# 针对3B模型的优化配置
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset




# 加载千问3B
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3B模型的LoRA配置（微调）
lora_config = LoraConfig(
    r=8,              # 3B模型用更小的rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)

# 训练参数（针对3B优化）
training_args = TrainingArguments(
    output_dir="./qwen3b-airfoil",
    num_train_epochs=5,           # 3B可以训练更多轮次
    per_device_train_batch_size=8, # 3B可以用更大batch size
    gradient_accumulation_steps=2,
    learning_rate=3e-4,           # 3B可以用稍高学习率
    logging_steps=10,
    save_steps=200,
    fp16=True,
    dataloader_drop_last=True,
)


