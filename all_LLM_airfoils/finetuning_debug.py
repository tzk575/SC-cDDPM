
import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType



os.environ["WANDB_DISABLED"] = "true"

def load_jsonl_data(file_path):
    """加载JSONL训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_training_data(data, tokenizer):
    """格式化训练数据为模型可用格式"""

    def format_prompt(instruction, output):
        # 使用Chat格式，符合Qwen模型的对话习惯
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        # 使用tokenizer的chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        return formatted

    formatted_texts = []
    for item in data:
        formatted_text = format_prompt(item['instruction'], item['output'])
        formatted_texts.append(formatted_text)

    return formatted_texts


def create_dataset(formatted_texts, tokenizer, max_length=512):
    """创建训练数据集 - 修复版本"""

    print("🔄 使用批处理方式创建数据集...")
    dataset = Dataset.from_dict({'text': formatted_texts})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # 批次内统一填充
            max_length=max_length,
        )
        #tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    # 批处理tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,  # 关键：批处理
        remove_columns=['text']
    )

    print(f"✅ 数据集创建完成，共 {len(tokenized_dataset)} 个样本")

    # 验证数据类型
    print("🔍 验证数据类型...")
    sample = tokenized_dataset[0]


    print(f"input_ids内容: {sample['input_ids']}")
    print(f"是否嵌套: {type(sample['input_ids'][0])}")

    return tokenized_dataset


def main():
    print("🚀 开始翼型LLM微调训练...")

    # 1. 加载模型和tokenizer
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"📥 加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 设置pad_token（Qwen可能需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. 设置LoRA配置
    print("🔧 配置LoRA微调...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        r=8,                           # 低秩矩阵维度
        lora_alpha=16,                 # 缩放参数
        lora_dropout=0.1,              # Dropout率
        target_modules=[               # 目标模块（适配Qwen2.5）
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数量



    # 3. 加载和处理训练数据
    print("📊 加载训练数据...")
    jsonl_file = "airfoil_llm_training.jsonl"  # 你的JSONL文件

    raw_data = load_jsonl_data(jsonl_file)
    print(f"✅ 成功加载 {len(raw_data)} 条训练样本")
    print(type(raw_data))
    print(raw_data[0])
    print(raw_data[0].keys())




if __name__ == "__main__":
    main()


