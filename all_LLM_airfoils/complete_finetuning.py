
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

import torch.nn.functional as F


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

        output_str = json.dumps(output, ensure_ascii=False)

        # 使用Chat格式，符合Qwen模型的对话习惯
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output_str}
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # 设置pad_token（Qwen可能需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )


    @torch.no_grad()
    def get_penultimate_embed(model, tokenizer, text: str):
        # 走你当前的 chat 模板，保持和训练一致
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)

        # 关键：在前向里显式要 hidden states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[-2]  # 倒数第二层，[B, T, H]

        # 用 attention mask 做 mean-pooling 得到句向量
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)

        # 归一化更稳（可选）
        return F.layer_norm(pooled, pooled.shape[-1:])  # [B, H]

    # 2. 设置LoRA配置
    print("🔧 配置LoRA微调...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        r=8,                           # 低秩矩阵维度
        lora_alpha=16,                 # 缩放参数
        lora_dropout=0.1,              # Dropout率
        # target_modules=[               # 目标模块（适配Qwen2.5）
        #     "q_proj", "v_proj", "k_proj", "o_proj",
        #     "gate_proj", "up_proj", "down_proj"
        # ]
        target_modules=[               # 目标模块（适配Qwen2.5）
            "q_proj", "v_proj", "k_proj", "o_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数数量



    # 3. 加载和处理训练数据
    print("📊 加载训练数据...")
    jsonl_file = "airfoil_train_mixed.jsonl"  # 你的JSONL文件

    try:
        raw_data = load_jsonl_data(jsonl_file)
        print(f"✅ 成功加载 {len(raw_data)} 条训练样本")
        #raw_data = raw_data[:100]

    except FileNotFoundError:
        print(f"❌ 找不到文件 {jsonl_file}")
        print("💡 请确保运行了 llm_data.py 生成训练数据")
        return

    # 格式化数据
    print("🔄 格式化训练数据...")
    formatted_texts = format_training_data(raw_data, tokenizer)
    train_dataset = create_dataset(formatted_texts, tokenizer)
    #train_dataset = train_dataset.remove_columns("labels")

    print(f"✅ 数据集创建完成，共 {len(train_dataset)} 个样本")




    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./qwen3b-airfoil-lora",        # 输出目录
        num_train_epochs=1,                        # 训练轮次
        per_device_train_batch_size=4,             # 批次大小（如果显存不够，改为2）
        gradient_accumulation_steps=4,             # 梯度累积（相当于batch_size=16）
        learning_rate=1e-4,                        # 学习率
        logging_steps=10,                          # 日志记录步数
        save_steps=100,                            # 保存模型步数
        save_total_limit=2,                        # 最多保存2个检查点
        fp16=True,                                 # 半精度训练
        dataloader_drop_last=True,                 # 丢弃不完整的最后一个批次
        report_to=None,                            # 不上报到wandb等平台
        remove_unused_columns=False,               # 保留未使用的列
    )

    # 5. 设置数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言模型
        #pad_to_multiple_of=8,
        return_tensors="pt"
    )


    # 6. 创建训练器
    print("🏋️ 创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7. 开始训练
    print("🎯 开始微调训练...")
    print("=" * 50)

    try:

        # trainer.train()
        # print("\n✅ 训练完成！")
        #
        # print("📉 Training logs:")
        # for log in trainer.state.log_history:
        #     if "loss" in log:
        #         print(log)
        #
        # # 保存最终模型
        # trainer.save_model("./qwen3b-airfoil-final")
        # tokenizer.save_pretrained("./qwen3b-airfoil-final")
        # print("💾 模型已保存到 ./qwen3b-airfoil-final")

        # 8. 简单测试
        print("\n🧪 快速测试:")
        test_prompt1 = '''
            指令: 设计目标：
            设计一个对称翼型。
            
            初始设计参数：
            cl = -0.0
            cd = 0.0045
            camber = 0.0
            thickness = 0.1
            
            任务说明：
            请确认或微调上述参数，使其适合用于后续条件扩散模型的几何生成。
            '''

        test_prompt2 = '''
            指令: 用于 Re=1e6、Mach=0.0、alpha=0 条件下的对称翼型，模仿 NACA0012。
            初始参数：cl=0.00, cd=0.00535, camber=0.007, thickness=0.12。
            翼型必须保持对称，其余参数可协调修正。
            '''

        SYSTEM_PROMPT = """
        你是一个翼型参数修正器（Airfoil Parameter Refiner）。
        你的任务是根据给定的设计目标和初始参数，
        输出一组【修正后的参数】。

        你必须严格且只能输出如下 JSON 格式：
        {
          "cl": float,
          "cd": float,
          "camber": float,
          "thickness": float
        }

        要求：
        - 不允许输出任何解释性文字
        - 不允许包含多余字段
        - 不允许输出 Markdown
        """

        test_prompts = [test_prompt1, test_prompt2]

        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
                print(f"输入: {prompt}")
                print(f"输出: {response}")
                print(type(response))
                print("-" * 30)

                # # 新增：拿倒数第二层句向量
                # penul = get_penultimate_embed(model, tokenizer, prompt)  # [1, H]
                # print(f"🔎 倒数第二层向量形状: {tuple(penul.shape)}")  # e.g. (1, 3072)
                # print("-" * 30)


    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        print("💡 可能的解决方案:")
        print("   - 减少 batch_size (改为2)")
        print("   - 增加 gradient_accumulation_steps")
        print("   - 检查显存使用情况")



if __name__ == "__main__":
    main()


