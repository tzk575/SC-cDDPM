# 基础库
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import warnings
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings('ignore')


class AirfoilLLMTester:
    def __init__(self, model_path="./qwen3b-airfoil-final"):
        """初始化微调后的LLM模型"""
        print("🔄 正在加载模型...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # 加载微调后的模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        print("✅ 模型加载完成!")

    def apply_physics_constraints(self, llm_output, user_input):
        """应用物理约束修正LLM输出"""
        try:
            # 解析LLM输出
            params = json.loads(llm_output)

            # 对称翼型约束
            if "对称" in user_input:
                params["camber"] = 0.0
                params["cl"] = max(0.0, min(params["cl"], 0.3))  # 对称翼型升力系数限制

            # 物理合理性检查
            if params["cd"] < 0.003:  # 阻力系数下限
                params["cd"] = 0.003

            return json.dumps(params)  # 返回修正后的JSON字符串
        except:
            # 如果解析失败，返回原始输出
            return llm_output

    @torch.no_grad()
    def get_penultimate_embed(self, text: str):
        # 保持和训练时一致的 chat 模板
        messages = [{"role": "user", "content": text}]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)

        # 显式拿 hidden states
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states[-2]  # 倒数第二层，[B, T, H]

        # attention mask mean-pooling → 句向量
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)

        # 归一化（可选）
        return F.layer_norm(pooled, pooled.shape[-1:])  # [B, H]

    def generate_airfoil_params(self, structured_prompt: str):
        """
        使用 Refiner 协议生成翼型参数
        返回：dict + penultimate embedding
        """

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
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": structured_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # -------- JSON 安全解析 --------
        try:
            params = json.loads(response)
        except Exception:
            raise ValueError(f"模型输出无法解析为 JSON: {response}")

        # -------- embedding（保留你原来的做法） --------
        penul = self.get_penultimate_embed(structured_prompt)

        return params, penul

    def test_multiple_cases(self):
        """测试多个案例"""
        test_cases = [
            "设计一个高升力低阻力的翼型",
            "无人机长航时翼型设计",
            "设计一个对称翼型",
            "民航客机巡航翼型设计",
            "战斗机机动性翼型",
            "风力发电叶片高效翼型"
        ]

        print("\n🧪 开始批量测试...")
        for i, case in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}/{len(test_cases)} ---")
            print(f"输入: {case}")

            try:
                result = self.generate_airfoil_params(case)
                print(f"输出: {result}")

                # 尝试解析JSON
                try:
                    json_result = json.loads(result)
                    print(f"✅ JSON格式正确: {json_result}")
                except:
                    print(f"❌ JSON格式错误")

            except Exception as e:
                print(f"❌ 生成失败: {e}")

            print("-" * 60)


def jitter_params(
    params: dict,
    batch: int = 16,
    enable_jitter: bool = True,
    jitter_ratio: dict = None,
    fixed_params: list = None,
):
    """
    根据 LLM 给出的参数，生成 batch 份 numpy 条件向量
    """

    if jitter_ratio is None:
        jitter_ratio = {
            "cl": 0.05,
            "cd": 0.03,
            "thickness": 0.08,
            "camber": 0.0,      # camber 默认不抖
        }

    if fixed_params is None:
        fixed_params = ["camber"]

    # 初始化数组
    cl = np.full(batch, params["cl"], dtype=np.float32)
    cd = np.full(batch, params["cd"], dtype=np.float32)
    camber = np.full(batch, params["camber"], dtype=np.float32)
    thickness = np.full(batch, params["thickness"], dtype=np.float32)

    if enable_jitter:
        for name, arr in zip(
            ["cl", "cd", "camber", "thickness"],
            [cl, cd, camber, thickness],
        ):
            if name in fixed_params:
                continue

            ratio = jitter_ratio.get(name, 0.0)
            noise = np.random.uniform(
                low=-ratio,
                high=ratio,
                size=batch
            )
            arr *= (1.0 + noise)

    # 组合 condition
    condition = np.stack(
        [cl, cd, camber, thickness],
        axis=1
    )

    return cl, cd, camber, thickness, condition




def main():
    print("\n🧪 Refiner 模式测试启动")

    # ========= 1. 初始化 Tester（保持你原来的方式） =========
    tester = AirfoilLLMTester()
    # ↑ 这一行不改，仍然由你原来的类负责加载模型和 tokenizer

    # ========= 2. 构造标准 Refiner Prompt =========
    test_prompt1 = '''
    指令: 设计目标：
    设计一个对称翼型。
    
    初始设计参数：
    cl = -0.0
    cd = 0.0045
    thickness = 0.1
    
    任务说明：
    请确认或微调上述参数，使其适合用于后续条件扩散模型的几何生成。
'''

    test_prompt2 = '''
指令: 用于 Re=1e6、Mach=0.0、alpha=0 条件下的对称翼型，模仿 NACA0012,设计一个对称翼型。
初始参数：cl=0.01, cd=0.00601, camber=0.002, thickness=0.12。
翼型必须保持对称，其余参数尽可能模仿NACA0012翼型。
'''

    test_prompts = [test_prompt1, test_prompt2]

    # ========= 3. 逐条调用 Refiner =========
    for i, prompt in enumerate(test_prompts):
        print("\n" + "=" * 60)
        print(f"📥 Case {i+1} | 输入 Prompt:")
        print(prompt.strip())

        try:
            params, embedding = tester.generate_airfoil_params(prompt)

            print("📤 Refined Parameters (dict):")
            print(params)
            print(type(params))
            print(params.keys())
            #print("🔎 Embedding shape:", embedding.shape)
            airfoil_nums = 16
            cl, cd, camber_list, thickness_list, condition = jitter_params(params, airfoil_nums, enable_jitter=True)
            print('相关信息： ', cl, cd, camber_list, thickness_list)
            print(type(cl))

        except Exception as e:
            print("❌ Refiner 调用失败")
            print(e)

    print("\n🎉 Refiner 主函数测试完成")



# 使用示例
if __name__ == "__main__":

    main()



