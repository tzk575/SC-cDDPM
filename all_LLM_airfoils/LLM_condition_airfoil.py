# 完整的自然语言到翼型生成Pipeline - 包导入版本
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

# 🎯 直接从包导入！
from LLM.airfoil_LLM import AirfoilLLMTester
from all_comprehensive_test.utils_1d_2channel import *
from all_comprehensive_test.LucidDiffusion import *
from all_no_fight_condition.data_deal import AirfoilDataset
from all_no_fight_condition.GaussianDiffusion import *
from all_no_fight_condition.utils import *



from utils.npy_to_dat import process_and_save_airfoil



# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


class NLPToAirfoilPipeline:
    def __init__(self,
                 llm_model_path="LLM/qwen3b-airfoil-final",
                 diffusion_model_path="all/models/lucid_thickness_camber_standardized_run_3/best_model.pt",
                 airfoil_data_path="all/coord_seligFmt",
                 airfoil_cache_path='all/airfoil_cache.pkl'):
        """
        初始化完整的NLP到翼型生成pipeline
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")

        # 1. 初始化LLM
        self._init_llm(llm_model_path)

        # 2. 初始化扩散模型
        self._init_diffusion_model(diffusion_model_path, airfoil_data_path, airfoil_cache_path)

        print("✅ Pipeline初始化完成!")

    def _init_llm(self, model_path):
        """初始化LLM模型"""
        print("🔄 正在加载LLM...")

        # 使用你已经写好的AirfoilLLMTester类
        self.llm_tester = AirfoilLLMTester(model_path)

        print("✅ LLM加载完成!")

    def _init_diffusion_model(self, model_path, data_path, cache_path):
        """初始化扩散模型"""
        print("🔄 正在加载扩散模型...")

        # 加载数据集和统计信息
        self.uiuc_dict = load_uiuc_airfoils('./all/uiuc_airfoils.pkl')
        dataset = AirfoilDataset(data_path, 100, cache_path)
        self.airfoil_x = dataset.get_x()

        # 加载扩散模型
        self.diffusion_model = Unet1DConditional(32, cond_dim=4, channels=2, dim_mults=(1, 2, 4)).to(self.device)
        self.diffusion_model.load_state_dict(torch.load(model_path, weights_only=True))
        self.diffusion_model.eval()
        self.diffusion = GaussianDiffusion1D(self.diffusion_model, seq_length=100).to(self.device)

        print("✅ 扩散模型加载完成!")

    def apply_physics_constraints(self, llm_output, user_input):
        """应用物理约束修正LLM输出"""
        try:
            params = json.loads(llm_output)

            # 对称翼型约束
            if any(word in user_input for word in ["对称", "symmetric"]):
                params["camber"] = 0.0
                params["cl"] = max(0.0, min(params["cl"], 0.3))

            # 高升力翼型约束
            if "高升力" in user_input:
                params["cl"] = max(1.0, params["cl"])
                params["camber"] = max(0.05, params["camber"])

            # 低阻力翼型约束
            if "低阻力" in user_input:
                params["cd"] = min(0.01, params["cd"])

            print('这里是测试  test testtest')
            print(params["cl"])
            print(params["cd"])

            # 物理边界约束
            params["cl"] = max(-0.5, min(params["cl"], 2.5))
            print('----------')
            print(params["cl"])


            params["cd"] = max(0.003, min(params["cd"], 0.5))
            print(params["cd"])


            params["camber"] = max(-0.1, min(params["camber"], 0.2))
            params["thickness"] = max(0.05, min(params["thickness"], 0.3))

            return json.dumps(params)
        except:
            return llm_output

    def nlp_to_parameters(self, user_input):
        """自然语言转换为翼型参数"""

        # 使用LLM测试器生成参数
        raw_response = self.llm_tester.generate_airfoil_params(user_input)

        print(f"🔍 LLM原始输出: {raw_response}")  # ← 新增

        # 判断是否有显式参数
        has_explicit_params = any(param in user_input for param in ["cl=", "cd=", "camber=", "thickness="])
        print(f"🔍 检测到显式参数: {has_explicit_params}")  # ← 新增

        if has_explicit_params:
            constrained_response = raw_response
            print(f"🔍 使用原始输出（不应用约束）")  # ← 新增
        else:
            constrained_response = self.apply_physics_constraints(raw_response, user_input)
            print(f"🔍 应用物理约束后: {constrained_response}")  # ← 新增



        try:
            parameters = json.loads(constrained_response)
            print(f"📋 解析参数: {parameters}")
            return parameters
        except:
            print(f"❌ JSON解析失败: {constrained_response}")
            return None

    def prepare_conditions(self, cl, cd, camber, thickness, batch_size=1):
        """准备扩散模型的条件输入"""

        # 创建批量数据
        cl_batch = torch.full((batch_size,), cl, dtype=torch.float32).to(self.device)
        cd_batch = torch.full((batch_size,), cd, dtype=torch.float32).to(self.device)
        camber_batch = torch.full((batch_size,), camber, dtype=torch.float32).to(self.device)
        thickness_batch = torch.full((batch_size,), thickness, dtype=torch.float32).to(self.device)

        uiuc_min_cl = torch.tensor(self.uiuc_dict['uiuc_min_cl'], dtype=torch.float32).to(self.device)
        uiuc_max_cl = torch.tensor(self.uiuc_dict['uiuc_max_cl'], dtype=torch.float32).to(self.device)
        uiuc_min_cd = torch.tensor(self.uiuc_dict['uiuc_min_cd'], dtype=torch.float32).to(self.device)
        uiuc_max_cd = torch.tensor(self.uiuc_dict['uiuc_max_cd'], dtype=torch.float32).to(self.device)
        uiuc_min_camber = torch.tensor(self.uiuc_dict['uiuc_min_camber'], dtype=torch.float32).to(self.device)
        uiuc_max_camber = torch.tensor(self.uiuc_dict['uiuc_max_camber'], dtype=torch.float32).to(self.device)
        uiuc_min_thickness = torch.tensor(self.uiuc_dict['uiuc_min_thickness'], dtype=torch.float32).to(self.device)
        uiuc_max_thickness = torch.tensor(self.uiuc_dict['uiuc_max_thickness'], dtype=torch.float32).to(self.device)



        # 标准化处理 - 和训练时一致
        cl_batch = normalize_conditioning_values(cl_batch, uiuc_min_cl, uiuc_max_cl)
        cd_batch = normalize_conditioning_values(cd_batch, uiuc_min_cd, uiuc_max_cd)
        camber_batch = normalize_conditioning_values(camber_batch, uiuc_min_camber, uiuc_max_camber)
        thickness_batch = normalize_conditioning_values(thickness_batch, uiuc_min_thickness, uiuc_max_thickness)

        # 偏移均值
        cl_batch = cl_batch + 2
        cd_batch = cd_batch + 2
        camber_batch = camber_batch + 2
        thickness_batch = thickness_batch + 2

        # 组合条件
        conditioning = torch.stack([cl_batch, cd_batch, thickness_batch, camber_batch], dim=1)


        return conditioning

    def generate_airfoil(self, user_input, num_samples=1, visualize=True):
        """
        完整的pipeline：自然语言 → 翼型几何
        """

        print(f"🎯 用户输入: {user_input}")

        # Step 1: NLP → 参数
        parameters = self.nlp_to_parameters(user_input)
        if parameters is None:
            print("❌ 参数解析失败")
            return None

        # Step 2: 参数 → 条件张量
        conditioning = self.prepare_conditions(
            cl=parameters["cl"],
            cd=parameters["cd"],
            camber=parameters["camber"],
            thickness=parameters["thickness"],
            batch_size=num_samples
        )

        print(f"📊 条件张量形状: {conditioning.shape}")
        print(f"条件情况: {conditioning}")

        # Step 3: 条件扩散生成
        print("🔄 正在生成翼型...")
        # airfoils = self.diffusion.sample(
        #     batch_size=num_samples,
        #     conditioning=conditioning
        # )
        #
        # # Step 4: 可视化
        # if visualize:
        #     self.visualize_results(airfoils, parameters, user_input)
        #
        # return airfoils, parameters

    def visualize_results(self, airfoils, parameters, user_input):
        """可视化生成结果"""

        num_airfoils = airfoils.shape[0]

        if num_airfoils == 1:
            # 单个翼型显示
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            y_coord_upper = airfoils[0, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[0, 1].cpu().detach().numpy()
            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, self.airfoil_x, s=0.1)

            ax.plot(self.airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f'生成翼型\nCL={parameters["cl"]:.3f}, CD={parameters["cd"]:.4f}, Camber={parameters["camber"]:.3f}, Thickness={parameters["thickness"]:.3f}')
            ax.set_xlabel('弦长位置')
            ax.set_ylabel('翼型厚度')

        else:
            # 多个翼型网格显示
            cols = min(4, num_airfoils)
            rows = (num_airfoils + cols - 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            if rows == 1:
                axs = [axs] if cols == 1 else axs
            else:
                axs = axs.flatten()

            for i in range(num_airfoils):
                ax = axs[i] if num_airfoils > 1 else axs

                y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
                y_coord_lower = airfoils[i, 1].cpu().detach().numpy()
                y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, self.airfoil_x, s=0.1)

                ax.plot(self.airfoil_x, y_coords, color='black', linewidth=2)
                ax.axis('equal')
                ax.axis('off')
                ax.set_title(f'样本 {i + 1}')

            # 隐藏多余的子图
            for i in range(num_airfoils, len(axs)):
                axs[i].axis('off')

        plt.suptitle(f'翼型生成结果\n输入: "{user_input}"', fontsize=14)
        plt.tight_layout()
        plt.show()

    def batch_test(self):
        """批量测试不同的自然语言输入"""

        test_cases = [
            "设计一个对称翼型",
            "设计一个高升力低阻力的翼型",
            "无人机长航时翼型设计",
            "民航客机巡航翼型",
            "风力发电叶片高效翼型"
        ]

        print("🧪 开始批量测试...")

        for i, case in enumerate(test_cases):
            print(f"\n{'=' * 60}")
            print(f"测试 {i + 1}/{len(test_cases)}: {case}")
            print("=" * 60)

            try:
                airfoils, params = self.generate_airfoil(case, num_samples=1, visualize=True)
                print(f"✅ 测试成功")
            except Exception as e:
                print(f"❌ 测试失败: {e}")


def npy_to_dat(npy_file, x_coords, output_name='airfoil'):


    temp_airfoil = npy_file

    # 分离上下表面
    y_upper = temp_airfoil[0]  # 前100个点是上表面
    y_lower = temp_airfoil[1]  # 后100个点是下表面
    y_upper = y_upper[::-1]

    # 组合成 (100, 2) 格式
    y_reshaped = np.column_stack([y_upper, y_lower])

    # 获取x坐标（只需要前100个）
    x = x_coords[:100][::-1]

    # 调用你的函数
    process_and_save_airfoil(x, y_reshaped, file_index=0, base_filename=output_name)
    print(f"✅ 转换完成: {output_name}_dwf.dat")


# 使用示例
if __name__ == "__main__":
    # 初始化pipeline
    pipeline = NLPToAirfoilPipeline()

    # 单个测试
    user_input = "设计一个对称翼型， cd=0.00539，cl=-0.0"
    #user_input = "设计一个模仿NACA0012的翼型"

    pipeline.generate_airfoil(user_input, 1, True)

    # airfoils, params = pipeline.generate_airfoil(user_input, 1, True)
    #
    # airfoil_x = pipeline.airfoil_x
    # airfoil_y = airfoils.cpu().squeeze(0).numpy()
    #
    # print(type(airfoil_y))
    # print(airfoil_y.shape)
    # print(airfoil_x.shape)
    #
    # npy_to_dat(airfoil_y, airfoil_x)



