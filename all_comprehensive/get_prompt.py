import json
import random
import numpy as np
import re
from scipy.stats import norm
import torch


def extract_params_and_sample(prompt, device=None):
    """
    从prompt中自动提取翼型个数和参数（支持范围【min,max】或定值=数值），
    范围则按正态分布采样，定值则直接复用，最终输出训练格式的torch张量
    :param prompt: 输入模板文本（支持范围/定值两种格式）
    :param device: 设备（cpu/cuda），默认自动判断
    :return: conditioning张量([num_airfoils,4])，翼型个数
    """
    # 自动判断设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_text = (
        '翼型数据描述了由一组离散的翼型坐标点表示组成，\n'
        '通过在给定设计条件下翼型几何形状的生成过程，\n'
        '以下是输入条件的说明：\n'
        '\n'
        '[BEGIN DATA]\n'
        '[领域]：我们关注的是二维翼型的几何建模，翼型由坐标点表示。\n'
        '[指令]：根据给定的条件信息，生成满足约束条件的翼型几何模型。\n'
        '\n'
        '[条件统计]：\n'
        '升力系数为<cl>，范围/定值：\n'
        '阻力系数为<cd>，范围/定值：\n'
        '翼型最大厚度为<max_thickness>，范围/定值：\n'
        '最大弯度为<max_camber>，范围/定值：\n'
        '\n'
        '[END DATA]\n'
    )
    print(prompt_text)

    # ========== 1. 提取翼型个数 ==========
    num_pattern = re.compile(r'(所需翼型个数|翼型个数|翼型数量)：?\s*(\d+)', re.IGNORECASE)
    num_match = num_pattern.search(prompt)
    num_airfoils = int(num_match.group(2)) if num_match else 10

    # ========== 2. 定义参数匹配规则（同时支持范围/定值） ==========
    # 格式：(匹配关键词, 参数标识)
    param_key_pairs = [
        ('cl', 'cl'),
        ('cd', 'cd'),
        ('厚度|thickness', '厚度'),
        ('弯度|camber', '弯度')
    ]
    param_config = {}  # 存储参数类型（range/fixed）和值

    for match_key, param_key in param_key_pairs:
        # 正则1：匹配范围格式 关键词：【min,max】
        range_pattern = re.compile(f'({match_key})：[【\[](.*?),(.*?)[】\]]', re.IGNORECASE)
        # 正则2：匹配定值格式 关键词：=数值 （兼容空格，如cl： = 0.1）
        fixed_pattern = re.compile(f'({match_key})：?\s*=\s*(-?\d+\.?\d*)', re.IGNORECASE)

        range_match = range_pattern.search(prompt)
        fixed_match = fixed_pattern.search(prompt)

        if range_match:
            # 匹配到范围：存储范围值
            min_val = float(range_match.group(2))
            max_val = float(range_match.group(3))
            param_config[param_key] = {
                'type': 'range',
                'value': (min_val, max_val)
            }
        elif fixed_match:
            # 匹配到定值：存储固定值
            fixed_val = float(fixed_match.group(2))
            param_config[param_key] = {
                'type': 'fixed',
                'value': fixed_val
            }
        else:
            # 无匹配：默认范围(0.0, 0.2)
            param_config[param_key] = {
                'type': 'range',
                'value': (0.0, 0.2)
            }

    # ========== 3. 按参数类型生成样本 ==========
    param_keys = ['cl', 'cd', '厚度', '弯度']
    sampled_params = []

    for key in param_keys:
        config = param_config[key]
        if config['type'] == 'range':
            # 范围：正态分布采样+裁剪
            min_v, max_v = config['value']
            mu = (min_v + max_v) / 2
            sigma = (max_v - min_v) / 6
            samples = norm.rvs(loc=mu, scale=sigma, size=num_airfoils)
            samples = np.clip(samples, min_v, max_v)
        else:
            # 定值：生成num_airfoils个相同值的数组
            fixed_val = config['value']
            samples = np.full(num_airfoils, fixed_val, dtype=np.float32)

        sampled_params.append(samples)

    # ========== 4. 复刻训练格式拼接张量 ==========
    cl_batch = torch.tensor(sampled_params[0], dtype=torch.float32)
    cd_batch = torch.tensor(sampled_params[1], dtype=torch.float32)
    thickness_batch = torch.tensor(sampled_params[2], dtype=torch.float32)
    camber_batch = torch.tensor(sampled_params[3], dtype=torch.float32)

    conditioning = torch.cat([
        cl_batch.unsqueeze(1),
        cd_batch.unsqueeze(1),
        thickness_batch.unsqueeze(1),
        camber_batch.unsqueeze(1)
    ], dim=1).to(device)

    return conditioning, num_airfoils


def get_condition_from_text(text, device=None):
    """
    新增函数：从文本提取参数并返回cl/cd/厚度/弯度的列表 + 拼接好的condition张量
    方便后续在generate_airfoils.py中进行标准化等后续处理
    :param text: 输入的参数描述文本
    :param device: 设备（cpu/cuda）
    :return:
        cl_list: 升力系数列表 (numpy.array)
        cd_list: 阻力系数列表 (numpy.array)
        thickness_list: 厚度列表 (numpy.array)
        camber_list: 弯度列表 (numpy.array)
        condition_tensor: 拼接后的condition张量 (torch.Tensor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先调用原有函数获取condition张量和翼型数量
    condition_tensor, num_airfoils = extract_params_and_sample(text, device)

    # 将张量拆分为单独的参数列表（转换为numpy数组，方便后续处理）
    cl_list = condition_tensor[:, 0].cpu().numpy()
    cd_list = condition_tensor[:, 1].cpu().numpy()
    thickness_list = condition_tensor[:, 2].cpu().numpy()
    camber_list = condition_tensor[:, 3].cpu().numpy()

    # print('================')
    # print(cl_list)
    # print(cd_list)
    # print(thickness_list)
    # print(camber_list)
    # print('====================')


    # 返回单独的列表 + 张量，兼顾后续处理的灵活性
    return cl_list, cd_list, thickness_list, camber_list, condition_tensor


# if __name__ == "__main__":
#     # 测试用例
#     # prompt3 = """
#     # 翼型数量：8
#     # 相关参数需求：
#     # cl：[0.1,0.3]
#     # cd：[0.001,0.003]
#     # thickness：[0.08,0.15]
#     # 弯度：[0.01,0.06]
#     # """
#
#
#     prompt3 = """
#     翼型数量：8
#     相关参数需求：
#     cl：= -0.0
#     cd：= 0.00539
#     thickness：[0.00,0.3]
#     弯度：= 0.0
#     其他需求：
#     模仿uiuc翼型库。
#     """
#
#     # 测试新增函数
#     cl_list, cd_list, thickness_list, camber_list, condition = get_condition_from_text(prompt3)
#
#     print("=== 新增函数测试结果 ===", 1111111111111)
#     print(f"翼型数量：{len(cl_list)}")
#     print(f"CL列表：{cl_list}")
#     print(f"CD列表：{cd_list}")
#     print(f"厚度列表：{thickness_list}")
#     print(f"弯度列表：{camber_list}")
#     print(f"Condition张量形状：{condition.shape}")
#     print(f"Condition张量（前3行）：\n{condition[:3]}")




