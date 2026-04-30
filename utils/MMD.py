


import torch




def rbf_kernel(x, y, sigma=1.0):
    """
    使用 RBF（高斯核）计算两个样本的核矩阵
    :param x: 样本 x，形状为 [batch_size, features]
    :param y: 样本 y，形状为 [batch_size, features]
    :param sigma: RBF 核的带宽参数
    :return: 核矩阵
    """
    dist = torch.cdist(x, y, p=2)  # 计算 x 和 y 之间的欧氏距离
    return torch.exp(- dist ** 2 / (2 * sigma ** 2))



def calculate_mmd(real_data, generated_data, sigma=1.0):
    """
    计算 MMD 值，衡量生成数据与真实数据的分布差异
    :param real_data: 验证集的真实数据
    :param generated_data: 模型生成的数据
    :param sigma: RBF 核的带宽参数
    :return: MMD 值
    """
    # 计算真实数据内部、生成数据内部以及真实数据与生成数据之间的核矩阵

    if not isinstance(real_data, torch.Tensor):
        real_data_tensor = torch.tensor(real_data, dtype=torch.float32)
    else:
        real_data_tensor = real_data

    if not isinstance(generated_data, torch.Tensor):
        generated_data_tensor = torch.tensor(generated_data, dtype=torch.float32)
    else:
        generated_data_tensor = generated_data



    K_real_real = rbf_kernel(real_data_tensor, real_data_tensor, sigma)
    K_gen_gen = rbf_kernel(generated_data_tensor, generated_data_tensor, sigma)
    K_real_gen = rbf_kernel(real_data_tensor, generated_data_tensor, sigma)

    # MMD 计算公式
    mmd_value = K_real_real.mean() + K_gen_gen.mean() - 2 * K_real_gen.mean()
    return mmd_value.item()




