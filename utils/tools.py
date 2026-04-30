
import numpy as np
import torch
from config.params import parse_args




params = parse_args()
default_device = params.device


def get_cuda(tensor, device=default_device):

    tensor = tensor.to(device)
    return tensor




def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
    x_new = x_newdata[1:] / 2
    return x_new



def trans_size(airfoil):


    # 检查输入类型并统一处理逻辑
    is_tensor = isinstance(airfoil, torch.Tensor)
    is_numpy = isinstance(airfoil, np.ndarray)

    if not (is_tensor or is_numpy):
        raise TypeError("输入必须是 torch.Tensor 或 numpy.ndarray 类型")

    # 提取上表面和下表面数据 (N, M)
    upper = airfoil[:, :, 0]  # 对应原代码的 airfoil[:, :, 0]
    lower = airfoil[:, :, 1]  # 对应原代码的 airfoil[:, :, 1]

    # 逆置上表面数据并去掉开头的第一个元素
    if is_tensor:
        # PyTorch 张量的翻转操作
        reversed_upper = torch.flip(upper[:, 1:], dims=[1])
        # 拼接操作
        combined_data = torch.cat((reversed_upper, lower), dim=1)
    else:
        # NumPy 数组的翻转操作
        reversed_upper = np.flip(upper[:, 1:], axis=1)
        # 拼接操作
        combined_data = np.concatenate((reversed_upper, lower), axis=1)

    return combined_data


def redo_trans(airfoil):
    n_points = airfoil.shape[1] // 2 + 1


    upper = airfoil[:, :n_points]
    lower = airfoil[:, n_points - 1:]




def airfoils_flat_to_split(airfoils_flat):

    #从摊平的翼型  回到(2, 点数)的格式
    # 「前缘 (LE)→上表面→后缘 (TE)→下表面→前缘 (LE)」

    airfoils_flat = np.asarray(airfoils_flat)
    if airfoils_flat.ndim != 2:
        raise ValueError("Expect (N, 2*M)")

    N, twoM = airfoils_flat.shape
    if twoM % 2 != 0:
        raise ValueError("Second dim must be even")

    M = twoM // 2

    upper = airfoils_flat[:, :M]
    lower = airfoils_flat[:, M:][:, ::-1]  # 翻转为 LE->TE

    airfoils = np.stack([upper, lower], axis=1)
    return airfoils



