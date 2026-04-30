import torch
from utils.tools import get_cuda, get_x_axis
from scipy.interpolate import CubicSpline



def vector_airfoil(airfoil):
    # 计算翼型坐标点 对应的向量
    # 分了上下两个表面  先上表面后下表面
    # 128*2 点 → 127*2 向量

    batch_size, n_points, _ = airfoil.size()


    upper = airfoil[:, :, 0]
    lower = airfoil[:, :, 1]
    vector_y_upper = -torch.diff(upper, dim=1)
    vector_y_lower = torch.diff(lower, dim=1)
    vector_y_tail = (upper[:, -1] - lower[:, -1]).unsqueeze(-1)

    x = torch.tensor(get_x_axis(n_points), dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(airfoil.device)
    vector_x_upper = -torch.diff(x)
    vector_x_lower = torch.diff(x)
    vector_x_tail = torch.zeros_like(vector_y_tail)


    #upper从右往左   lower从左往右
    vector_upper = torch.stack((vector_x_upper, vector_y_upper), dim=2)
    vector_lower = torch.stack((vector_x_lower, vector_y_lower), dim=2)
    vector_tail = torch.stack((vector_x_tail, vector_y_tail), dim=2)

    vector_lower = torch.flip(vector_lower, dims=[1])


    vector_combined = torch.cat((vector_upper, vector_tail, vector_lower), dim=1)
    return vector_combined


def angle_between_vectors(vector):
    #vector = vector_upper, vector_tail, vector_lower (vector_lower还是逆置过的   保证头尾相连)
    #vector_tail是那一根竖起来的

    epsilon = 1e-7
    batch_size, num_vector, _ = vector.size()
    n_points = num_vector // 2 + 1

    # upper从右往左   lower从左往右
    # 但是从tensor的角度来看   都是从左往右存储的  所以有了上面说vector_lower逆置
    # 将第一个向量复制并放到最后，形成 256 个向量，确保环形
    extended_vector = torch.cat((vector, vector[:, :1, :]), dim=1)

    # 计算所有相邻向量的点积 范数
    dot_products = torch.sum(extended_vector[:, :-1, :] * extended_vector[:, 1:, :], dim=2)
    norms_a = torch.norm(extended_vector[:, :-1, :], dim=2)
    norms_b = torch.norm(extended_vector[:, 1:, :], dim=2)


    # 计算夹角的余弦值 和 夹角
    cos_theta = torch.clamp(dot_products / (norms_a * norms_b), -1.0 + epsilon, 1.0 - epsilon)
    angles = torch.acos(cos_theta)
    true_angles = angles * 180 / torch.pi
    max_angle, max_index = torch.max(true_angles, dim=1, keepdim=True)


    # 顺序是上表面最前缘（非头）→ 尾巴 → 竖着的向量 → 尾巴 下表面最前缘 → 头
    # print(true_angles)
    # print('average angle', torch.mean(true_angles, dim=1))
    # print('max angle', max_angle, max_index)

    return angles, true_angles



def angle_all(airfoil):
    batch_size = airfoil.size(0)
    vector_all = vector_airfoil(airfoil)

    angles, true_angles = angle_between_vectors(vector_all)

    # 对每一个翼型求角度和  弧度制
    sum_angles = torch.sum(angles, dim=1, keepdim=True)


    print('the airfoils angle as following')
    for i in range(batch_size):

        print(f'airfoil {i}: {sum_angles[i]}')
        print(i, '----------------------------')
        print(true_angles[i])


    # 对有多少个翼型求一个平均
    angles_loss = torch.mean(sum_angles)

    return angles_loss, true_angles



def delta_angles(airfoil):
    batch_size = airfoil.size(0)

    vector_all = vector_airfoil(airfoil)

    angles, true_angles = angle_between_vectors(vector_all)



    delta_angles = torch.diff(angles, dim=1)


    # for i in range(batch_size):
    #     print(f'airfoil {i}: {true_angles[i]}')
    #     print(f'airfoil {i}: {delta_angles[i]}')
    #     print('-----------------------------------')
    #

    delta_angles_loss = torch.mean(delta_angles)

    return true_angles, delta_angles
