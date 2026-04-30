
import numpy as np


def calculate_thickness_camber(airfoil_coords):
    """
    计算翼型的厚度和弯度分布

    参数:
        airfoil_coords: numpy数组，形状为 (n_points,) 或 (batch_size, n_points)
                       前半部分是上表面y坐标，后半部分是下表面y坐标

    返回:
        thickness: 厚度分布 (上表面 - 下表面)
        camber: 弯度分布 (上表面 + 下表面) / 2
    """

    # 如果是1维数组，添加一个维度
    if airfoil_coords.ndim == 1:
        airfoil_coords = airfoil_coords[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    # 计算中点位置
    middle_point = airfoil_coords.shape[1] // 2 + 1

    # 提取上表面和下表面（注意下表面需要反转顺序以匹配x坐标）
    upper_surface = airfoil_coords[:, 0:middle_point]
    lower_surface = airfoil_coords[:, -1:- 1 -middle_point:-1]

    # 计算厚度：上表面 - 下表面
    thickness = (upper_surface - lower_surface)[:, ::-1]  # 反转以匹配从前缘到后缘

    # 计算弯度：(上表面 + 下表面) / 2
    camber = ((upper_surface + lower_surface) / 2)[:, ::-1]  # 反转以匹配从前缘到后缘

    # 如果输入是1维，返回1维
    if squeeze_output:
        thickness = thickness.squeeze()
        camber = camber.squeeze()

    return thickness, camber


def calculate_max_thickness_camber(airfoil_coords):
    """
    计算翼型的最大厚度和最大弯度

    参数:
        airfoil_coords: numpy数组，形状为 (n_points,) 或 (batch_size, n_points)

    返回:
        max_thickness: 最大厚度值
        max_camber: 最大弯度值（绝对值）
    """
    thickness, camber = calculate_thickness_camber(airfoil_coords)

    if airfoil_coords.ndim == 1:
        max_thickness = np.max(thickness)
        max_camber = np.max(np.abs(camber))
    else:
        max_thickness = np.max(thickness, axis=1)
        max_camber = np.max(np.abs(camber), axis=1)

    return max_thickness, max_camber


# ============= 修正后的批量处理代码 =============
def batch_process_airfoils(num='512_'):
    """
    批量处理不同阶数的Bernstein翼型数据
    """
    name = 'test_' + num + 'Bernstein_order_'
    type = '.npy'

    # 加载数据
    order4 = np.load(name + '4' + type)
    order6 = np.load(name + '6' + type)
    order8 = np.load(name + '8' + type)
    order10 = np.load(name + '10' + type)
    order12 = np.load(name + '12' + type)

    print(f"数据形状: {order6.shape}")

    # 对每个阶数计算厚度和弯度
    orders = {
        '4': order4,
        '6': order6,
        '8': order8,
        '10': order10,
        '12': order12
    }

    for order_name, order_data in orders.items():
        thickness, camber = calculate_thickness_camber(order_data)

        # 保存结果
        np.save(f"{name}thickness_{order_name}{type}", thickness)
        np.save(f"{name}camber_{order_name}{type}", camber)

        print(f"Order {order_name}: 厚度形状 {thickness.shape}, 弯度形状 {camber.shape}")




if __name__ == "__main__":
    # 示例：单个翼型
    print("=== 单个翼型示例 ===")
    # 假设有一个翼型的坐标数据（这里用随机数据示例）
    n_points = 100
    single_airfoil = np.random.randn(n_points)

    thickness, camber = calculate_thickness_camber(single_airfoil)
    max_t, max_c = calculate_max_thickness_camber(single_airfoil)

    print(f"厚度分布形状: {thickness.shape}")
    print(f"弯度分布形状: {camber.shape}")
    print(f"最大厚度: {max_t:.4f}")
    print(f"最大弯度: {max_c:.4f}")

    # 示例：批量翼型
    print("\n=== 批量翼型示例 ===")
    batch_airfoils = np.random.randn(10, n_points)  # 10个翼型

    thickness_batch, camber_batch = calculate_thickness_camber(batch_airfoils)
    max_t_batch, max_c_batch = calculate_max_thickness_camber(batch_airfoils)

    print(f"批量厚度分布形状: {thickness_batch.shape}")
    print(f"批量弯度分布形状: {camber_batch.shape}")
    print(f"批量最大厚度: {max_t_batch.shape}")
    print(f"批量最大弯度: {max_c_batch.shape}")

