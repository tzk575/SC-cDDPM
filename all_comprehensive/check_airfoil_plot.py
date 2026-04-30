
#用于检测生成的翼型
#能够通过可视化直接观察
#同时可以得到一份一份的.npy文件  也可以转变为.dat文件




import numpy as np
import matplotlib.pyplot as plt
import os
from utils.npy_to_dat import process_and_save_airfoil



def plot_single_airfoil(x_coords, y_coords, index=0, show_flag=True, save_path='.'):

    # 获取单个翼型
    temp_airfoil = y_coords[index]  # (200,)

    # 重新组织格式：前100个是上表面，后100个是下表面
    y_upper = temp_airfoil[:100]
    y_lower = temp_airfoil[100:]
    y_lower = y_lower[::-1]

    # 组合成 (100, 2) 格式
    airfoil_reshaped = np.column_stack([y_upper, y_lower])

    # 获取x坐标（取前100个点）
    x = x_coords[:100]

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_upper, '-o', label="Upper Surface", markersize=2)
    plt.plot(x, y_lower, '-o', label="Lower Surface", markersize=2)

    plt.axis('equal')
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"Airfoil {index}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if show_flag:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/airfoil_{index}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"保存到: {filename}")

    plt.close()


def npy_to_dat(npy_file, x_coords, index, output_name='airfoil'):

    print(type(npy_file))
    temp_airfoil = npy_file

    # 分离上下表面
    y_upper = temp_airfoil[:100]  # 前100个点是上表面
    y_lower = temp_airfoil[100:]  # 后100个点是下表面
    y_upper = y_upper[::-1]

    # 组合成 (100, 2) 格式
    y_reshaped = np.column_stack([y_upper, y_lower])

    # 获取x坐标（只需要前100个）
    x = x_coords[:100][::-1]

    # 调用你的函数
    process_and_save_airfoil(x, y_reshaped, file_index=index, base_filename=output_name)
    print(f"✅ 转换完成: {output_name}_{index}.dat")



def load_all_airfoils(save_path='airfoils_data.npz'):
    """加载所有翼型数据"""
    data = np.load(save_path)
    return data['x'], data['y']



# 读取方法：
x_coords, y_coords = load_all_airfoils('airfoils_data.npz')

print(f"x坐标: {x_coords.shape}, y坐标: {y_coords.shape}")

#TE(上表面) → LE → TE(下表面)


for index in range(16):

    temp_airfoil = y_coords[index]
    npy_to_dat(temp_airfoil, x_coords, index)  #将一个numpy数组转变为一个dat文件  供xfoil计算


#plot_single_airfoil(x_coords, y_coords, index, show_flag=True)

