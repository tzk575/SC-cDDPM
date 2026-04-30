

import math
import random
import numpy as np

import matplotlib.pyplot as plt
from utils.tools import get_x_axis
from scipy import interpolate
from scipy.interpolate import UnivariateSpline, CubicSpline

#测试  一阶导  二阶导  曲率  曲率变化率对问题点的改进

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置英文字体为 Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def compute_derivatives_spline(x, y):
    # 使用样条插值计算导数
    spline = CubicSpline(x, y)
    dy_dx = spline(x, 1)
    d2y_dx2 = spline(x, 2)

    return dy_dx, d2y_dx2

def approximate_derivatives_spline(x, y):
    dy, d2y = compute_derivatives_spline(x, y)

    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    return curvature


# 绘制翼型和曲率的函数
def plot_airfoil_and_curvature(x_t, y_t_upper, y_t_lower, name, num_interval):
    curvature_upper = approximate_derivatives_spline(x_t, y_t_upper)
    curvature_lower = approximate_derivatives_spline(x_t, y_t_lower)

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # 绘制翼型
    #ax1.axis('equal')
    ax2.plot(x_t[::num_interval], y_t_upper[::num_interval], '-', color='C0', label='上表面', linewidth=4)
    ax2.plot(x_t[::num_interval], y_t_lower[::num_interval], '-', color='C1', label='下表面', linewidth=4)
    ax2.set_title('翼型', fontsize=15)
    ax2.set_xlabel('x/c', fontsize=15)
    ax2.set_ylabel('y', fontsize=15)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax2.legend(fontsize=15)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=12.5)


    ax3.plot(x_t[::num_interval], curvature_upper[::num_interval], color='C0', linewidth=2)
    ax3.plot(x_t[::num_interval], -curvature_lower[::num_interval], color='C1', linewidth=2)
    #ax2.axhline(0, color='black', linewidth=1)
    ax3.set_title('曲率分布', fontsize=15)
    ax3.set_xlabel('x/c', fontsize=15)
    ax3.set_ylabel('曲率', fontsize=15)
    ax3.set_xlim(0.0, 1.0)
    #ax2.set_xticks(np.arange(0.0, 0.21, 0.05))
    ax3.set_ylim(-4, 4)
    ax3.set_yticks(np.arange(-4, 4.1, 2))
    ax3.set_yticklabels([f'{abs(tick)}' for tick in np.arange(-4, 4.1, 2)])
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=12.5)




    # ax1.plot(x_t[::num_interval], curvature_upper[::num_interval], color='C0', linewidth=2)
    # ax1.plot(x_t[::num_interval], -curvature_lower[::num_interval], color='C1', linewidth=2)
    # #ax3.axhline(0, color='black', linewidth=1)
    # ax1.set_title('前缘细节', fontsize=15)
    # ax1.set_xlabel('x/c', fontsize=15)
    # ax1.set_ylabel('曲率', fontsize=15)
    # ax1.set_xlim(0.04, 0.21)
    # ax1.set_xticks(np.arange(0.06, 0.21, 0.05))
    # ax1.set_ylim(-2.0, 2.0)
    # ax1.set_yticks(np.arange(-2, 2.1, 1))
    # ax1.set_yticklabels([f'{abs(tick)}' for tick in np.arange(-2, 2.1, 1)])
    # ax1.grid(True)
    # ax1.tick_params(axis='both', which='major', labelsize=12.5)


    plt.tight_layout()
    plt.savefig(f'curvature_{name}.png', dpi=300)
    plt.show()

    plt.close()



def airfoil_curvature(airfoil, num_interval, index):
    n_points = airfoil.shape[0]
    x_coords = get_x_axis(n_points)
    x = x_coords
    y_upper = airfoil[:, 0]
    y_lower = airfoil[:, 1]

    plot_airfoil_and_curvature(x, y_upper, y_lower, index, num_interval)

#
# # 加载数据并随机选择一个翼型
# path = '../data/train_data.npy'
# data = np.load(path)
#
# for i in range(50, 60):
#
#
#     airfoil = data[i]
#     airfoil_curvature(airfoil)



def calculate_curvature_loss(airfoil):

    print(1)