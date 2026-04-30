import math
import random
import numpy as np
import torch
import torch.nn as nn




# 将npy文件 保存为 .dat 文件
def process_and_save_airfoil(x_coords, y_coords, file_index, base_filename='airfoil'):

    # 将上下两个表面的数据分别处理
    upper_surface = np.column_stack((x_coords, y_coords[:, 0]))
    lower_surface = np.column_stack((x_coords, y_coords[:, 1]))

    # 重新排序：上表面从1.0到0.0，下表面从0.0到1.0
    upper_sorted = upper_surface[::-1]
    lower_sorted = lower_surface

    # 将上下表面的数据垂直堆叠
    combined_data_sorted = np.vstack((upper_sorted, lower_sorted))

    # 生成 .dat 文件路径
    dat_file_path = f'{base_filename}_{file_index}.dat'

    # 保存为 .dat 文件
    with open(dat_file_path, 'w') as f:
        # 写入翼型名字
        f.write(f'{base_filename}_0_{file_index}\n')
        # 写入数据
        np.savetxt(f, combined_data_sorted, fmt='%.6f', delimiter=' ')










