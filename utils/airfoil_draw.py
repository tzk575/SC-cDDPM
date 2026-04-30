
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils.tools import get_x_axis




def airfoil_plot(airfoil, show_flag, index=1, save_path='.'):
    n_points = airfoil.shape[0]

    if not isinstance(airfoil, np.ndarray):
        airfoil = airfoil.detach().cpu().numpy()


    y_up = airfoil[:, 0]
    y_low = airfoil[:, 1]

    y_up = y_up.reshape(-1, 1)
    y_low = y_low.reshape(-1, 1)


    # 生成x坐标
    air_x = get_x_axis(n_points)
    x = air_x

    # 绘制翼型图像
    plt.figure()
    #plt.axis("equal")

    plt.plot(x, y_up, '-o', label="Upper Surface", markersize=4)
    plt.plot(x, y_low, '-o', label="Lower Surface", markersize=4)

    plt.ylim(-0.25, 0.25)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title("Airfoil")
    plt.legend()
    plt.grid(True)


    if show_flag:
        plt.show()

    else:
        filename = f"{save_path}/{index}.png"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(filename, dpi=300)


    plt.close()





def draw_angles(array, index, save_path):
    plt.figure()
    plt.plot(range(1, len(array) + 1), array, marker='o', label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    filename = f"{save_path}/angles_{index}.png"

    plt.savefig(filename, dpi=300)
    #plt.show()
    plt.close()









