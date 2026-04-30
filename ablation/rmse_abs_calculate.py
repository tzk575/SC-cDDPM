
#另一些工具   如何计算rmse和绝对值误差



import numpy as np
from scipy.interpolate import interp1d





def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
    x_new = x_newdata[1:] / 2
    return x_new


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

    upper = airfoils_flat[:, :M][:, ::-1]
    lower = airfoils_flat[:, M:]  # 翻转为 LE->TE

    airfoils = np.stack([upper, lower], axis=1)
    return airfoils



def calc_shape_rmse(gen_airfoils, target_64):
    """
    计算生成翼型与目标翼型的shape RMSE
    自动完成64→128插值

    input
        gen_airfoils : (N,2,128)
        target_64 : (64,2)

    output
        rmse : (N,)
    """

    # -------- 生成x坐标 --------
    x_target = get_x_axis(target_64.shape[0])
    x_gen = get_x_axis(gen_airfoils.shape[2])

    # -------- 插值target --------
    upper = target_64[:,0]
    lower = target_64[:,1]

    f_upper = interp1d(x_target, upper, kind='linear', fill_value="extrapolate")
    f_lower = interp1d(x_target, lower, kind='linear', fill_value="extrapolate")

    upper_new = f_upper(x_gen)
    lower_new = f_lower(x_gen)

    target_new = np.vstack([upper_new, lower_new])


    # -------- RMSE计算 --------
    target_expand = np.expand_dims(target_new, axis=0)  #加一层括号

    mse = np.mean((gen_airfoils - target_expand) ** 2, axis=(1,2))
    rmse = np.sqrt(mse)

    return rmse



def calc_abs_error(values, target_value):
    """
    计算数组中每个元素与目标值的绝对误差

    input
        values: (N,)  预测值数组
        target_value: float  目标值

    output
        abs_error: (N,)
    """

    values = np.asarray(values).reshape(-1)

    abs_error = np.abs(values - target_value)

    return abs_error



if __name__ == '__main__':


    diffusion_airfoils = np.load('111airfoils_data_rae2822.npy') * 1.55
    target_airfoil = np.load('rae2822.npy')[0]
    print(diffusion_airfoils.shape)
    print(target_airfoil.shape)

    diffusion_rmse = calc_shape_rmse(diffusion_airfoils, target_airfoil)
    print(diffusion_rmse)
    print('================================')






