

#一些工具   如何计算批量翼型的四大指标
# 这个是用来被调用的


import numpy as np
import aerosandbox as asb


def get_x_axis(num):
    x_n = num
    theta = np.zeros([x_n + 1])
    x_newdata = np.zeros(x_n + 1)
    for i in range(1, x_n + 1):
        theta[i] = np.pi * (i - 1) / x_n
        x_newdata[i] = 1 - np.cos(theta[i])
    x_new = x_newdata[1:] / 2
    return x_new

def compute_thickness_camber_batch(airfoils: np.ndarray):
    """
    airfoils: (N, 2, M)

    return:
        thickness: (N, M)
        camber:    (N, M)
    """
    airfoils = np.asarray(airfoils)
    if airfoils.ndim != 3 or airfoils.shape[1] != 2:
        raise ValueError("Expect airfoils shape (N, 2, M)")

    y_upper = airfoils[:, 0, :]
    y_lower = airfoils[:, 1, :]

    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)


    return thickness, camber

def extract_thickness_camber_max(thickness, camber):
    """
    thickness:
        (M,)    or (N, M)
    camber:
        (M,)    or (N, M)

    return:
        单个翼型:
            t_max, t_idx, c_max, c_idx
        批量翼型:
            t_max, t_idx, c_max, c_idx
            shape: (N,)
    """
    thickness = np.asarray(thickness)
    camber = np.asarray(camber)

    if thickness.shape != camber.shape:
        raise ValueError("thickness and camber must have the same shape")

    # -------- 单个翼型 --------
    if thickness.ndim == 1:
        t_idx = int(np.argmax(thickness))
        c_idx = int(np.argmax(camber))

        t_max = float(thickness[t_idx])
        c_max = float(camber[c_idx])

        return t_max, t_idx, c_max, c_idx

    # -------- 批量翼型 --------
    elif thickness.ndim == 2:
        t_idx = np.argmax(thickness, axis=1)
        c_idx = np.argmax(camber, axis=1)

        t_max = thickness[np.arange(thickness.shape[0]), t_idx]
        c_max = camber[np.arange(camber.shape[0]), c_idx]

        return t_max, t_idx, c_max, c_idx

    else:
        raise ValueError("Expect thickness dim 1 or 2")

def aero_coeff_from_airfoil_single(airfoil, x, alpha=0.0, Re=1e6, mach=0.0):
    """
    airfoil: (2, M)  [upper_y, lower_y]
    x:        (M,)

    return:
        dict with CL, CD, CM
    """
    coords = np.vstack([
        np.stack([x, airfoil[0]], axis=1),      # upper
        np.stack([x[::-1], airfoil[1][::-1]], axis=1)  # lower
    ])

    af = asb.Airfoil(coordinates=coords)
    coef = af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=mach)

    return {
        "CL": float(coef["CL"][0]),
        "CD": float(coef["CD"][0]),
        "CM": float(coef["CM"][0])
    }

def aero_coeff_from_airfoil_batch(airfoils, x, alpha=0.0, Re=1e6, mach=0.0):

    """
    airfoils: (N, 2, M)

    return:
        CL, CD, CM: (N,)
    """
    CL, CD, CM = [], [], []

    for i in range(airfoils.shape[0]):
        coef = aero_coeff_from_airfoil_single(
            airfoils[i], x,
            alpha=alpha, Re=Re, mach=mach
        )
        CL.append(coef["CL"])
        CD.append(coef["CD"])
        CM.append(coef["CM"])

    return np.array(CL), np.array(CD), np.array(CM)


#extract_airfoil_features


# def extract_airfoil_features(airfoils, alpha=0.0, Re=1e6, mach=0.0):
#     """
#     输入:
#         airfoils: (N, 2, M)
#
#     输出:
#         t_max, t_idx, c_max, c_idx, CL, CD, CM
#     """
#
#     # x 坐标
#     air_x = get_x_axis(airfoils.shape[2])
#
#     # ---------- 几何 ----------
#     thk_all, cam_all = compute_thickness_camber_batch(airfoils)
#     t_max, t_idx, c_max, c_idx = extract_thickness_camber_max(thk_all, cam_all)
#
#     # ---------- 气动 ----------
#     CL, CD, CM = aero_coeff_from_airfoil_batch(
#         airfoils,
#         air_x,
#         alpha=alpha,
#         Re=Re,
#         mach=mach
#     )
#
#     return t_max, t_idx, c_max, c_idx, CL, CD, CM




def main():
    path = 'airfoils_data_new.npy'
    airfoils = np.load(path)
    air_x = get_x_axis(airfoils.shape[2])


    thk_all, cam_all = compute_thickness_camber_batch(airfoils)
    thk0, th0_idx, cam0, cam0_idx = extract_thickness_camber_max(thk_all, cam_all)
    print(thk0.shape)
    print(cam0.shape)


    airfoil_CL, airfoil_CD, airfoil_CM = aero_coeff_from_airfoil_batch(airfoils, air_x)

    print('-----------------------')
    print(airfoil_CL.shape)
    print(airfoil_CD.shape)


if __name__ == '__main__':
    main()



