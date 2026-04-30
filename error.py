
# 调用geo_pne_calculate  来计算那些误差

from geo_pne_calculate import *
from rmse_abs_calculate import *




def extract_airfoil_features(airfoils, alpha=0.0, Re=1e6, mach=0.0):
    """
    输入:
        airfoils: (N, 2, M)

    输出:
        t_max, t_idx, c_max, c_idx, CL, CD, CM
    """

    # x 坐标
    air_x = get_x_axis(airfoils.shape[2])

    # ---------- 几何 ----------
    thk_all, cam_all = compute_thickness_camber_batch(airfoils)
    t_max, t_idx, c_max, c_idx = extract_thickness_camber_max(thk_all, cam_all)

    # ---------- 气动 ----------
    CL, CD, CM = aero_coeff_from_airfoil_batch(
        airfoils,
        air_x,
        alpha=alpha,
        Re=Re,
        mach=mach
    )

    return t_max, t_idx, c_max, c_idx, CL, CD, CM


def evaluate_airfoils(airfoils, target_vals, target_airfoil):

    """
    计算一组翼型的所有误差
    """

    # ---------- 几何气动特征 ----------
    t_max, t_idx, c_max, c_idx, CL, CD, CM = extract_airfoil_features(airfoils)

    # ---------- shape RMSE ----------
    shape_rmse = calc_shape_rmse(airfoils, target_airfoil)

    results = {}

    results["shape_rmse"] = shape_rmse
    results["thickness_error"] = calc_abs_error(t_max, target_vals["thickness"])
    results["camber_error"] = calc_abs_error(c_max, target_vals["camber"])
    results["CL_error"] = calc_abs_error(CL, target_vals["cl"])
    results["CD_error"] = calc_abs_error(CD, target_vals["cd"])

    return results


def print_statistics(results):

    for key, values in results.items():

        mean = np.mean(values)
        std = np.std(values)

        print(f"{key}: {mean:.6f} ± {std:.6f}")







if __name__ == '__main__':

    # target参数
    target_vals = {
        "cl": 0.0,
        "cd": 0.00535,
        "thickness": 0.120,
        "camber": 0.0
    }
    target_airfoil = np.load('naca0012.npy')[0]


    target_vals_rae = {
        "cl": 0.2235,
        "cd": 0.00552,
        "thickness": 0.121,
        "camber": 0.0126
    }
    target_rae_airfoil = np.load('rae2822.npy')[0]



    # 数据路径
    paths = {
        "naca_only_Diffusion": "airfoils_naca_onlydiffusion.npy",
        "naca_LLM_Diffusion": "airfoils_naca_LLMdiffusion.npy",
        "rae_only_Diffusion": "airfoils_rae_onlydiffusion.npy",
        "rae_LLM_Diffusion": "airfoils_rae_LLMdiffusion.npy",
    }

    # 读取数据
    datasets = {}

    for name, path in paths.items():

        data = np.load(path)
        print(name, data.shape)
        # VAE/GAN需要transpose
        if data.shape[1] != 2:
            data = data.transpose(0 ,2 ,1)

        datasets[name] = data



    # ---------- 批量计算 ----------
    all_results = {}

    for name, airfoils in datasets.items():

        print(f"\nProcessing {name} ...")

        results = evaluate_airfoils(airfoils, target_vals, target_airfoil)
        all_results[name] = results


    for name, res in all_results.items():
        print(f"\n{name} statistics")
        print_statistics(res)



    import pandas as pd

    records = []

    for model_name, res in all_results.items():

        n = len(res["shape_rmse"])

        for i in range(n):
            records.append({
                "Model": model_name,
                "Target": "NACA0012",
                "shape_rmse": res["shape_rmse"][i],
                "thickness_error": res["thickness_error"][i],
                "camber_error": res["camber_error"][i],
                "CL_error": res["CL_error"][i],
                "CD_error": res["CD_error"][i]
            })

    df_errors = pd.DataFrame(records)

    print(df_errors.head())

    df_errors.to_csv("airfoil_compare_errors111.csv", index=False)

    print("Saved → airfoil_compare_errors.csv")



