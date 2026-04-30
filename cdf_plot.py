

import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文黑体
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False        # 负号正常显示



#
# # ---------------------- 1 读取数据 ----------------------
# df = pd.read_csv("airfoil_compare_errors.csv")
#
# # ---------------------- 2 修改模型名称 ----------------------
# df["Model"] = df["Model"].str.replace("only_Diffusion", "cDDPM", regex=False)
# df["Model"] = df["Model"].str.replace("LLM_Diffusion", "SC-cDDPM", regex=False)
#
# # ---------------------- 3 绘图风格 ----------------------
#
#
# plt.rcParams["axes.titlesize"] = 14
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["xtick.labelsize"] = 11
# plt.rcParams["ytick.labelsize"] = 11
#
# # ---------------------- 4 需要绘制的指标 ----------------------
# metrics = [
#     "shape_rmse",
#     "thickness_error",
#     "camber_error"
# ]
#
# # ---------------------- 5 画CDF ----------------------
#
# for metric in metrics:
#
#     plt.figure(figsize=(8,6))
#
#     style_map = {
#         "cDDPM": "--",
#         "SC-cDDPM": "-"
#     }
#
#     color_map = {
#         "naca": "#1f77b4",
#         "rae": "#d62728"
#     }
#
#     models = df["Model"].unique()
#
#     for model in models:
#
#         data = df[df["Model"] == model][metric].values
#
#         data_sorted = np.sort(data)
#         cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
#
#         # 解析模型名
#         parts = model.split("_")
#
#         airfoil = parts[0].lower()
#         method = parts[1]
#
#         # legend名字
#         if airfoil == "naca":
#             airfoil_name = "naca0012"
#         elif airfoil == "rae":
#             airfoil_name = "rae"
#         else:
#             airfoil_name = airfoil
#
#         label = f"{method}_{airfoil_name}"
#
#         # 颜色 + 线型
#         color = color_map.get(airfoil, "black")
#         linestyle = style_map.get(method, "-")
#
#         plt.plot(
#             data_sorted,
#             cdf,
#             color=color,
#             linestyle=linestyle,
#             linewidth=2,
#             label=label
#         )
#
#     plt.xlabel(metric)
#     plt.ylabel("CDF")
#
#     plt.title(f"{metric} CDF Comparison")
#
#     plt.legend()
#
#     plt.grid(True)
#
#     plt.tight_layout()
#
#     plt.savefig(f"{metric}_cdf.png", dpi=600)
#
#     # plt.show()




# ---------------------- 1 读取数据 ----------------------



df = pd.read_csv("airfoil_compare_errors.csv")

# ---------------------- 2 修改模型名称 ----------------------
df["Model"] = df["Model"].str.replace("only_Diffusion", "cDDPM", regex=False)
df["Model"] = df["Model"].str.replace("LLM_Diffusion", "SC-cDDPM", regex=False)

# ---------------------- 3 图形参数 ----------------------
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18

# ---------------------- 4 需要绘制的指标 ----------------------
metrics = [
    "shape_rmse",
    "thickness_error",
    "camber_error"
]

# 中文名称
metric_names = {
    "shape_rmse": "翼型形状重建误差",
    "thickness_error": "厚度误差",
    "camber_error": "弯度误差"
}

# ---------------------- 5 画CDF ----------------------

for metric in metrics:

    plt.figure(figsize=(8,6))

    style_map = {
        "cDDPM": "--",
        "SC-cDDPM": "-"
    }

    color_map = {
        "naca": "#1f77b4",
        "rae": "#d62728"
    }

    models = df["Model"].unique()

    for model in models:

        data = df[df["Model"] == model][metric].values

        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

        parts = model.split("_")
        airfoil = parts[0].lower()
        method = parts[1]

        if airfoil == "naca":
            airfoil_name = "NACA0012"
        elif airfoil == "rae":
            airfoil_name = "RAE2822"
        else:
            airfoil_name = airfoil

        label = f"{method}-{airfoil_name}"

        color = color_map.get(airfoil, "black")
        linestyle = style_map.get(method, "-")

        plt.plot(
            data_sorted,
            cdf,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label
        )

    # # 中文坐标轴
    # plt.xlabel(metric_names[metric])
    # plt.ylabel("累计分布函数")
    #
    # # 中文标题
    # plt.title(f"{metric_names[metric]}的CDF对比")

    # 英文坐标轴
    plt.xlabel(metric)
    plt.ylabel("CDF")

    # 中文标题
    plt.title(f"{metric} CDF Comparison")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{metric}_cdf_中文.png", dpi=600)

    # plt.show()