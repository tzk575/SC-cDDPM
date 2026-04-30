

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde





#
# # ---------------------- 1 读取数据 ----------------------
# df = pd.read_csv("airfoil_compare_errors.csv")
#
# # ---------------------- 2 修改模型名称 ----------------------
# df["Model"] = df["Model"].str.replace("only_Diffusion", "cDDPM", regex=False)
# df["Model"] = df["Model"].str.replace("LLM_Diffusion", "SC-cDDPM", regex=False)
#
# # ---------------------- 3 绘图风格 ----------------------
# sns.set_style("whitegrid")
#
# plt.rcParams["axes.titlesize"] = 14
# plt.rcParams["axes.labelsize"] = 12
# plt.rcParams["xtick.labelsize"] = 11
# plt.rcParams["ytick.labelsize"] = 11
#
# # ---------------------- 4 指标 ----------------------
# metrics = [
#     "CL_error",
#     "CD_error"
# ]
#
# # ---------------------- 5 颜色和线型 ----------------------
# style_map = {
#     "cDDPM": "--",
#     "SC-cDDPM": "-"
# }
#
# color_map = {
#     "naca": "#1f77b4",
#     "rae": "#d62728"
# }
#
# # ---------------------- 6 绘制Density ----------------------
# for metric in metrics:
#
#     plt.figure(figsize=(8,6))
#
#     models = df["Model"].unique()
#
#     for model in models:
#         print(model)
#
#         parts = model.split("_")
#
#         airfoil = parts[0].lower()
#         method = parts[1]
#
#         # legend名称
#         if airfoil == "naca":
#             airfoil_name = "naca0012"
#         elif airfoil == "rae":
#             airfoil_name = "rae2822"
#         else:
#             airfoil_name = airfoil
#
#         label = f"{method}_{airfoil_name}"
#
#         color = color_map.get(airfoil, "black")
#         linestyle = style_map.get(method, "-")
#
#         data = df[df["Model"] == model][metric]
#
#         sns.kdeplot(
#             data=data,
#             linewidth=2,
#             linestyle=linestyle,
#             color=color,
#             label=label
#         )
#
#     plt.xlabel(metric)
#     plt.ylabel("Density")
#
#     plt.title(f"{metric} Density Distribution")
#
#     plt.legend()
#
#     plt.tight_layout()
#
#     plt.savefig(f"{metric}_density.png", dpi=600)
#
#     # plt.show()




plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文黑体
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False        # 负号正常显示


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

# ---------------------- 4 指标 ----------------------
metrics = [
    "CL_error",
    "CD_error"
]

metric_names = {
    "CL_error": "升力系数误差",
    "CD_error": "阻力系数误差"
}

# ---------------------- 5 颜色和线型 ----------------------
style_map = {
    "cDDPM": "--",
    "SC-cDDPM": "-"
}

color_map = {
    "naca": "#1f77b4",
    "rae": "#d62728"
}

# ---------------------- 6 绘制Density ----------------------

for metric in metrics:

    plt.figure(figsize=(8,6))

    models = df["Model"].unique()

    for model in models:

        parts = model.split("_")

        airfoil = parts[0].lower()
        method = parts[1]

        # legend名称
        if airfoil == "naca":
            airfoil_name = "NACA0012"
        elif airfoil == "rae":
            airfoil_name = "RAE2822"
        else:
            airfoil_name = airfoil

        label = f"{method}-{airfoil_name}"

        color = color_map.get(airfoil, "black")
        linestyle = style_map.get(method, "-")

        data = df[df["Model"] == model][metric].values

        kde = gaussian_kde(data, bw_method=0.4)

        xmin = min(data)
        xmax = max(data)

        margin = (xmax - xmin) * 0.1

        x = np.linspace(xmin - margin, xmax + margin, 300)

        y = kde(x)

        plt.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label
        )

    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.title(f"{metric} Density Distribution")


    # plt.xlabel(metric_names[metric])
    # plt.ylabel("概率密度")
    # plt.title(f"{metric_names[metric]}密度分布")


    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{metric}_density_中文.png", dpi=600)

    # plt.show()



