import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文黑体
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False        # 负号正常显示


#
# # ---------------------- 1 读取数据 ----------------------
# df = pd.read_csv("all_errors.csv")
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
# # ---------------------- 4 需要绘制的指标 ----------------------
# metrics = [
#     "shape_rmse",
#     "thickness_error",
#     "camber_error",
#     "CL_error",
#     "CD_error"
# ]
#
# # ---------------------- 5 绘制箱型图 ----------------------
# for metric in metrics:
#
#     plt.figure(figsize=(8,6))
#
#     sns.boxplot(
#         x="Model",
#         y=metric,
#         data=df,
#         palette="Set2",
#         width=0.6,
#         showfliers=False
#     )
#
#     plt.title(f"{metric} Distribution")
#     plt.xlabel("Model")
#     plt.ylabel(metric)
#
#     plt.tight_layout()
#
#     # 论文图
#     plt.savefig(f"{metric}_boxplot.png", dpi=600)
#
#     # plt.show()



# ---------------------- 字体设置 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 读取数据 ----------------------
df = pd.read_csv("all_errors.csv")

df["Model"] = df["Model"].str.replace("only_Diffusion", "cDDPM", regex=False)
df["Model"] = df["Model"].str.replace("LLM_Diffusion", "SC-cDDPM", regex=False)

metrics = [
    "shape_rmse",
    "thickness_error",
    "camber_error",
    "CL_error",
    "CD_error"
]

# metric_names = {
#     "shape_rmse": "翼型形状重建误差",
#     "thickness_error": "厚度误差",
#     "camber_error": "弯度误差",
#     "CL_error": "升力系数误差",
#     "CD_error": "阻力系数误差"
# }

metric_names = {
    "shape_rmse": "Shape_RMSE",
    "thickness_error": "Thickness_Error",
    "camber_error": "Camber_Error",
    "CL_error": "CL_Error",
    "CD_error": "CL_Error"
}



models = df["Model"].unique()

# 颜色（基本和seaborn set2接近）
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})


for metric in metrics:

    plt.figure(figsize=(8,6))

    data = []
    for model in models:
        data.append(df[df["Model"] == model][metric].values)

    box = plt.boxplot(
        data,
        labels=models,
        patch_artist=True,
        showfliers=False
    )

    # 设置颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title(f"{metric_names[metric]} distribution")
    plt.xlabel("Model")

    # plt.title(f"{metric_names[metric]}分布")
    # plt.xlabel("模型种类")

    plt.ylabel(metric_names[metric])

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{metric}_boxplot.png", dpi=600)

    # plt.show()


