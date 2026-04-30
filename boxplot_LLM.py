import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False



# ---------------------- 1 读取数据 ----------------------
df = pd.read_csv("airfoil_compare_errors.csv")

# ---------------------- 2 修改模型名称 ----------------------
df["Model"] = df["Model"].str.replace("only_Diffusion", "cDDPM", regex=False)
df["Model"] = df["Model"].str.replace("LLM_Diffusion", "SC-cDDPM", regex=False)

# ---------------------- 3 绘图风格 ----------------------
sns.set_style("whitegrid")



plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18


# ---------------------- 4 指标 ----------------------
# metrics = [
#     "shape_rmse",
#     "thickness_error",
#     "camber_error",
#     "CL_error",
#     "CD_error"
# ]

metrics = [
    "翼型形状重建误差",
    "厚度误差",
    "弯度误差",
    "升力系数误差",
    "阻力系数误差"
]

# ---------------------- 5 分组 ----------------------
groups = {
    "NACA": df[df["Model"].str.contains("naca")].copy(),
    "RAE": df[df["Model"].str.contains("rae")].copy()
}

# ---------------------- 6 每组单独绘图 ----------------------
for group_name, group_df in groups.items():

    # 去掉前缀（只保留模型名）
    group_df["Model"] = group_df["Model"].str.replace("naca_", "", regex=False)
    group_df["Model"] = group_df["Model"].str.replace("rae_", "", regex=False)

    for metric in metrics:

        plt.figure(figsize=(8,6))

        sns.boxplot(
            x="Model",
            y=metric,
            data=group_df,
            palette="Set2",
            width=0.6,
            showfliers=False
        )

        # plt.title(f"{metric} Distribution ({group_name})")
        # plt.xlabel("Model")

        plt.title(f"{metric} 分布 ({group_name})")
        plt.xlabel("模型种类")

        plt.ylabel(metric)

        plt.tight_layout()

        # 如果需要保存
        plt.savefig(f"{group_name}_{metric}_boxplot.png", dpi=600)

        # plt.show()