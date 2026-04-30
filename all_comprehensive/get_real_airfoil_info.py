
import aerosandbox as asb
from airfoil_dataset_1d_2channel import AirfoilDataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# 1. 读取内置翼型（不需要 .dat 文件）
airfoil = asb.Airfoil("NACA0012")   # 也可以 "RAE2822" 等

print("Name:", airfoil.name)

# 2. 坐标（上、下表面 + 全部点）
coords = airfoil.coordinates            # shape: (N, 2)
x = coords[:, 0]
y = coords[:, 1]

print("坐标点数:", coords.shape[0])
print("前 5 个点:\n", coords[:5])


max_thickness = airfoil.max_thickness()
max_camber = airfoil.max_camber()
TE_thickness = airfoil.TE_thickness()
TE_angle = airfoil.TE_angle()

print('------------------------------')
print("最大厚度:", max_thickness)
print("最大弯度:", max_camber)
print("尾缘厚度:", TE_thickness)
print("尾缘开口角(弧度):", TE_angle)


coef = airfoil.get_aero_from_neuralfoil(
    alpha=0,     # 攻角，单位：deg
    Re=1e6,      # 雷诺数
    mach=0.0     # 马赫数
)

CL = coef["CL"][0]
CD = coef["CD"][0]
CM = coef["CM"][0]

print("CL =", CL)
print("CD =", CD)
print("CM =", CM)


