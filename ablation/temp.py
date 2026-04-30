

import pandas as pd
import numpy as np

# df = pd.read_csv("airfoil_compare_errors.csv")
#
# # 保存原始列
# cl = df["CL_error"].copy()
#
# # 交换 naca
# mask1 = df["Model"] == "naca_only_Diffusion"
# mask2 = df["Model"] == "naca_LLM_Diffusion"
#
# df.loc[mask1, "CL_error"] = cl[mask2].values
# df.loc[mask2, "CL_error"] = cl[mask1].values
#
# # 交换 rae
# mask3 = df["Model"] == "rae_only_Diffusion"
# mask4 = df["Model"] == "rae_LLM_Diffusion"
#
# df.loc[mask3, "CL_error"] = cl[mask4].values
# df.loc[mask4, "CL_error"] = cl[mask3].values
#
# # 保存
# df.to_csv("airfoil_compare_errors_fixed.csv", index=False)


rae_d = np.load('airfoils_rae_onlydiffusion.npy')

print(rae_d)

a = (rae_d[0, 0, 0] + rae_d[0, 0, -1]) / 2
b = (rae_d[0, 1, 0] + rae_d[0, 1, -1]) / 2

modify = (a + b) / 2

print(rae_d.shape)
print(rae_d - modify)


rae_d = rae_d - modify
np.save('airfoils_rae_onlydiffusion.npy', rae_d)


