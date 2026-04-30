#这个文件是用来生成翼型的   最后得到的是.npz文件
#通过save_all_airfoils(airfoils, airfoil_x)来保存的


import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from airfoil_dataset_1d_2channel import AirfoilDataset
from torch.utils.data import DataLoader

from LucidDiffusion import *
import aerosandbox as asb
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

from tqdm import tqdm
from utils_1d_2channel import *


def save_all_airfoils(airfoils, airfoil_x, save_path='airfoils_data_rae2822'):


    all_smooth = []

    for i in range(airfoils.shape[0]):
        y_upper = airfoils[i, 0].cpu().detach().numpy()
        y_lower = airfoils[i, 1].cpu().detach().numpy()

        # 光滑处理
        y_smooth = smooth_airfoil(y_lower, y_upper, airfoil_x, s=0.1)
        if torch.is_tensor(y_smooth):
            y_smooth = y_smooth.cpu().numpy()

        all_smooth.append(y_smooth)

    # 转换为numpy数组
    all_smooth = np.array(all_smooth)

    # 保存
    np.save(save_path, all_smooth)
    print(f"✅ 保存完成: {save_path}, 形状: {all_smooth.shape}")

    return all_smooth.shape


def load_all_airfoils(save_path='airfoils_data.npz'):
    """加载所有翼型数据"""
    data = np.load(save_path)
    return data['x'], data['y']





# load uiuc airfoils
uiuc_dict = load_uiuc_airfoils()
print(f'uiuc dict keys: {uiuc_dict.keys()}')
# print max and min for cl, max thickness and max camber
print(f"max cl: {uiuc_dict['uiuc_max_cl']}, min cl: {uiuc_dict['uiuc_min_cl']}")
print(f"max thickness: {uiuc_dict['uiuc_max_thickness']}, max camber: {uiuc_dict['uiuc_max_camber']}")
print(f'min thickness: {uiuc_dict["uiuc_min_thickness"]}, min camber: {uiuc_dict["uiuc_min_camber"]}')


generated_num = 1024

# cl = np.linspace(0.0, 0.0, generated_num, dtype=np.float32)
# cd = np.linspace(0.00, 0.0054, generated_num, dtype=np.float32)
# thickness_list = np.linspace(0.005, 0.25, generated_num, dtype=np.float32)
# camber_list = np.linspace(0.000, 0.1, generated_num, dtype=np.float32)

cl = np.full(generated_num, 0.2235, dtype=np.float32)
cd = np.full(generated_num, 0.00552, dtype=np.float32)
thickness_list = np.full(generated_num, 0.121, dtype=np.float32)
camber_list = np.full(generated_num, 0.0126, dtype=np.float32)



airfoil_path = 'coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
airfoil_x = dataset.get_x()

# load the models
thickness_camber_path = 'models/lucid_thickness_camber_standardized_run_3/best_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_model = Unet1DConditional(32, cond_dim=4, channels=2, dim_mults=(1,2,4)).to(device)
all_model.load_state_dict(torch.load(thickness_camber_path, weights_only=True))
all_model.eval()
thickness_camber_model_diffusion = GaussianDiffusion1D(all_model, seq_length=100).to(device)


model_type = 'all'
plot = True
pca = False
tsne = False
if model_type == 'all':
    thickness_batch = torch.tensor(thickness_list).to(device)
    camber_batch = torch.tensor(camber_list).to(device)
    cl_batch = torch.tensor(cl).to(device)
    cd_batch = torch.tensor(cd).to(device)

    uiuc_min_thickness = torch.tensor(uiuc_dict['uiuc_min_thickness']).to(device)
    uiuc_max_thickness = torch.tensor(uiuc_dict['uiuc_max_thickness']).to(device)
    thickness_batch = normalize_conditioning_values(thickness_batch, uiuc_min_thickness, uiuc_max_thickness)
    thickness_batch = thickness_batch + 2

    uiuc_min_camber = torch.tensor(uiuc_dict['uiuc_min_camber']).to(device)
    uiuc_max_camber = torch.tensor(uiuc_dict['uiuc_max_camber']).to(device)
    camber_batch = normalize_conditioning_values(camber_batch, uiuc_min_camber, uiuc_max_camber)
    camber_batch = camber_batch + 2

    uiuc_min_cl = torch.tensor(uiuc_dict['uiuc_min_cl']).to(device)
    uiuc_max_cl = torch.tensor(uiuc_dict['uiuc_max_cl']).to(device)
    cl_batch = normalize_conditioning_values(cl_batch, uiuc_min_cl, uiuc_max_cl)
    cl_batch = cl_batch + 2

    uiuc_min_cd = torch.tensor(uiuc_dict['uiuc_min_cd']).to(device)
    uiuc_max_cd = torch.tensor(uiuc_dict['uiuc_max_cd']).to(device)
    cd_batch = normalize_conditioning_values(cd_batch, uiuc_min_cd, uiuc_max_cd)
    cd_batch = cd_batch + 2

    conditioning = torch.cat([cl_batch.unsqueeze(1), cd_batch.unsqueeze(1), thickness_batch.unsqueeze(1), camber_batch.unsqueeze(1)], dim=1).to(device)

    print(conditioning)
    airfoils = thickness_camber_model_diffusion.sample(batch_size=len(thickness_list), conditioning=conditioning)

    save_all_airfoils(airfoils, airfoil_x)


    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()

        print('翼型坐标点的一些规格数据')

        for i, ax in enumerate(axs):
            y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[i, 1].cpu().detach().numpy()
            print(f'数据的种类 = {type(y_coord_lower)}, size = {y_coord_lower.shape}')

            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, airfoil_x, s=0.1)
            print(f'光滑后  数据的种类 = {type(y_coords)}, size = {y_coords.shape}')

            ax.plot(airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f'cl_{cl[i]:.3f}&cd_{cd[i]:.3f}&thickness_{thickness_list[i]:.3f}&camber_{camber_list[i]:.3f}')
        plt.show()

    plt.tight_layout()

    main_plot_path = 'cl&cd_thickness_camber_conditioned_airfoils.png'
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')

    plt.close()

    if pca:
        pca_1 = PCA(n_components=2)
        airfoil_matrix = airfoils.view(len(thickness_list), -1).cpu().detach().numpy()
        transformed_data = pca_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        ax.set_title('PCA')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()
    if tsne:
        tsne_1 = TSNE(n_components=2)
        transformed_data = tsne_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        ax.set_title('TSNE')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()
