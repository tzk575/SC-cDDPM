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

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from get_prompt import get_condition_from_text
from LLM.airfoil_LLM import AirfoilLLMTester


def save_all_airfoils(airfoils, airfoil_x, save_path='airfoils_data_cd'):

    modify_number = airfoils[0, 0, 0].detach().cpu().numpy()
    all_smooth = []

    for i in range(airfoils.shape[0]):
        y_upper = airfoils[i, 0].detach().cpu().numpy() - modify_number
        y_lower = airfoils[i, 1].detach().cpu().numpy() - modify_number

        # 光滑处理    在utils_1d_2channel中
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



prompt3 = """
    翼型数量：32
    相关参数需求：
    cl：=0.00
    cd：=0.00535
    thickness：=0.1204
    弯度：=0.0
    其他需求：
    模仿NACA0012。
    """


# prompt3 = """
#     翼型数量：1024
#     相关参数需求：
#     cl：[0.0, 0.03]
#     cd：[0.0, 0.001]
#     thickness：[0.00,0.2]
#     弯度：= [0.0, 0.02]
#     其他需求：
#     模仿uiuc翼型库。
#     """



# load uiuc airfoils
uiuc_dict = load_uiuc_airfoils()
# print(f'uiuc dict keys: {uiuc_dict.keys()}')
# # print max and min for cl, max thickness and max camber
# print(f"max cl: {uiuc_dict['uiuc_max_cl']}, min cl: {uiuc_dict['uiuc_min_cl']}")
# print(f"max thickness: {uiuc_dict['uiuc_max_thickness']}, max camber: {uiuc_dict['uiuc_max_camber']}")
# print(f'min thickness: {uiuc_dict["uiuc_min_thickness"]}, min camber: {uiuc_dict["uiuc_min_camber"]}')

cl, cd, thickness_list, camber_list, condition = get_condition_from_text(prompt3)
print(cl, cd, thickness_list, camber_list, condition)
print(cl.shape)
print('---0' * 10)



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

# 加载大模型部分

MODEL_NAME = "./qwen3b-airfoil-final"

_text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
if _text_tokenizer.pad_token is None:
    _text_tokenizer.pad_token = _text_tokenizer.eos_token

_text_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
_text_model.eval()

print("加载 Qwen 用于文本嵌入...")


@torch.no_grad()
def get_text_embed(model, tokenizer, texts):
    """
    texts: List[str] 或单个 str
    return: Tensor [B, H]  (和训练时保持一致)
    """
    if isinstance(texts, str):
        texts = [texts]

    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False
        )
        for t in texts
    ]

    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states[-2]  # [B, T, H]

    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    pooled = F.layer_norm(pooled, pooled.shape[-1:])
    return pooled  # [B, H]





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

    conditioning_num = torch.cat([cl_batch.unsqueeze(1), cd_batch.unsqueeze(1),
                              thickness_batch.unsqueeze(1),
                              camber_batch.unsqueeze(1)], dim=1).to(device)
    print(conditioning_num)


    with torch.no_grad():
        airfoils = thickness_camber_model_diffusion.sample(batch_size=len(thickness_list),
                                                           conditioning=conditioning_num)


    save_all_airfoils(airfoils, airfoil_x)
    #smooth方法在utils_1d_2channel中

    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()

        print('翼型坐标点的一些规格数据')

        for i, ax in enumerate(axs):
            y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[i, 1].cpu().detach().numpy()

            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, airfoil_x, s=0.1)

            ax.plot(airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f'cl={cl[i]:.3f}\n'
                         f'cd={cd[i]:.5f}\n'
                         f'thickness={thickness_list[i]:.3f}\n'
                         f'camber={camber_list[i]:.3f}',
                         fontsize=10, loc='left')
        plt.show()

    plt.tight_layout()

    main_plot_path = 'cl&cd_thickness_camber_conditioned_airfoils.png'
    #plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')

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
