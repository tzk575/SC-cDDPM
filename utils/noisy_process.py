
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import logging
from config.params import parse_args
from utils.tools import get_cuda
from utils.roughness import vector_airfoil, angle_between_vectors, angle_all
from utils.airfoil_draw import airfoil_plot
from model.DDPM import GaussianDiffusion
from model.UNet_1D import UNet



params = parse_args()
path = '../data/train_data.npy'
data = np.load(path)
data = torch.tensor(data, dtype=torch.float32)
#newdata = new_dataset(data, num_interval)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=True)

gaussian = GaussianDiffusion(params.noise_factor, params.timesteps)


batch_size = params.batch_size
timesteps = params.timesteps
noise_factor_list = params.noise_factor
boundary_point = params.boundary_point



for step, data in enumerate(dataloader):
    airfoil = get_cuda(data[0])
    x_start = airfoil[0]
    t = get_cuda(torch.randint(0, timesteps, (1,))).long()
    print(x_start.size())

    mask = torch.zeros_like(x_start)
    boundary_point1, boundary_point2 = boundary_point

    mask_head = mask[:boundary_point1] = 1
    mask_front = mask[boundary_point1:boundary_point2] = 1
    mask_middle = mask[boundary_point2:] = 1

    noise_factor1, noise_factor2, noise_factor3 = noise_factor_list
    noise_middle = torch.randn_like(x_start) * noise_factor1
    noise_front = torch.randn_like(x_start) * noise_factor2
    noise_head = torch.randn_like(x_start) * noise_factor3

    noise = noise_middle * mask_middle + noise_front * mask_front + noise_head * mask_head

    q_sample = gaussian.q_sample
    x_noisy = q_sample(x_start, t, noise)


    print(x_noisy.size())

    airfoil_plot(x_start, True)
    airfoil_plot(x_noisy, True)


    break












