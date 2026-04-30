import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from config.params import parse_args

from tqdm import tqdm

from utils.tools import setup_logging  # Ensure you have these functions in model_tools.py
from utils.airfoil_draw import save_images, save_images_conditional, save_images_organized
from model.UNet import UNet_conditional, EMA  # Ensure these are in modules.py

import logging

from data.airfoil_dataset import AirfoilDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import matplotlib.pyplot as plt

from model.ddpm import Diffusion

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):

    # 日志和设备设置
    # 数据管道（数据集和数据加载器）
    # 神经网络模型（UNet_conditional）
    # 优化器和学习率调度器
    # 损失函数
    # 扩散过程
    # 监控系统
    # EMA机制  辅助

    setup_logging(args.run_name)
    device = args.device
    cache_file = args.cache_file

    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points, cache_file=cache_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet_conditional(c_in=1, c_out=1, cond_dim=args.cond_dim, time_dim=64, base_dim=16).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=True)   #学习率调整器

    mse = nn.MSELoss(reduction='none')
    l1 = nn.L1Loss()

    diffusion = Diffusion(num_points_per_side=args.num_airfoil_points, device=device)

    logger = SummaryWriter(os.path.join("logs/runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)


    airfoil_x = dataset.get_x()

    training_loss = []

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        print(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader) #实时训练进度条
        epoch_loss = 0

        for i, airfoil in enumerate(pbar):
            train_coords = airfoil['train_coords_y'].to(device).float().unsqueeze(1)    #翼型y坐标
            cl = airfoil['CL'].to(device).float().unsqueeze(1)  #也就应该有batch_size个数据
            cd = airfoil['CD'].to(device)


            # t = torch.randint(low=1, high=self.noise_steps, size=(n,))
            t = diffusion.sample_timesteps(train_coords.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_coords, t)    #前向加噪过程
            # train_coords：原始清晰翼型
            # x_t：在时间步t的噪声版本翼型
            # noise：添加的实际噪声（这是模型需要预测的目标）

            if np.random.random() < 0.1:
                cl = None
            predicted_noise = model(x_t, t, cl)
            #loss = mse(noise, predicted_noise)     #完全平方（a-b）^2
            loss = l1(noise, predicted_noise)       #绝对值


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            epoch_loss += loss.item()


            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=loss.item(), learning_rate=current_lr)
            logger.add_scalar(f"loss: {epoch}", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)



        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}")





        if epoch % 100 == 0:

            cl_condition_nums = 10  # CL条件的个数
            generated_nums = 6  # 每个条件下生成的翼型个数

            cl_values = torch.linspace(-0.2, 1.5, cl_condition_nums).to(device)  # (cl_condition_nums,)
            cl = cl_values.repeat_interleave(generated_nums).unsqueeze(1)  # (cl_condition_nums * generated_nums, 1)
            total_samples = cl_condition_nums * generated_nums
            print(f'生成设置: {cl_condition_nums}个CL条件 × {generated_nums}个翼型 = {total_samples}个翼型')
            print(f'CL范围: {cl_values[0]:.3f} 到 {cl_values[-1]:.3f}')

            sampled_images = diffusion.sample(model, n=total_samples, conditioning=cl)
            ema_sampled_images = diffusion.sample(ema_model, n=total_samples, conditioning=cl)  # Use cl instead of labels
            print(f'Sampled images shape: {sampled_images.shape}')

            #plot_images(sampled_images)
            save_images(sampled_images, airfoil_x, os.path.join("logs/results", args.run_name, f"save_images_{epoch}.jpg"))
            save_images_conditional(sampled_images, airfoil_x, os.path.join("logs/results", args.run_name, f"{epoch}.jpg"), cl)
            save_images_conditional(ema_sampled_images, airfoil_x, os.path.join("logs/results", args.run_name, f"{epoch}_ema.jpg"), cl)
            save_images_organized(sampled_images, airfoil_x, os.path.join("logs/results", args.run_name, f"save_images_{epoch}.jpg"),
                                  cl, cl_condition_nums, generated_nums)


            torch.save(model.state_dict(), os.path.join("logs/models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("logs/models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("logs/models", args.run_name, f"optim.pt"))




    # save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join("logs/results", args.run_name, "training_loss.jpg"))
    #wandb.save(os.path.join("results", args.run_name, "training_loss.jpg"))



def launch():

    args = parse_args()

    train(args)




if __name__ == '__main__':
    launch()



