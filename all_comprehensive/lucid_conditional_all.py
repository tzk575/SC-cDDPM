import os
import copy
import numpy as np
import json
import random as rnd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils_1d_2channel import * 
import logging
from airfoil_dataset_1d_2channel import AirfoilDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import math
import matplotlib.pyplot as plt
import aerosandbox as asb
from LucidDiffusion import *
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def normalize_conditioning_values(conditioning_values, min_values, max_values):
    # Normalize the conditioning values
    normalized_conditioning_values = (conditioning_values - min_values) / (max_values - min_values)
    return normalized_conditioning_values

def standardize_conditioning_values(conditioning_values, mean_values, std_values):
    # Standardize the conditioning values
    standardized_conditioning_values = (conditioning_values - mean_values) / std_values
    return standardized_conditioning_values

def save_images_conditional(airfoils,airfoil_x, path, conditioning, num_cols=4):
    # input tensor cl is cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device) convert to numpy
    thickness = conditioning[:,0].cpu().numpy()
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()

    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu()
        y_coords = torch.cat([airfoil[0], airfoil[1]])
        ax.scatter(airfoil_x, y_coords, color='black')
        max_thickness_string = f'max thickness={thickness[i]:.2f}'
        ax.set_title(f'Airfoil {i+1}, \n {max_thickness_string}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(num_airfoils, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


MODEL_NAME = "./qwen3b-airfoil-final"

print("加载 Qwen 用于文本嵌入...")
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


@torch.no_grad()
def get_text_embed(model, tokenizer, texts):
    """
    texts: List[str] 或单个 str
    return: Tensor [B, H]
    """

    if isinstance(texts, str):
        texts = [texts]

    # ---- 使用 Chat 模板（重点）----
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False
        )
        for t in texts
    ]

    # 批量 tokenize
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(model.device)

    # 前向 + hidden_states
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hs = outputs.hidden_states[-2]  # [B, T, H]

    # attention mask pooling
    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # 可选：归一化
    pooled = F.layer_norm(pooled, pooled.shape[-1:])

    return pooled  # [B, H]

# load uiuc airfoil data
uiuc_path = 'uiuc_airfoils.pkl'

with open(uiuc_path, 'rb') as f:
    uiuc_data = pickle.load(f)

    uiuc_cl_mean = uiuc_data['uiuc_cl_mean']
    uiuc_cl_std = uiuc_data['uiuc_cl_std']
    uiuc_cd_mean = uiuc_data['uiuc_cd_mean']
    uiuc_cd_std = uiuc_data['uiuc_cd_std']

    uiuc_max_cl = uiuc_data['uiuc_max_cl']
    uiuc_max_cd = uiuc_data['uiuc_max_cd']
    uiuc_min_cl = uiuc_data['uiuc_min_cl']
    uiuc_min_cd = uiuc_data['uiuc_min_cd']

    uiuc_max_thickness_mean = uiuc_data['uiuc_max_thickness_mean']
    uiuc_max_thickness_std = uiuc_data['uiuc_max_thickness_std']
    uiuc_max_camber_mean = uiuc_data['uiuc_max_camber_mean']
    uiuc_max_camber_std = uiuc_data['uiuc_max_camber_std']

    uiuc_max_thickness = uiuc_data['uiuc_max_thickness']
    uiuc_min_thickness = uiuc_data['uiuc_min_thickness']
    uiuc_max_camber = uiuc_data['uiuc_max_camber']
    uiuc_min_camber = uiuc_data['uiuc_min_camber']

text_map = {}

with open('airfoil_llm_training.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        text_map.setdefault(obj['source_airfoil'], []).append(obj['instruction'])



def train(args):
    setup_logging(args.run_name)
    device = args.device

    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)

    #print(222, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Unet1DConditional(32, cond_dim=4, dim_mults=(1, 2, 4), channels=2, dropout=0.2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10)
    l1 = nn.L1Loss()
    diffusion = GaussianDiffusion1D(model, seq_length=args.num_airfoil_points, objective='pred_noise', timesteps=1000).to(device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    #print(111111, l)


    airfoil_x = dataset.get_x()

    training_loss = []

    best_loss = float('inf')
    patience = 200
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, airfoil in enumerate(pbar):
            train_coords = airfoil['train_coords_y'].to(device).float()
            cl = airfoil['CL'].to(device).float().unsqueeze(1)
            cd = airfoil['CD'].to(device).float().unsqueeze(1)
            max_thickness = airfoil['max_thickness'].to(device).float().unsqueeze(1)
            max_camber = airfoil['max_camber'].to(device).float().unsqueeze(1)

            # normalize the conditioning
            cl = normalize_conditioning_values(cl, uiuc_min_cl.to(device), uiuc_max_cl.to(device))
            cd = normalize_conditioning_values(cd, uiuc_min_cd.to(device), uiuc_max_cd.to(device))
            max_thickness = normalize_conditioning_values(max_thickness, uiuc_min_thickness.to(device), uiuc_max_thickness.to(device))
            max_camber = normalize_conditioning_values(max_camber, uiuc_min_camber.to(device), uiuc_max_camber.to(device))

            # shift mean away from 0
            cl = cl + 2
            cd = cd + 2
            max_thickness = max_thickness + 2
            max_camber = max_camber + 2

            print(cl.shape)
            conditioning_num = torch.cat([cl, cd, max_thickness, max_camber], dim=1).float()
            print(conditioning_num.shape, 789789798)


            t = torch.randint(0, diffusion.num_timesteps, (train_coords.shape[0],), device=device).long()
            noise = torch.randn_like(train_coords, device=device)
            x_t = diffusion.q_sample(train_coords, t, noise=noise)

            if torch.rand(1).item() < .2:
                max_thickness = None
                max_camber = None

            predicted_noise = diffusion.model(x_t, t, conditioning_num)
            error = l1(noise, predicted_noise)

            # Optionally calculate CL error if CL is provided
            '''
            if cl is not None:
                gen_cl = get_cl_values(x_t, t, model, diffusion, vae, cl, airfoil_x).to(device)
                cl_error += l1(cl.squeeze(1), gen_cl)
            '''

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            epoch_loss += error.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=error.item(), learning_rate=current_lr)

            logger.add_scalar(f"loss: {epoch}", error.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join("models", args.run_name, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epoch} epochs.")
            break

        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}, epochs no improvement: {epochs_no_improve}, best loss: {best_loss}")

        if epoch % 100 == 0:

            max_thickness = torch.linspace(0, 0.4, 5).unsqueeze(1).to(device)
            max_camber = torch.linspace(0, .2, 5).unsqueeze(1).to(device)
            cl = torch.zeros_like(max_thickness)
            cd = torch.ones_like(max_thickness) * 0.00539

            max_thickness = standardize_conditioning_values(max_thickness, uiuc_max_thickness_mean, uiuc_max_thickness_std)
            max_camber = standardize_conditioning_values(max_camber, uiuc_max_camber_mean, uiuc_max_camber_std)
            cl = standardize_conditioning_values(cl, uiuc_cl_mean, uiuc_cl_std)
            cd = standardize_conditioning_values(cd, uiuc_cd_mean, uiuc_cd_std)

            max_thickness = max_thickness + 2
            max_camber = max_camber + 2
            cl = cl + 2
            cd = cd + 2

            conditioning_num = torch.cat([cl, cd, max_thickness, max_camber], dim=1)
            sampled_images = diffusion.sample(batch_size=5, conditioning_num=conditioning_num)
            save_images_conditional(sampled_images, airfoil_x, os.path.join("results", args.run_name, f"{epoch}.jpg"), max_thickness)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))



    # Save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    loss_curve_path = os.path.join("results", args.run_name, "training_loss.jpg")
    plt.savefig(loss_curve_path)




def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_thickness_camber_standardized_run_3")
    parser.add_argument('--epochs', type=int, default=301)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()


