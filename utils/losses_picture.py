
import numpy as np
import matplotlib.pyplot as plt
import torch



def plot_losses(losses, name):

    #losses_numpy = [loss.detach().cpu().item() for loss in losses]

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Losses', marker='o', markersize=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')

    #plt.savefig(name)

    plt.show()
    plt.close()