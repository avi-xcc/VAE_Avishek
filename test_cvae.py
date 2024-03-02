import matplotlib.pyplot as plt
import torch
import numpy as np
from vae import cVAE


@torch.no_grad()
def generate_images():
    network = cVAE(shape=(3, 32, 32), nClass1=117, nClass2=2, nClass3=5, nhid=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    save_name = "models/cVAE.pt"

    network.load_state_dict(torch.load(save_name))
    network.eval()
    images = network.generate(age=[22], gender=[1], race=[4])
    np_images = images.cpu().detach().numpy().transpose((0, 2, 3, 1))

    for i in range(len(np_images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np_images[i])
    plt.show()


if __name__ == '__main__':
    generate_images()
