import matplotlib.pyplot as plt
import torch

from vae import VAE


@torch.no_grad()
def generate_images():
    network = VAE((3, 32, 32), nhid=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    save_name = "models/VAE.pt"

    network.load_state_dict(torch.load(save_name))

    images = network.generate(batch_size=16)
    np_images = images.cpu().detach().numpy().transpose((0, 2, 3, 1))

    for i in range(len(np_images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np_images[i])
    plt.show()


if __name__ == '__main__':
    generate_images()
