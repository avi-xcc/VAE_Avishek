import os.path
from time import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ColorJitter, RandomHorizontalFlip, RandomRotation

from data import CustomImageDataset
from vae import vae_loss, cVAE

transformations = Compose([
    Resize((32, 32)),
    ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    RandomRotation(degrees=(-15, 15)),
    RandomHorizontalFlip()
])

dataset = CustomImageDataset(img_dir=r"D:\PythonProjects\AgeGenderRace_CNN\archive\UTKFace", transforms=transformations)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

network = cVAE(shape=(3, 32, 32), nClass1=117, nClass2=2, nClass3=5, nhid=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
network.to(device)

save_name = "models/cVAE.pt"

if os.path.exists(save_name):
    network.load_state_dict(torch.load(save_name))

lr = 1e-3
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
epochs = 2500

for epoch in range(epochs):
    train_loss = 0.0
    rcn_loss = 0.0
    kldiv_loss = 0.0
    n = 0
    start_time = time()

    for images, label1, label2, label3 in dataloader:
        # print(images.shape)
        # np_images = images.numpy().transpose((0, 2, 3, 1))
        # for i in range(len(np_images)):
        #     plt.subplot(4, 4, i+1)
        #     plt.imshow(np_images[i])
        # plt.show()
        x_hat, mu, logsig = network(images.to(device), label1.to(device), label2.to(device), label3.to(device))

        l, rcn_lss, kldiv_lss = vae_loss(images.to(device), x_hat, mu, logsig)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        rcn_loss += rcn_lss.cpu().item()
        kldiv_loss += kldiv_lss.cpu().item()
        n += images.shape[0]

    train_loss /= n
    print(f"Epoch {epoch}: Training loss: {train_loss:.4f}; Reconstruction: {rcn_loss / n:.4f},"
          f"KL Div: {kldiv_loss / n:.4f}; Time: {time() - start_time:.2f} sec")
    torch.save(network.state_dict(), save_name)

