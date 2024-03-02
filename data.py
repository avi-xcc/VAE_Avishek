from torch.utils.data import Dataset
from torchvision.io import read_image
import os


def extract_label_from_name(img_name):
    labels = img_name.split("_")
    if len(labels) < 4:
        return 0, 0, 0
    return int(labels[0]), int(labels[1]), int(labels[2])


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transform = transforms
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_names[item])
        image = read_image(img_path)
        label1, label2, label3 = extract_label_from_name(self.img_names[item])

        if self.transform:
            image = self.transform(image)

        return image.float() / 255., label1, label2, label3
